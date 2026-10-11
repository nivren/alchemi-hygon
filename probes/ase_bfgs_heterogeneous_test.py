# SPDX-License-Identifier: Apache-2.0
"""M5-3B eigensystem, streams/lifetime, ASE/restart and atomic failure gates."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from ase_bfgs_heterogeneous import Scheduler, load
from ase_bfgs_heterogeneous_math import step
from ase_bfgs_jacobi_candidate_test import validate, bfgs_gate
from nvalchemiops.bfgs_ase import BFGSNumericalError, bfgs_ase_prepare_state


def snapshot(qs,states):
    return [q.clone() for q in qs],[s.state_dict() for s in states]


def unchanged(qs,states,saved):
    assert all(torch.equal(q,v) for q,v in zip(qs,saved[0]))
    for state,expected in zip(states,saved[1]):
        for key,value in expected.items():
            actual=state.state_dict()[key]
            assert torch.equal(actual,value) if isinstance(value,torch.Tensor) else actual==value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('correctness','bfgs'),required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--real-oracles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    load()
    cases=torch.load(args.dataset,map_location='cpu',weights_only=True)
    real=torch.load(args.real_oracles,map_location='cpu',weights_only=True)
    result=dict(status='running',mode=args.mode,variants=[])
    for variant in ('serial','streams2','streams4','native'):
        with Scheduler(variant) as scheduler:
            if args.mode=='correctness':
                rows=[]
                # Different spectra, mixed even/odd widths, zero and upper poison
                # share one call; inactive tails and per-matrix freeze are exercised.
                inputs=[c['input'].unsqueeze(0).cuda() for c in cases]
                before=[a.view(torch.int64).clone() for a in inputs]
                values=scheduler.raw(inputs)
                for a,v,ref,old in zip(inputs,values,cases,before):
                    rows.append(dict(n=ref['n'],name=ref['name'],**validate(a,v,[ref])))
                    assert torch.equal(a.view(torch.int64),old)
                    assert all(t.device==a.device and t.is_contiguous() for t in v)
                    assert v[1].data_ptr()!=a.data_ptr()
                if variant!='serial':
                    # Identical reduction/rotation sequence must retain all diagnostics.
                    serial=load()
                    for a,v in zip(inputs,values):
                        expected=serial.eigh(a,100)
                        assert all(torch.equal(x,y) for x,y in zip(v,expected))
                real_rows=[]
                real_inputs=[r['input'].cuda() for r in real]
                for a,v,ref in zip(real_inputs,scheduler.raw(real_inputs),real):
                    real_rows.append(dict(scope=ref['scope'],shape=list(a.shape),**validate(a,v,ref['references'])))
                producer=torch.cuda.Stream()
                consumer=torch.cuda.Stream()
                producer.wait_stream(torch.cuda.current_stream())
                refs=[c for c in cases if c['n'] in (15,7,200,513,561)][:5]
                with torch.cuda.stream(producer):
                    temporary=[c['input'].unsqueeze(0).cuda().clone() for c in refs]
                    # Skip selected-triangle scalar checks here: those would
                    # synchronize the producer and hide a missing worker event.
                    outputs=scheduler.raw(temporary,checked=False)
                    done=torch.cuda.Event(); done.record()
                    del temporary
                    # Allocator reuse on producing stream after input release.
                    junk=[torch.empty((c['n'],c['n']),device='cuda',dtype=torch.float64).fill_(1000) for c in refs]
                consumer.wait_event(done)
                with torch.cuda.stream(consumer):
                    for group in outputs:
                        for tensor in group: tensor.record_stream(consumer)
                    copied=[tuple(t.clone() for t in group) for group in outputs]
                    consumed=torch.cuda.Event(); consumed.record()
                torch.cuda.current_stream().wait_event(consumed)
                for out,ref in zip(copied,refs):
                    validate(ref['input'].unsqueeze(0).cuda(),out,[ref])
                rejected=0
                bad=[[],[torch.ones(2,2)],[None],[inputs[0].float()], [inputs[0].transpose(-1,-2)], [inputs[0].clone().requires_grad_()]]
                for value in bad:
                    try: scheduler.raw(value)
                    except ValueError: rejected+=1
                    else: raise AssertionError('invalid metadata admitted')
                for sweeps in (0,101,True,1.5):
                    try: scheduler.raw(inputs[:1],sweeps)
                    except ValueError: rejected+=1
                    else: raise AssertionError('invalid sweeps admitted')
                poison=real_inputs[0].clone(); poison[0,1,0]=float('nan')
                try: scheduler.raw([poison])
                except BFGSNumericalError: rejected+=1
                else: raise AssertionError('nonfinite LOWER admitted')
                row=dict(variant=variant,cases=rows,real_cases=real_rows,
                         all_five_outputs_serial_bitwise=True if variant!='serial' else None,
                         producer_consumer_early_release_and_reuse=True,rejected_inputs=rejected)
            else:
                # Reuse ASE polynomial/restart oracle unchanged; switch only the
                # dispatch schedule. Deliberate old error injections remain checked.
                import nvalchemiops.bfgs_ase_batch_reference as math
                original=math._step_batch_with_eigh
                def adapter(q,f,s,eigh):
                    grouped=(lambda hs:[eigh(h,UPLO='L') for h in hs]) if eigh.__name__=='fail_eigh' else scheduler
                    return step(q,f,s,grouped)
                math._step_batch_with_eigh=adapter
                path=args.output.with_name(args.output.stem+'-'+variant+'-ase.json')
                try:
                    bfgs_gate(SimpleNamespace(widths=[15,21,30,45,57,93,201,489,189,189,561],output=path))
                finally: math._step_batch_with_eigh=original
                row=json.loads(path.read_text()); row['variant']=variant
                widths=[15,21,201,285]
                qs=[torch.ones(n//3,3,device='cuda',dtype=torch.float64)*.01 for n in widths]
                states=[bfgs_ase_prepare_state(n,device='cuda',system_id=i) for i,n in enumerate(widths)]
                random=next(c for c in cases if c['n']==285 and c['name']=='random')['input'].cuda()
                states[-1].hessian.copy_(random)
                saved=snapshot(qs,states)
                try: step(qs,[-q for q in qs],states,lambda hs:scheduler(hs,max_sweeps=1))
                except BFGSNumericalError: pass
                else: raise AssertionError('real late-group nonconvergence admitted')
                unchanged(qs,states,saved)
                def invalid_vectors(hs):
                    values=scheduler(hs)
                    values[-1]=(values[-1][0],values[-1][1]*2)
                    return values
                try: step(qs,[-q for q in qs],states,invalid_vectors)
                except BFGSNumericalError: pass
                else: raise AssertionError('late invalid eigensystem admitted')
                unchanged(qs,states,saved)
                row.update(real_late_group_nonconvergence_atomic=True,late_invalid_vectors_atomic=True)
            result['variants'].append(row)
            print(json.dumps(dict(variant=variant,status='passed',mode=args.mode)),flush=True)
    result['status']='passed'
    args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
