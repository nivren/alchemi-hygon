# SPDX-License-Identifier: Apache-2.0
"""Process-only HIP adapter injection and HCU framework lifecycle/CPU oracle gates."""
import argparse
import json
import sys
from pathlib import Path

import torch
from ase_bfgs_heterogeneous import Scheduler
from ase_bfgs_heterogeneous_math import step
from nvalchemiops import bfgs_ase_batch_reference as math
from nvalchemiops import bfgs_ase_hip as hip


class Injection:
    def __init__(self,scheduler):
        self.scheduler=scheduler
        self.original_math=math._step_batch_with_eigh
        self.original_single=hip._eigh
        self.shapes=[]
    def grouped(self,hs):
        self.shapes.append([list(h.shape) for h in hs])
        return self.scheduler(hs)
    def batch(self,q,f,s,eigh):
        if q[0].device.type=='cuda' and eigh is self.single:
            return step(q,f,s,self.grouped)
        return self.original_math(q,f,s,eigh)
    def single(self,h,UPLO='L'):
        if UPLO!='L': raise ValueError('LOWER only')
        return self.scheduler([h])[0]
    def __enter__(self):
        # Save bound method identity once, as product batch dispatch receives it.
        self.single=self.single
        hip._eigh=self.single; math._step_batch_with_eigh=self.batch
        return self
    def __exit__(self,*_):
        hip._eigh=self.original_single; math._step_batch_with_eigh=self.original_math


def gates(args):
    sys.path.insert(0,str(Path('packages/framework').resolve()))
    from test.compatibility.test_ase_bfgs_batch import mixed,BatchCell,BatchPolynomial,Dataset,check_single_restart_promotion
    from test.compatibility.test_ase_bfgs_cell import assert_snapshot_equal
    from nvalchemi.data import Batch
    from nvalchemi.dynamics import ASECompatibleBFGS,ASECompatibleBFGSVariableCell,FusedStage
    from nvalchemi.dynamics.sampler import SizeAwareSampler
    from nvalchemi.dynamics.sinks import HostMemory
    rows=[]
    for variant in ('serial','streams2','streams4','native'):
        with Scheduler(variant) as scheduler, Injection(scheduler) as injected:
            for variable,options in ((False,{}),(True,{}),(True,dict(cell_factor=2.7,scalar_pressure=.037)),(True,dict(mask=[1,0,1,0,1,0]))):
                cls=ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
                model_cls=BatchCell if variable else BatchPolynomial
                cpu=mixed(variable); gpu=cpu.clone().to('cuda')
                reference=cls(model_cls(),fmax=1e-10,**options)
                candidate=cls(model_cls(),backend='hip',fmax=1e-10,**options)
                alternate=torch.cuda.Stream(); alternate.wait_stream(torch.cuda.current_stream())
                snapshots=[]
                for iteration in range(7):
                    reference.step(cpu)
                    with torch.cuda.stream(alternate):
                        candidate.step(gpu); done=torch.cuda.Event(); done.record()
                    torch.cuda.current_stream().wait_event(done)
                    torch.testing.assert_close(gpu.positions.cpu(),cpu.positions,atol=5e-12,rtol=5e-12)
                    if variable: torch.testing.assert_close(gpu.cell.cpu(),cpu.cell,atol=5e-12,rtol=5e-12)
                    for identifier in reference.bfgs_states:
                        torch.testing.assert_close(candidate.bfgs_states[identifier].hessian.cpu(),reference.bfgs_states[identifier].hessian,atol=6e-10,rtol=6e-11)
                assert candidate.model.calls==reference.model.calls==8
                snapshot=candidate.state_dict(); paired=gpu.clone()
                resumed=cls(model_cls(),backend='hip',fmax=1e-10,**options); resumed.load_state_dict(snapshot)
                candidate.step(gpu); resumed.step(paired)
                assert torch.equal(gpu.positions,paired.positions)
                if variable: assert torch.equal(gpu.cell,paired.cell)
                old=candidate.bfgs_states[12].state_dict(); saved=gpu.clone()
                candidate._masked_pre_update(gpu,torch.tensor([True,False,True],device='cuda'))
                assert_snapshot_equal(old,candidate.bfgs_states[12].state_dict())
                assert torch.equal(gpu.positions[2:5],saved.positions[2:5])
                if variable: assert torch.equal(gpu.cell[1],saved.cell[1])
                before={i:s.state_dict() for i,s in candidate.bfgs_states.items()}
                reordered=gpu.index_select(torch.tensor([2,0,1],device='cuda'))
                candidate._ensure_state_initialized(reordered)
                for i,s in candidate.bfgs_states.items(): assert_snapshot_equal(before[i],s.state_dict())
                data=reordered.to_data_list(); data[2].system_id.fill_(90)
                replaced=Batch.from_data_list(data[:1]+data[2:])
                survivor=candidate.bfgs_states[13]; candidate._ensure_state_initialized(replaced)
                assert set(candidate.bfgs_states)=={13,90} and candidate.bfgs_states[13] is survivor
                assert candidate.bfgs_states[90].step_count==0
                rows.append(dict(variant=variant,variable=variable,options=options,steps=7,
                                 cpu_reference=True,restart_bitwise=True,nondefault_stream=True,
                                 freeze_reorder_remove_arrival=True,shared_model_calls=8))
            for variable in (False,True):
                check_single_restart_promotion(variable,device='cuda',backend='hip')
                cls=ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
                model=BatchCell() if variable else BatchPolynomial()
                d=cls(model,backend='hip',fmax=1e-10,n_steps=3,device_type='cuda')
                sink=HostMemory(capacity=4)
                dataset=Dataset(variable)
                dataset.rows=[row.to('cuda') for row in dataset.rows]
                stage=FusedStage([(0,d)],sampler=SizeAwareSampler(dataset,max_atoms=6,max_batch_size=2),sinks=[sink],refill_frequency=1,device_type='cuda')
                assert stage.run(n_steps=20) is None
                assert len(sink)==4 and sorted(sink.read().source_index.flatten().tolist())==list(range(4))
                assert model.calls==stage.step_count+2 and not d.bfgs_states
                rows.append(dict(variant=variant,variable=variable,single_restart_promotion=True,
                                 native_sampler_budgeted_refill=True,shared_model_calls=model.calls))
            assert injected.shapes, 'joint adapter was not actually used'
        print(json.dumps(dict(variant=variant,status='passed',framework_cases=6)),flush=True)
    args.output.write_text(json.dumps(dict(status='passed',cases=rows),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); gates(p.parse_args())
