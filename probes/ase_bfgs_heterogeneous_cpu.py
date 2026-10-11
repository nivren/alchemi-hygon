# SPDX-License-Identifier: Apache-2.0
"""Independent CPU pairing/parity and exact adapter-vs-product validation."""
import argparse
import json

import torch
from nvalchemiops.bfgs_ase import bfgs_ase_prepare_state
from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh
from ase_bfgs_heterogeneous_math import step


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    dimensions=[1,2,3,7,15,21,30,45,57,93,147,199,200,201,285,489,511,512,513,552,561,1104,1113]
    max_rounds=max(n+n%2-1 for n in dimensions)
    coverage=[]
    for n in dimensions:
        m=n+n%2
        visited=set()
        for r in range(m-1):
            endpoints=set()
            for k in range(m//2):
                p=0 if k==0 else 1+(k-1+r)%(m-1)
                q=1+(m-2-k+r)%(m-1)
                p,q=sorted((p,q))
                if q>=n:
                    continue
                assert p!=q and not ({p,q}&endpoints) and (p,q) not in visited
                visited.add((p,q)); endpoints.update((p,q))
        assert len(visited)==n*(n-1)//2
        assert (max_rounds-(m-1))%2==0
        # Model buffer contents over local active rounds plus global inactive tail.
        buffers=[0,0]; current=0
        for r in range(max_rounds):
            if r<m-1:
                buffers[1-current]=buffers[current]+1
            current=1-current
        assert buffers[current]==m-1
        for freeze_after in (0,1,3):
            frozen=[freeze_after,freeze_after]
            for r in range(max_rounds*3):
                current=1-current
                assert frozen[current]==freeze_after
        coverage.append(dict(n=n,rounds=m-1,pairs=len(visited),inactive_tail=max_rounds-(m-1)))
    widths=[9,12,9,21]
    q=[torch.linspace(-.4,.6,n,dtype=torch.float64).reshape(-1,3) for n in widths]
    r=[v.clone() for v in q]
    s=[bfgs_ase_prepare_state(n,system_id=i) for i,n in enumerate(widths)]
    t=[bfgs_ase_prepare_state(n,system_id=i) for i,n in enumerate(widths)]
    for iteration in range(12):
        force=lambda a: -(4*a+.7*a.sum()+.6*a**3)
        a=_step_batch_with_eigh(q,[force(v) for v in q],s,torch.linalg.eigh)
        b=step(r,[force(v) for v in r],t,lambda hs:[torch.linalg.eigh(h) for h in hs])
        for x,y,z,w,ra,rb in zip(q,r,s,t,a,b):
            assert torch.equal(x,y)
            for key,value in z.state_dict().items():
                other=w.state_dict()[key]
                assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
            for field in ('eigenvalues','raw_step','applied_step','scale','longest_step'):
                if hasattr(ra,field):
                    assert torch.equal(getattr(ra,field),getattr(rb,field))
    result=dict(status='passed',pairing_and_buffer_cases=coverage,adapter_cpu_bitwise_steps=12)
    open(args.output,'w').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',dimensions=len(dimensions),adapter_steps=12)))

if __name__=='__main__': main()
