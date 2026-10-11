# SPDX-License-Identifier: Apache-2.0
"""Real MLIP process-only comparison; unchanged product defaults and public API."""
import argparse
import json
import sys
import time
from pathlib import Path

import ase_bfgs_csp
import torch
from ase_bfgs_heterogeneous import Scheduler
from ase_bfgs_heterogeneous_framework import Injection

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--input-set',choices=('li','t2'),required=True)
p.add_argument('--variant',choices=('rocsolver','serial','streams2','streams4','native'),required=True)
p.add_argument('--measure-warm-updates',type=int,default=None)
a,remaining=p.parse_known_args()
inputs=json.loads(a.manifest.read_text())[a.input_set]
sys.argv=[sys.argv[0],*remaining,'--cifs',*[row['path'] for row in inputs]]
samples=[]
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
original_update=ASECompatibleBFGSVariableCell._masked_pre_update
def timed_update(owner,batch,mask):
    if batch.num_graphs<2 or a.measure_warm_updates is None:
        return original_update(owner,batch,mask)
    counts=[s.step_count for s in owner.bfgs_states.values()]
    warm=bool(counts) and min(counts)>=a.measure_warm_updates
    if not warm:
        return original_update(owner,batch,mask)
    torch.cuda.synchronize(); begin=time.perf_counter()
    result=original_update(owner,batch,mask)
    torch.cuda.synchronize()
    samples.append(dict(before_counts=counts,seconds=time.perf_counter()-begin))
    return result
if a.measure_warm_updates is not None:
    if a.measure_warm_updates<2: raise ValueError('at least two warm updates required')
    ASECompatibleBFGSVariableCell._masked_pre_update=timed_update
with Scheduler(a.variant) as scheduler:
    if a.variant=='rocsolver':
        ase_bfgs_csp.main(); shapes=[]
    else:
        with Injection(scheduler) as injected: ase_bfgs_csp.main()
        shapes=injected.shapes
output=Path(sys.argv[sys.argv.index('--output')+1]); result=json.loads(output.read_text())
result.update(experimental_variant=a.variant,joint_shapes=shapes,probe_only=True,input_set=a.input_set,
              manifest=str(a.manifest),complete_convergence_acceptance=False,
              warm_complete_update_samples=samples,
              warm_complete_update_scope='synchronized Framework generalized pack, Hessian update, joint solver, guards, direction, physical unpack and commit; MLIP and oracle outside',
              excluded_first_updates=a.measure_warm_updates)
if a.measure_warm_updates is not None:
    assert len(samples)>=7, len(samples)
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],input_set=a.input_set,variant=a.variant,joint_calls=len(shapes))),flush=True)
