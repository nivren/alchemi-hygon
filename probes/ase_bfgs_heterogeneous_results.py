# SPDX-License-Identifier: Apache-2.0
"""Aggregate raw independent runs without replacing or trimming any samples."""
import argparse
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--directory',type=Path,required=True); a=p.parse_args()
root=a.directory; runs=[json.loads((root/f'benchmark-{i}.json').read_text()) for i in (1,2,3)]
result=dict(status='passed',processes=3,warmup=2,repeats_per_process=7,cases=[])
for i,scope in enumerate(('li','t2','mixed')):
    row=dict(scope=scope,shapes=runs[0]['cases'][i]['shapes'])
    for field in ('checked_solver','complete_math'):
        section={}
        for variant in ('rocsolver','serial','streams2','streams4','native'):
            samples=[v for run in runs for v in run['cases'][i][field]['wall_ms'][variant]['samples']]
            section[variant]=dict(samples=samples,median_ms=statistics.median(samples),
                                  process_medians_ms=[run['cases'][i][field]['wall_ms'][variant]['median'] for run in runs],
                                  min_ms=min(samples),max_ms=max(samples),stdev_ms=statistics.pstdev(samples))
        for variant,data in section.items():
            data['reduction_vs_serial_pct']=100*(1-data['median_ms']/section['serial']['median_ms'])
            data['reduction_vs_rocsolver_pct']=100*(1-data['median_ms']/section['rocsolver']['median_ms'])
        row[field]=section
    row['peak_extra_torch_allocator_bytes']=runs[0]['cases'][i]['peak_extra_torch_allocator_bytes']
    result['cases'].append(row)
result['framework'] = json.loads((root/'framework-hcu.json').read_text())
result['resource_observations']={}
for path in root.glob('*-exit.json'):
    r=json.loads(path.read_text())
    if 'resource_observations' in r:
        counts=[sum("HCU Index: ['4']" in block for block in obs['output'].split('PID: ')[1:]) for obs in r['resource_observations']]
        result['resource_observations'][path.name]=dict(exit_code=r['exit_code'],count=len(counts),max_observed_allocated_hcu4_processes=max(counts,default=0),elapsed_s=r['elapsed_s'])
result['profiler']=[json.loads(path.read_text()) for path in sorted(root.glob('trace-*-summary.json'))]
result['real_workflow']=[]
for scope in ('li','t2'):
    for variant in ('serial','streams2'):
        paths=sorted(root.glob(f'{scope}-{variant}-workflow-*.json'))
        paths=[path for path in paths if '-exit' not in path.stem]
        if not paths: continue
        reports=[json.loads(path.read_text()) for path in paths]
        samples=[row['seconds']*1000 for r in reports for row in r['warm_complete_update_samples']]
        result['real_workflow'].append(dict(scope=scope,variant=variant,runs=len(reports),samples=samples,
            median_ms=statistics.median(samples),model_evaluations=[r['model_evaluations'] for r in reports],
            measured_updates=[len(r['warm_complete_update_samples']) for r in reports],
            checked_updates=[r['checked_updates'] for r in reports],
            ase_force_source=reports[0]['ase_force_source'],all_converged=[r['all_converged'] for r in reports],
            errors_max_abs={k:max(r['errors_max_abs'][k] for r in reports) for k in reports[0]['errors_max_abs']}))
(root/'results-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for row in result['cases']:
    print(row['scope'],{k:round(d['median_ms'],3) for k,d in row['checked_solver'].items()},'math',{k:round(d['median_ms'],3) for k,d in row['complete_math'].items()})
for row in result['real_workflow']: print('real',row['scope'],row['variant'],row['median_ms'],row['runs'])
