# SPDX-License-Identifier: Apache-2.0
"""Sanitized hipprof window evidence: distinct-queue overlap and API copy sizes."""
import argparse
import collections
import json
import re
import sqlite3

p=argparse.ArgumentParser(); p.add_argument('--database',required=True); p.add_argument('--metadata',required=True); p.add_argument('--output',required=True); a=p.parse_args()
r=json.load(open(a.metadata)); row=r['cases'][0] if r.get('cases') else r; lo,hi=row['window_unix_ns']
c=sqlite3.connect(a.database)
tables=[r[0] for r in c.execute("select name from sqlite_master where type='table'")]
names={str(i):name for i,name in c.execute('select STR_ID,STR_NAME from STR_TABLE where TYPE=6')}
kernels=[]; apis=[]
for t in tables:
    if t.startswith('HIPOPS_'):
        kernels += list(c.execute('select BeginNs,EndNs,queue_id,Name from "'+t+'" where BeginNs>=? and EndNs<=?',(lo,hi)))
    if t.startswith('HIP_'):
        apis += list(c.execute('select BeginNs,EndNs,Name,args from "'+t+'" where BeginNs>=? and EndNs<=?',(lo,hi)))
events=[]; byname=collections.Counter(); queues=collections.Counter()
for begin,end,queue,name in kernels:
    events.extend([(begin,1,queue),(end,-1,queue)])
    byname[names.get(str(name),str(name))]+=end-begin
    queues[queue]+=end-begin
active=collections.Counter(); union=0; overlap=0; max_queues=0; last=lo
for timestamp,change,queue in sorted(events):
    count=sum(v>0 for v in active.values())
    if count: union+=timestamp-last
    if count>=2: overlap+=timestamp-last
    active[queue]+=change; last=timestamp
    max_queues=max(max_queues,sum(v>0 for v in active.values()))
copies=collections.defaultdict(list)
for begin,end,name,args in apis:
    text=args or ''
    size=re.search(r'sizeBytes\s*=\s*(\d+)',text); kind=re.search(r'kind\s*=\s*(\d+)',text)
    if size and kind: copies[kind.group(1)].append(int(size.group(1)))
result=dict(status='analyzed',variant=row.get('variant','candidate'),scope=row['scope'],window_ms=(hi-lo)/1e6,
            kernel_count=len(kernels),kernel_union_ms=union/1e6,
            kernel_sum_ms=sum(end-begin for begin,end,*_ in kernels)/1e6,
            simultaneous_distinct_queue_ms=overlap/1e6,max_simultaneous_kernel_queues=max_queues,
            queue_kernel_sum_ms=sorted(v/1e6 for v in queues.values()),
            top_kernels=[dict(name=k,sum_ms=v/1e6) for k,v in byname.most_common(10)],
            hip_api_copy_sizes_by_kind={k:dict(count=len(v),total_bytes=sum(v),sizes=sorted(set(v))) for k,v in copies.items()},
            copy_kind_contract={'1':'H2D','2':'D2H','3':'D2D'},
            private_identifiers_omitted=True)
open(a.output,'w').write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
