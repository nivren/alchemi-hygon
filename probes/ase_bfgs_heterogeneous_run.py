# SPDX-License-Identifier: Apache-2.0
"""Bounded subprocess with resource observations and retained command/exit code."""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--record',type=Path,required=True)
p.add_argument('--timeout',type=int,required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
if a.command[0]=='--': a.command=a.command[1:]
stop=threading.Event(); observations=[]
def observe():
    while not stop.is_set():
        r=subprocess.run(['/opt/hyhal/bin/hy-smi','--showpids'],capture_output=True,text=True,timeout=10)
        observations.append(dict(unix_ns=time.time_ns(),exit_code=r.returncode,output=r.stdout+r.stderr))
        stop.wait(10)
t=threading.Thread(target=observe,daemon=True); t.start()
start=time.time_ns()
try:
    r=subprocess.run(a.command,timeout=a.timeout)
    code=r.returncode
except subprocess.TimeoutExpired: code=124
finally:
    stop.set(); t.join(timeout=12)
a.record.write_text(json.dumps(dict(command=a.command,timeout_s=a.timeout,exit_code=code,
                                    elapsed_s=(time.time_ns()-start)/1e9,resource_observations=observations),indent=2)+'\n')
raise SystemExit(code)
