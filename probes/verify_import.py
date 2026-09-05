"""Verify exact trees, reachable histories, and absent nested repositories."""
import json
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
lock = json.loads((root/'docs/UPSTREAM_LOCK.yaml').read_text())
results = []
for name, item in lock['repositories'].items():
    path, sha = item['import_path'], item['commit']
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    actual = git('rev-parse',f'HEAD:{path}')
    expected = git('rev-parse',f'{sha}^{{tree}}')
    assert actual == expected, f'{path}: tree differs'
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',sha,'HEAD'],check=True)
    assert not list((root/path).rglob('.git'))
    results.append({'repository':name,'commit':sha,'tree':actual,'identical':True,'ancestor':True})
print(json.dumps(results,indent=2))
