"""Read-only AST/dependency inventory of locked reference clones."""
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
result = {}
for name, namespace in [('nvalchemi-toolkit', 'nvalchemi'), ('nvalchemi-toolkit-ops', 'nvalchemiops')]:
    root = ROOT / 'external' / name
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    modules = {}
    for path in files:
        if path.startswith(namespace + '/') and path.endswith('.py'):
            tree = ast.parse((root / path).read_text())
            modules[path] = {
                'symbols': [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))],
                'imports': sorted({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}),
            }
    result[name] = {'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                    'modules': modules, 'tests': [p for p in files if p.startswith('test/')],
                    'examples': [p for p in files if p.startswith('examples/')],
                    'configs': [p for p in files if p.endswith(('.yaml', '.yml', '.toml'))],
                    'licenses': [p for p in files if any(s in p.lower() for s in ['license', 'notice'])]}
# Resolve every explicit framework import from ops using static module/export declarations.
ops = ROOT / 'external/nvalchemi-toolkit-ops'
imports = []
for path in result['nvalchemi-toolkit']['modules']:
    tree = ast.parse((ROOT / 'external/nvalchemi-toolkit' / path).read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or '').startswith('nvalchemiops'):
            base = ops.joinpath(*n.module.split('.'))
            module = base.with_suffix('.py') if base.with_suffix('.py').exists() else base / '__init__.py'
            exported = set()
            if module.exists():
                for node in ast.walk(ast.parse(module.read_text())):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)): exported.add(node.name)
                    if isinstance(node, (ast.Import, ast.ImportFrom)): exported.update(a.asname or a.name for a in node.names)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store): exported.add(node.id)
            for alias in n.names:
                ok = alias.name in exported or (base / (alias.name + '.py')).exists() or (base / alias.name / '__init__.py').exists()
                imports.append({'caller': path, 'module': n.module, 'symbol': alias.name, 'static_resolved': ok})
result['cross_package_imports'] = imports
out = ROOT / 'reports/upstream_inventory.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'counts': {k: {f: len(v[f]) for f in ['modules','tests','examples','configs']} for k,v in result.items() if k != 'cross_package_imports'}, 'imports': len(imports), 'unresolved': [i for i in imports if not i['static_resolved']]}, indent=2))
