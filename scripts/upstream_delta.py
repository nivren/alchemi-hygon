#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read frozen Git objects and write an upstream risk report (Python >=3.11).

No fetch, checkout, import of upstream code, merge, or dependency installation.
The .yaml inputs use this repository's JSON-compatible YAML convention.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib


class DeltaError(ValueError):
    """Input or Git evidence is insufficient for a reproducible report."""


class Git:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1", "GIT_NO_REPLACE_OBJECTS": "1"}

    def run(self, *args, input=None, allowed=(0,)):
        result = subprocess.run(
            ["git", "--no-pager", "-C", str(self.path), "-c", "core.fsmonitor=false",
             "-c", "core.hooksPath=/dev/null", *args],
            input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, check=False, timeout=120,
        )
        if result.returncode not in allowed:
            raise DeltaError(f"git {' '.join(args)}: {result.stderr.decode(errors='replace').strip()}")
        return result.stdout

    def text(self, *args, **kwargs):
        return self.run(*args, **kwargs).decode("utf-8").strip()

    def commit(self, sha):
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise DeltaError(f"A complete lowercase SHA is required: {sha}")
        actual = self.text("rev-parse", "--verify", sha + "^{commit}")
        if actual != sha:
            raise DeltaError(f"Commit identity changed: {sha}")
        if self.text("rev-parse", "--is-shallow-repository") != "false":
            raise DeltaError("Shallow repository: acquire full reachable history first")
        return actual

    def snapshot(self, tree):
        entries = {}
        for record in self.run("ls-tree", "-rz", tree).split(b"\0"):
            if not record:
                continue
            meta, path = record.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            entries[path.decode("utf-8")] = {"mode": mode, "kind": kind, "oid": oid}
        blobs = list(dict.fromkeys(v["oid"] for v in entries.values() if v["kind"] == "blob"))
        raw = self.run("cat-file", "--batch", input="".join(s + "\n" for s in blobs).encode())
        data, offset = {}, 0
        for oid in blobs:
            end = raw.index(b"\n", offset)
            header = raw[offset:end].decode().split()
            if len(header) != 3 or header[1] != "blob" or header[0] != oid:
                raise DeltaError(f"Missing Git blob: {oid}")
            size = int(header[2])
            data[oid] = raw[end + 1:end + 1 + size]
            offset = end + size + 2
        for entry in entries.values():
            entry["data"] = data.get(entry["oid"], b"")
        return entries

    def changes(self, before, after):
        tokens = self.run("diff", "--no-ext-diff", "--no-textconv", "--name-status",
                          "-z", "-M", before, after, "--").decode().split("\0")
        changes, i = [], 0
        while i < len(tokens) and tokens[i]:
            status, path = tokens[i:i + 2]
            i += 2
            other = path
            if status[0] in "RC":
                other = tokens[i]
                i += 1
            changes.append({"status": status, "before": None if status[0] == "A" else path,
                            "after": None if status[0] == "D" else other})
        return changes

    def history(self, base, target):
        merge_bases = self.text("merge-base", "--all", base, target, allowed=(0, 1)).splitlines()
        left, right = map(int, self.text("rev-list", "--left-right", "--count", base + "..." + target).split())
        relation = ("identical" if base == target else "baseline_ancestor" if not left
                    else "candidate_ancestor" if not right else "diverged" if merge_bases else "unrelated")
        def commits(spec):
            return [{"commit": row.split("\t", 1)[0], "subject": row.split("\t", 1)[1]}
                    for row in self.text("log", "--format=%H%x09%s", spec).splitlines()]
        return {"relation": relation, "merge_bases": sorted(merge_bases),
                "baseline_only_count": left, "candidate_only_count": right,
                "baseline_only_commits": commits(target + ".." + base),
                "candidate_only_commits": commits(base + ".." + target)}


def decode(entry):
    if entry["mode"] == "120000" or entry["kind"] != "blob":
        raise DeltaError("Symlinks and submodules are not parsed as Python/TOML")
    return entry["data"].decode("utf-8-sig")


def api_inventory(snapshot, namespace):
    inventory, failures = {}, []
    for path, entry in sorted(snapshot.items()):
        if not path.startswith(namespace + "/") or not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(decode(entry), filename=path)
        except (SyntaxError, UnicodeError, DeltaError) as exc:
            failures.append({"path": path, "reason": str(exc)})
            continue

        def add(name, signature, node):
            key = path + ":" + name
            item = {"signature": signature,
                    "body_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}
            inventory.setdefault(key, []).append(item)

        def scan(nodes, parent=""):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_") and node.name != "__init__":
                        continue
                    kind = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                    signature = kind + "(" + ast.unparse(node.args) + ")"
                    signature += " -> " + ast.unparse(node.returns) if node.returns else " -> unknown"
                    if node.decorator_list:
                        signature += " @ " + ", ".join(ast.unparse(d) for d in node.decorator_list)
                    add(parent + node.name, signature, node)
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    signature = "class(" + ", ".join(ast.unparse(b) for b in [*node.bases, *node.keywords]) + ")"
                    if node.decorator_list:
                        signature += " @ " + ", ".join(ast.unparse(d) for d in node.decorator_list)
                    add(parent + node.name, signature, ast.ClassDef(
                        name=node.name, bases=node.bases, keywords=node.keywords,
                        body=[], decorator_list=node.decorator_list))
                    scan(node.body, parent + node.name + ".")
                elif isinstance(node, ast.AnnAssign) and parent and isinstance(node.target, ast.Name):
                    if not node.target.id.startswith("_"):
                        add(parent + node.target.id,
                            ast.unparse(node.annotation) + " = " + (ast.unparse(node.value) if node.value else "required"), node)
                elif isinstance(node, (ast.If, ast.Try, ast.With)):
                    scan(node.body, parent)
                    scan(getattr(node, "orelse", []), parent)
                    scan(getattr(node, "finalbody", []), parent)
                    for handler in getattr(node, "handlers", []):
                        scan(handler.body, parent)
        scan(tree.body)
        if path.endswith("/__init__.py"):
            exports = []
            def module_nodes(nodes):
                for node in nodes:
                    yield node
                    if isinstance(node, (ast.If, ast.Try, ast.With)):
                        yield from module_nodes(node.body)
                        yield from module_nodes(getattr(node, "orelse", []))
                        for handler in getattr(node, "handlers", []):
                            yield from module_nodes(handler.body)
            for node in module_nodes(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        if not name.startswith("_"):
                            exports.append(ast.unparse(node))
                elif isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                        exports.append(ast.unparse(node))
            inventory[path + ":exports"] = [{"signature": " ; ".join(sorted(set(exports))), "body_sha256": ""}]
    return inventory, failures


def api_changes(before, after, namespace):
    a, a_errors = api_inventory(before, namespace)
    b, b_errors = api_inventory(after, namespace)
    changes, body_changes = [], []
    for key in sorted(a.keys() | b.keys()):
        left = [x["signature"] for x in a.get(key, [])]
        right = [x["signature"] for x in b.get(key, [])]
        if left != right:
            changes.append({"symbol": key, "status": "added" if key not in a else "removed" if key not in b else "changed",
                            "before": left, "after": right})
        elif a[key] != b[key]:
            body_changes.append(key)
    return {"changes": changes, "body_changes_with_same_signature": body_changes,
            "parse_failures": {"baseline": a_errors, "candidate": b_errors}}


def dependencies(snapshot):
    if "pyproject.toml" not in snapshot:
        raise DeltaError("Missing pyproject.toml")
    config = tomllib.loads(decode(snapshot["pyproject.toml"]))
    project = config.get("project", {})
    uv = config.get("tool", {}).get("uv", {})
    return {"requires-python": project.get("requires-python"),
            "dependencies": project.get("dependencies", []),
            "optional-dependencies": project.get("optional-dependencies", {}),
            "dependency-groups": config.get("dependency-groups", {}),
            "build-system": config.get("build-system", {}),
            "uv": {k: uv.get(k) for k in ("sources", "dependency-metadata", "override-dependencies", "conflicts")}}


def version(snapshot, namespace):
    constants = {}
    for node in ast.parse(decode(snapshot[namespace + "/__init__.py"])).body:
        if isinstance(node, ast.Assign):
            try:
                value = constants.get(node.value.id, "dynamic/unknown") if isinstance(node.value, ast.Name) else ast.literal_eval(node.value)
            except (ValueError, TypeError):
                value = "dynamic/unknown"
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
            if "__version__" in constants:
                return constants["__version__"]
    return "unknown"


def changed_paths(changes):
    return {c[k] for c in changes for k in ("before", "after") if c[k] is not None}


def patch_index(text):
    result = {}
    for line in text.splitlines():
        match = re.match(r"\|\s*(LP-\d+)\s*\|", line)
        if match:
            # Read only the path column, not prose/test-report links or private URLs.
            paths = line.split("|", 3)[2]
            for path in re.findall(r"packages/(framework|ops)/([^`\s；;|]+)", paths):
                result.setdefault(path, []).append(match.group(1))
            for path in re.findall(r"(?<![\w/])(nvalchemi(?:ops)?/[^`\s；;|]+)", paths):
                key = "ops" if path.startswith("nvalchemiops/") else "framework"
                result.setdefault((key, path), []).append(match.group(1))
    return {k: sorted(set(v)) for k, v in result.items()}


def bugfix_index(text):
    return [{"id": match.group(1), "title": match.group(2).strip()}
            for match in re.finditer(r"^## (UBF-\d+)：(.+)$", text, re.M)]


def analyze_component(upstream, local, baseline, target, local_commit, prefix, namespace, patches):
    upstream.commit(baseline)
    upstream.commit(target)
    local.commit(local_commit)
    # Unsquashed subtree imports retain the baseline commit in the product repository.
    local.commit(baseline)
    local_tree = local_commit + ":" + prefix
    before, after = upstream.snapshot(baseline), upstream.snapshot(target)
    changes = upstream.changes(baseline, target)
    local_changes = local.changes(baseline, local_tree)
    overlap = sorted(changed_paths(changes) & changed_paths(local_changes))
    key = "framework" if namespace == "nvalchemi" else "ops"
    return {"baseline": baseline, "target": target, "local_commit": local_commit, "prefix": prefix,
            "baseline_version": version(before, namespace), "candidate_version": version(after, namespace),
            "history": upstream.history(baseline, target), "files": changes, "local_files": local_changes,
            "potential_conflicts": [{"path": p, "patches": patches.get((key, p), [])} for p in overlap],
            "api": api_changes(before, after, namespace),
            "dependencies": {"baseline": dependencies(before), "candidate": dependencies(after)},
            "local_worktree_changes_excluded": local.text("diff", "--name-only", local_commit, "--", prefix).splitlines()
                + local.text("ls-files", "--others", "--exclude-standard", "--", prefix).splitlines()}


def load_input(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DeltaError(f"Duplicate input key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows] + [""]


def render_report(result):
    lines = ["# Upstream Delta V1 — " + result["observed_date"], "",
             "只读静态分析；比较冻结 Git 对象，未运行候选源码/安装依赖/修改产品或版本锁。",
             "本地产品精确提交：`" + result["local_commit"] + "`。候选状态：Candidate，尚非 Validated。", "",
             "## 方法与复核边界", "",
             "文件变化采用两端 tree diff（包含 rename 两端），历史采用 merge-base 与双侧独有提交。",
             "同时比较基线上游→候选、基线上游→本地产品子树并取交集；交集仅为潜在冲突。",
             "API 为 AST 可见非私有函数/类/方法、构造参数、注解字段及 __init__ 导出；不等于所有正式公共 API。",
             "记录参数、默认值、返回注解、装饰器；动态导出、实际返回值和数值语义须人工复核。",
             "签名不变但 AST 实现变化单独标记。源码不导入、不执行；parse failure 不被当作无变化。",
             "Markdown 包含全部文件/签名变化；可选 JSON 保留每个同签名实现变化及完整依赖声明。", ""]
    lines += table(["分析环境", "记录"], result["analysis_runtime"].items())
    lines += table(["输入记录", "SHA256"], result["input_hashes"].items())
    lines += ["## 摘要与使用入口", ""]
    lines += table(["仓库", "历史关系", "上游文件变化", "本地交集", "AST 接口/导出变化"],
                   ((key, c["history"]["relation"], len(c["files"]), len(c["potential_conflicts"]), len(c["api"]["changes"]))
                    for key, c in result["repositories"].items()))
    lines += ["复现命令与独立 Git 对象获取方式见 [UPSTREAM_SYNC.md](../docs/UPSTREAM_SYNC.md#4-差异分析及补丁审查)。",
              "依赖阻塞、未移植特性与 retain/backport/defer 决策见文末人工审阅；详细清单可展开阅读。", ""]
    for key, component in result["repositories"].items():
        history = component["history"]
        url = component["url"].removesuffix(".git")
        lines += ["## " + key, ""]
        lines += table(["版本", "完整 SHA", "声明版本"], [
            ("Baseline", component["baseline"], component["baseline_version"]),
            ("Candidate (" + component["observed_ref"] + ")", component["target"], component["candidate_version"])])
        lines += ["官方源码：[基线](" + url + "/tree/" + component["baseline"] + ")、[候选]("
                  + url + "/tree/" + component["target"] + ")。", "",
                  f"历史关系：**{history['relation']}**；基线独有 {history['baseline_only_count']} 个提交，"
                  f"候选独有 {history['candidate_only_count']} 个提交。",
                  "merge-base：" + (", ".join(history["merge_bases"]) or "无共同祖先"), "",
                  "### 文件与主要模块", ""]
        counts = Counter(c["status"][0] for c in component["files"])
        lines += [f"上游文件变化 {len(component['files'])}：{dict(sorted(counts.items()))}；"
                  f"本地文件变化 {len(component['local_files'])}；潜在交集 {len(component['potential_conflicts'])}。", ""]
        modules = Counter("/".join((c["after"] or c["before"]).split("/")[:2]) for c in component["files"])
        lines += table(["主要模块/路径前缀", "变化文件"], sorted(modules.items()))
        lines += ["<details><summary>完整上游文件变化（A/M/D/R/T 等 Git 状态）</summary>", ""]
        lines += table(["状态", "之前", "之后"], ((c["status"], c["before"] or "—", c["after"] or "—") for c in component["files"]))
        lines += ["</details>", "", "### 潜在冲突与本地补丁", ""]
        lines += table(["交集路径", "登记 LP"], ((c["path"], ", ".join(c["patches"]) or "无路径匹配，人工复核") for c in component["potential_conflicts"]))
        lines += ["LP 按显式 packages/ 或 nvalchemi/nvalchemiops 路径匹配；简写、通配和语义关联需人工复核。", "",
                  "<details><summary>完整本地变更文件（相对导入基线）</summary>", ""]
        lines += table(["状态", "之前", "之后"], ((c["status"], c["before"] or "—", c["after"] or "—") for c in component["local_files"]))
        lines += ["</details>", "", "### 静态 API 变化", ""]
        api = component["api"]
        lines += [f"{len(api['changes'])} 个 AST 接口/导出条目变化；"
                  f"{len(api['body_changes_with_same_signature'])} 个签名相同的实现变化，实际返回值/行为 unknown，需复核。", ""]
        lines += ["<details><summary>完整静态接口变化（含参数、默认值和返回注解）</summary>", ""]
        lines += table(["状态", "路径:符号", "基线", "候选"],
                       ((c["status"], c["symbol"], " ; ".join(c["before"]) or "—", " ; ".join(c["after"]) or "—") for c in api["changes"]))
        lines += ["</details>", "", "解析失败：" + cell(api["parse_failures"]), "", "### 依赖变化", ""]
        dep = component["dependencies"]
        rows = [(field, dep["baseline"][field], dep["candidate"][field]) for field in ("requires-python", "dependencies")]
        extras = dep["baseline"]["optional-dependencies"].keys() | dep["candidate"]["optional-dependencies"].keys()
        rows += [("extra:" + extra, dep["baseline"]["optional-dependencies"].get(extra, []),
                  dep["candidate"]["optional-dependencies"].get(extra, [])) for extra in sorted(extras)]
        lines += table(["依赖范围（保留 marker）", "基线", "候选"], rows)
        lines += ["uv sources/metadata/override/conflicts、build-system 和开发依赖组完整保留于 JSON；",
                  "以上是声明差异，不是依赖求解或平台兼容性证明。", "",
                  "未纳入对象分析的产品工作树变化：" + cell(component["local_worktree_changes_excluded"]), "",
                  "<details><summary>两侧独有提交（分叉时均需审阅）</summary>", ""]
        for side in ("baseline_only_commits", "candidate_only_commits"):
            lines += ["#### " + side, ""]
            lines += table(["提交", "主题"], (("[" + c["commit"] + "](" + url + "/commit/" + c["commit"] + ")", c["subject"]) for c in history[side]))
        lines += ["</details>", ""]
    lines += ["## 本地 UBF 与官方修复线索", "",
              "本节自动收集现有 UBF 编号，核验人工登记的官方 fix 是否包含于候选；",
              "commit 包含性不证明数学/边界等价，未经 C 审阅的条目 equivalence=unknown、action=retain。", ""]
    lines += table(["UBF", "问题", "官方修复线索", "包含于候选", "当前判断/行动"],
                   ((b["id"], b["title"], ("[PR](" + b["upstream_pr"] + ") " + b["fix_commit"]) if b.get("fix_commit") else "unknown", b.get("included", "unknown"),
                     b.get("review_scope", "unknown；retain，待 C 核验")) for b in result["bugfixes"]))
    lines += ["## 人工审阅：依赖、特性及决策", "",
              "下表来自候选记录的人工审阅输入；工具未从提交主题推断修复等价或 HCU 支持。", ""]
    lines += table(["领域", "源码依据", "评估/未移植范围", "行动"],
                   ((n["area"], " ; ".join(n["evidence"]), n["assessment"], n["action"]) for n in result["review_notes"]))
    lines += ["## 风险与后续", ""]
    for risk in result["risks"]:
        lines += ["- " + risk]
    validation = result.get("tool_validation")
    if validation:
        lines += ["", "## B 工具验证记录", ""]
        lines += table(["字段", "记录"], validation.items())
        lines += ["验证记录由阶段执行后登记；生成报告时不自动运行测试。重复生成字节一致，",
                  "产品源码、版本锁、生产环境与缓存 refs 保持原值；未运行候选 CPU/HCU gate。", ""]
    lines += ["", "B 仅生成差异与风险；C 将建立完整 UBF 等价/额外边界表，D 再确认隔离验证范围。",
              "保留当前稳定基线；backport、补丁退休、整版本升级均须另行验证与确认。", ""]
    return "\n".join(lines)


def build_report(root, framework_repo, ops_repo, lock_path, candidate_path, local_ref=None):
    root = Path(root).resolve()
    lock = load_input(lock_path)
    candidate = load_input(candidate_path)
    if candidate.get("status") != "Candidate":
        raise DeltaError("Expected a Candidate input; no automatic Baseline promotion")
    local = Git(root)
    local_commit = local.commit(local_ref or candidate["local_commit"])
    inputs = {"lock": Path(lock_path), "candidate": Path(candidate_path),
              "patches": root / "docs/UPSTREAM.md", "bugfixes": root / "docs/UPSTREAM_BUGFIXES.md"}
    patches = patch_index(inputs["patches"].read_text())
    result = {"schema_version": 1, "observed_date": candidate["observed_date"], "local_commit": local_commit,
              "input_hashes": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in inputs.items()},
              "repositories": {}, "bugfixes": bugfix_index(inputs["bugfixes"].read_text()),
              "review_notes": candidate.get("review_notes", []), "risks": [],
              "tool_validation": candidate.get("tool_validation")}
    result["input_hashes"]["tool"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["analysis_runtime"] = {"python": platform.python_version(), "git": local.text("--version")}
    specs = [("framework", framework_repo, "nvalchemi-toolkit", "nvalchemi"),
             ("ops", ops_repo, "nvalchemi-toolkit-ops", "nvalchemiops")]
    upstreams = {}
    for key, path, name, namespace in specs:
        base, target = lock["repositories"][name], candidate["repositories"][name]
        if target["url"] != base["url"]:
            raise DeltaError(f"Candidate source URL differs from locked upstream: {name}")
        upstream = upstreams[key] = Git(path)
        entry = analyze_component(upstream, local, base["commit"], target["commit"], local_commit,
                                  base["import_path"], namespace, patches)
        entry.update(url=base["url"], observed_ref=target["observed_ref"])
        result["repositories"][key] = entry
        if entry["history"]["relation"] in ("diverged", "unrelated", "candidate_ancestor"):
            result["risks"].append(key + " 历史关系为 " + entry["history"]["relation"] + "，须先审阅基线独有提交，不直接 subtree pull。")
        if entry["potential_conflicts"]:
            result["risks"].append(key + f" 存在 {len(entry['potential_conflicts'])} 个本地/上游修改交集，需三方源码审阅。")
        if entry["api"]["changes"] or entry["api"]["body_changes_with_same_signature"]:
            result["risks"].append(key + " 静态接口或实现发生变化；动态返回值、语义与平台兼容性尚未验证。")
        if any(entry["api"]["parse_failures"].values()):
            result["risks"].append(key + " 有 Python 解析失败，API 分析不完整。")
        if entry["dependencies"]["baseline"] != entry["dependencies"]["candidate"]:
            result["risks"].append(key + " 依赖声明/开发环境配置改变；未执行依赖求解。")
        if entry["local_worktree_changes_excluded"]:
            result["risks"].append(key + " 存在未纳入冻结提交的工作树变更，见文件清单。")
    hints = {b["id"]: b for b in candidate.get("bugfix_hints", [])}
    for bug in result["bugfixes"]:
        if bug["id"] not in hints:
            continue
        hint = hints[bug["id"]]
        fix = upstreams[hint["repository"]].commit(hint["fix_commit"])
        target = result["repositories"][hint["repository"]]["target"]
        # A related/newer commit is insufficient: require exact fix ancestry.
        bases = upstreams[hint["repository"]].text("merge-base", "--all", fix, target, allowed=(0, 1)).splitlines()
        included = fix in bases
        bug.update(fix_commit=fix, included=included, upstream_pr=hint["upstream_pr"], review_scope=hint["review_scope"])
    return result


def output_path(root, value):
    path = Path(value).resolve()
    relative = path.relative_to(Path(root).resolve())
    if not relative.parts or relative.parts[0] not in ("reports", "artifacts"):
        raise DeltaError("Output must be inside reports/ or artifacts/; product, external, lock and inputs are protected")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--framework-repo", type=Path, required=True)
    parser.add_argument("--ops-repo", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--local-ref", help="Optional complete local product SHA override")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        lock = args.lock or root / "docs/UPSTREAM_LOCK.yaml"
        candidate = args.candidate or root / "docs/UPSTREAM_CANDIDATES.yaml"
        output = output_path(root, args.output)
        json_output = output_path(root, args.json_output) if args.json_output else None
        protected = {p.resolve() for p in (lock, candidate, root / "docs/UPSTREAM.md", root / "docs/UPSTREAM_BUGFIXES.md")}
        if output in protected or json_output in protected or output == json_output:
            raise DeltaError("Report destinations overlap inputs or each other")
        result = build_report(root, args.framework_repo, args.ops_repo, lock, candidate, args.local_ref)
        report = render_report(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        if json_output:
            json_output.parent.mkdir(parents=True, exist_ok=True)
            json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Report: {output}; risks: {len(result['risks'])}; Candidate remains unvalidated")
        return 0
    except (DeltaError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"upstream_delta: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
