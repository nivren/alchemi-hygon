# SPDX-License-Identifier: Apache-2.0
"""Small offline Git fixtures for Upstream Delta V1; no product/DCU imports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("upstream_delta", Path(__file__).with_name("upstream_delta.py"))
delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delta)


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null",
         "-c", "user.name=Delta Fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, stderr=subprocess.PIPE, text=True,
    ).strip()


def write(repo, path, value):
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value)


def commit(repo, subject):
    git(repo, "add", ".")  # Isolated temporary fixture only.
    git(repo, "commit", "-m", subject)
    return git(repo, "rev-parse", "HEAD")


def snapshot(files):
    return {p: {"kind": "blob", "mode": "100644", "oid": "fixture", "data": s.encode()}
            for p, s in files.items()}


class StaticTests(unittest.TestCase):
    def test_signatures_returns_fields_and_exports(self):
        before = snapshot({"pkg/core.py": "class C:\n    size: int = 1\n    def __init__(self, a, /, *, b=2): pass\ndef f(x=1) -> int: return x\n",
                           "pkg/__init__.py": "from .core import f\n__all__ = ['f']\n"})
        after = snapshot({"pkg/core.py": "class C:\n    size: int = 2\n    def __init__(self, a, /, *, b=3): pass\ndef f(x=4) -> str: return str(x)\n",
                          "pkg/__init__.py": "from .core import C\n__all__ = ['C']\n"})
        changes = {v["symbol"]: v for v in delta.api_changes(before, after, "pkg")["changes"]}
        self.assertIn("pkg/core.py:C.size", changes)
        self.assertIn("b=3", changes["pkg/core.py:C.__init__"]["after"][0])
        self.assertIn("/", changes["pkg/core.py:C.__init__"]["after"][0])
        self.assertIn("-> str", changes["pkg/core.py:f"]["after"][0])
        self.assertIn("pkg/__init__.py:exports", changes)

    def test_same_signature_body_and_parse_failure(self):
        before = snapshot({"pkg/a.py": "def f(x): return x\n"})
        after = snapshot({"pkg/a.py": "def f(x): return x + 1\n", "pkg/broken.py": "def ("})
        result = delta.api_changes(before, after, "pkg")
        self.assertEqual(result["body_changes_with_same_signature"], ["pkg/a.py:f"])
        self.assertEqual(result["parse_failures"]["candidate"][0]["path"], "pkg/broken.py")

    def test_version_alias_no_execution_and_runtime_import_not_export(self):
        files = snapshot({"pkg/__init__.py": "version='0.2.0'\n__version__=version\nraise RuntimeError('must not execute')\ndef load():\n    import torch\n"})
        self.assertEqual(delta.version(files, "pkg"), "0.2.0")
        inventory, _ = delta.api_inventory(files, "pkg")
        self.assertNotIn("torch", inventory["pkg/__init__.py:exports"][0]["signature"])

    def test_markers_and_uv_git_source_preserved(self):
        files = snapshot({"pyproject.toml": '[project]\nrequires-python=">=3.11"\ndependencies=["numpy>=2; python_version > \'3.10\'"]\n[project.optional-dependencies]\ntorch=["torch>=2.10"]\n[tool.uv.sources]\nops={git="https://example.invalid/ops",rev="rc"}\n'})
        dep = delta.dependencies(files)
        self.assertIn("python_version", dep["dependencies"][0])
        self.assertEqual(dep["optional-dependencies"]["torch"], ["torch>=2.10"])
        self.assertEqual(dep["uv"]["sources"]["ops"]["rev"], "rc")

    def test_patch_and_ubf_indices_do_not_copy_private_urls(self):
        text = "| LP-001 | `packages/framework/pkg/a.py`；`packages/ops/pkg/b.py` | http://private.invalid |\n"
        text += "| LP-002 | `nvalchemi/core.py`；`nvalchemiops/core.py` | description |\n"
        index = delta.patch_index(text)
        self.assertEqual(index[("framework", "pkg/a.py")], ["LP-001"])
        self.assertEqual(index[("framework", "nvalchemi/core.py")], ["LP-002"])
        self.assertEqual(index[("ops", "nvalchemiops/core.py")], ["LP-002"])
        self.assertNotIn("private.invalid", str(index))
        self.assertEqual(delta.bugfix_index("## UBF-001：Test\n")[0]["id"], "UBF-001")

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.yaml"
            path.write_text('{"status": "Candidate", "status": "Baseline"}')
            with self.assertRaises(delta.DeltaError):
                delta.load_input(path)


class GitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="alchemi-delta-fixture-")
        cls.root = Path(cls.temp.name)
        cls.repos, cls.refs, cls.lock, cls.candidate = {}, {}, {"repositories": {}}, {
            "status": "Candidate", "observed_date": "2026-10-09", "repositories": {}}
        product = cls.root / "product"
        product.mkdir()
        git(product, "init", "--initial-branch=main")
        for key, name, namespace in (("framework", "nvalchemi-toolkit", "nvalchemi"),
                                     ("ops", "nvalchemi-toolkit-ops", "nvalchemiops")):
            repo = cls.root / key
            repo.mkdir()
            git(repo, "init", "--initial-branch=main")
            write(repo, namespace + "/__init__.py", "__version__='0.1.0'\nfrom .core import f\n")
            write(repo, namespace + "/core.py", "def f(x=1) -> int:\n    return x\n")
            write(repo, namespace + "/deleted.py", "def old(): return None\n")
            write(repo, "rename file\tname.py", "# unchanged content\n" * 20)
            write(repo, "pyproject.toml", '[project]\nrequires-python=">=3.11"\ndependencies=["numpy"]\n')
            common = commit(repo, "common")
            write(repo, "baseline-only.txt", "baseline-only\n")
            baseline = commit(repo, "baseline")
            git(repo, "checkout", "-b", "candidate", common)
            write(repo, namespace + "/core.py", "def f(x=2) -> str:\n    return str(x)\n")
            (repo / namespace / "deleted.py").unlink()
            write(repo, namespace + "/added.py", "def new(): return 1\n")
            git(repo, "mv", "rename file\tname.py", "renamed.py")
            target = commit(repo, "candidate")
            cls.repos[key], cls.refs[key] = repo, (baseline, target, common)
            url = "https://example.invalid/" + name + ".git"
            cls.lock["repositories"][name] = {"url": url, "commit": baseline, "import_path": "packages/" + key}
            cls.candidate["repositories"][name] = {"url": url, "commit": target, "observed_ref": "candidate"}
            git(product, "fetch", str(repo), baseline)
            for path, entry in delta.Git(repo).snapshot(baseline).items():
                write(product, "packages/" + key + "/" + path, entry["data"].decode())
            write(product, "packages/" + key + "/" + namespace + "/core.py", "def f(x=88) -> int: return x\n")
        write(product, "docs/UPSTREAM_LOCK.yaml", json.dumps(cls.lock))
        write(product, "docs/UPSTREAM.md", "| LP-001 | `packages/framework/nvalchemi/core.py` |\n")
        write(product, "docs/UPSTREAM_BUGFIXES.md", "## UBF-001：Fixture\n")
        cls.candidate["local_commit"] = commit(product, "local patch")
        cls.product = product
        write(product, "docs/UPSTREAM_CANDIDATES.yaml", json.dumps(cls.candidate))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def report(self):
        return delta.build_report(self.product, self.repos["framework"], self.repos["ops"],
                                  self.product / "docs/UPSTREAM_LOCK.yaml",
                                  self.product / "docs/UPSTREAM_CANDIDATES.yaml")

    def test_diverged_and_ancestor_history(self):
        baseline, target, common = self.refs["framework"]
        repo = delta.Git(self.repos["framework"])
        history = repo.history(baseline, target)
        self.assertEqual(history["relation"], "diverged")
        self.assertEqual((history["baseline_only_count"], history["candidate_only_count"]), (1, 1))
        self.assertEqual(repo.history(common, target)["relation"], "baseline_ancestor")
        self.assertEqual(repo.history(target, common)["relation"], "candidate_ancestor")
        self.assertEqual(repo.history(common, common)["relation"], "identical")

    def test_no_common_ancestor(self):
        other = self.root / "unrelated"
        other.mkdir(exist_ok=True)
        git(other, "init", "--initial-branch=main")
        write(other, "only.txt", "unrelated")
        ref = commit(other, "unrelated")
        git(other, "fetch", str(self.repos["framework"]), self.refs["framework"][0])
        self.assertEqual(delta.Git(other).history(ref, self.refs["framework"][0])["relation"], "unrelated")

    def test_rename_delete_add_and_potential_overlap(self):
        report = self.report()["repositories"]["framework"]
        statuses = {c["status"][0] for c in report["files"]}
        self.assertTrue({"R", "A", "D", "M"} <= statuses)
        renamed = next(c for c in report["files"] if c["status"].startswith("R"))
        self.assertEqual(renamed["before"], "rename file\tname.py")
        conflict = report["potential_conflicts"][0]
        self.assertEqual(conflict["path"], "nvalchemi/core.py")
        self.assertEqual(conflict["patches"], ["LP-001"])

    def test_exact_sha_and_missing_object_rejected(self):
        repo = delta.Git(self.repos["framework"])
        for ref in ("candidate", "abcd1234", "0" * 40):
            with self.subTest(ref=ref), self.assertRaises(delta.DeltaError):
                repo.commit(ref)

    def test_shallow_history_rejected(self):
        repo = self.root / "shallow"
        git(self.root, "clone", "--depth=1", self.repos["framework"].as_uri(), str(repo))
        with self.assertRaises(delta.DeltaError):
            delta.Git(repo).commit(git(repo, "rev-parse", "HEAD"))

    def test_bugfix_hint_requires_fix_ancestry(self):
        candidate = json.loads(json.dumps(self.candidate))
        base, target, common = self.refs["ops"]
        for fix, expected in ((common, True), (target, True), (base, False)):
            candidate["bugfix_hints"] = [{"id": "UBF-001", "repository": "ops", "fix_commit": fix,
                                           "upstream_pr": "https://example.invalid/pull/1", "review_scope": "unknown; retain"}]
            path = self.root / "hint.yaml"
            path.write_text(json.dumps(candidate))
            result = delta.build_report(self.product, self.repos["framework"], self.repos["ops"],
                                        self.product / "docs/UPSTREAM_LOCK.yaml", path)
            self.assertEqual(result["bugfixes"][0]["included"], expected)

    def test_cli_deterministic_and_only_outputs_written(self):
        files_before = {p.relative_to(self.product): p.read_bytes() for p in self.product.rglob("*")
                        if p.is_file() and ".git" not in p.parts and "reports" not in p.parts}
        argv = ["--root", str(self.product), "--framework-repo", str(self.repos["framework"]),
                "--ops-repo", str(self.repos["ops"]), "--output", str(self.product / "reports/delta.md")]
        refs_before = {key: git(repo, "show-ref") for key, repo in self.repos.items()}
        self.assertEqual(delta.main(argv), 0)
        first = (self.product / "reports/delta.md").read_bytes()
        self.assertEqual(delta.main(argv), 0)
        self.assertEqual(first, (self.product / "reports/delta.md").read_bytes())
        for path, value in files_before.items():
            self.assertEqual((self.product / path).read_bytes(), value)
        self.assertEqual(refs_before, {key: git(repo, "show-ref") for key, repo in self.repos.items()})

    def test_protected_and_outside_outputs_rejected(self):
        for dest in (self.product / "packages/framework/no.py", self.product / "docs/UPSTREAM_LOCK.yaml",
                     self.product / "external/no.md", self.root / "outside.md"):
            with self.subTest(dest=dest), self.assertRaises((delta.DeltaError, ValueError)):
                delta.output_path(self.product, dest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
