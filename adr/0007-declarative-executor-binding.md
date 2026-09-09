# ADR 0007: declarative executor binding

- Status: accepted
- Date: 2026-09-09
- Scope: B1 implementation metadata, lazy loading and operation dispatch

## Context

`Implementation.executor` was previously metadata-only. Runtime dispatch was spread across
operation-specific `if-elif` branches, so adding an implementation required a catalog edit and
several dispatcher edits. The two declarations could drift without an import-time failure, and
the repeated legacy identifier made semantic changes a 22-site edit. The result was manageable
for the number of operations, but not for the product of operations and backend implementations.

## Decision

- Non-legacy `Implementation` metadata declares a dotted executor module, an `entrypoints` tuple,
  and an `executor_owner` of `ops` or `framework`. The legacy wildcard entry is framework-owned,
  has `executor=None`, and has no entrypoints.
- `load_entrypoint(selection, name, registry=...)` is an operation-neutral function independent of
  the registry object. It validates the pre-resolved selection against metadata, lazy-imports the
  declared module, checks the declared name and callable, and caches the loaded callable. Import
  and load errors include the implementation, operation and `ops-owned`/`framework-owned` package
  context.
- `execute_selected(selection, entrypoint_name, legacy_fn, *args, **kwargs)` is the only generic
  binding adapter. It has one legacy implementation boundary, calls the caller-supplied legacy
  handler for that boundary, and otherwise loads/calls the declared entrypoint. It contains no
  operation dispatch table. Operation-specific legacy behavior remains in a local closure at the
  dispatcher so the adapter cannot become a new central merge hotspot.
- Each entrypoint must have the public ABI of its operation dispatcher. Multiple phase entrypoints
  may share one selection, but each phase has its own documented ABI. Loader tests may verify
  importability; acceptance must additionally call at least one second registered implementation
  through an unchanged dispatcher.
- The process-local callable cache remains intentionally stable for production. A
  `clear_entrypoint_cache()` hook is provided for tests and supported module reload tooling; tests
  that replace `sys.modules` use it to avoid cross-test contamination. Production dispatch does not
  inspect module identity on every step or perform automatic invalidation.
- `backend=None`, `backend="warp"`, legacy Warp behavior and current capability widths remain
  unchanged. Unsupported requests and missing executor packages fail explicitly; there is no
  implicit CPU fallback.

## Consequences

Adding a new implementation now normally changes its catalog metadata and executor module, not a
central operation switch. The dispatcher still owns operation-specific argument adaptation and
legacy closure construction, while the registry owns capability selection and the binding layer
owns lazy executable lookup. This separates capability from execution without starting M2
PlatformFingerprint/Profile/Planner work.

The public operation API is unchanged. The B0-style candidate HCU golden-path gate passed; the
cache-isolation supplement is test infrastructure and does not require a second device run. The
existing feature compatibility widths are not expanded by this ADR.

Evidence: `reports/b1-executor-binding.md`; implementation commits are recorded in
`docs/BACKEND_PLATFORM_PIPELINE_PLAN.md` and `docs/STATUS.md`.
