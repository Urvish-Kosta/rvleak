# Contributing

## Getting set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## The one rule that matters

**A change to the analysis must come with a test that would fail without it.**

Leakage tooling has a nasty failure mode: a broken detector that always fires
looks exactly like a very sensitive one, and a broken detector that never fires
looks like a clean bill of health. Every bug found so far in this project
(documented in the README) was of that shape. Consequently:

- New detection logic needs a **false-positive control** — a case with no
  channel where it must stay silent.
- New attack logic needs a **synthetic ground-truth test**, independent of the
  simulator, where the answer is known by construction.
- A new victim needs a **functional correctness test** against an independent
  reference before any leakage claim about it is accepted.
- If you disable or weaken a step in the pipeline, add the **ablation** as a
  test, so the failure mode stays pinned rather than merely documented.

## Numbers in documentation

No result may be typed into a document by hand. If you want to publish a number,
generate it from `scripts/reproduce.py` and regenerate `docs/results.md`:

```bash
make results
```

CI regenerates results on every push, so a change that alters a published figure
shows up as a diff rather than silently going stale.

## Style

- `ruff check src tests scripts` must pass.
- Comments explain *why*, not *what*. The interesting content in this codebase
  is the reasoning about what a statistic does and does not license you to
  claim; prefer writing that down over restating the code.
- Keep `isa.py` free of timing and leakage concepts. The separation between
  architectural semantics and microarchitectural cost is what makes the
  functional model independently checkable.

## Commit messages

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`,
`build:`, `ci:`, `chore:`. One logical change per commit.
