# Instructions for AI Agents

This file gives AI coding agents (Claude Code, Codex, Cursor, Copilot, Devin, etc.) the rules they must follow when contributing to [Darts](https://github.com/unit8co/darts). It complements [CONTRIBUTING.md](CONTRIBUTING.md), which remains the source of truth for humans. Agents must respect every rule there as well.

If a rule here conflicts with `CONTRIBUTING.md`, follow `CONTRIBUTING.md` and flag the discrepancy in the PR description.

## MUST READ FIRST — CONTRIBUTION GATE (DO NOT SKIP)

**STOP. Do not open or draft a PR until this gate is satisfied.**

For any contribution that might become a PR, verify with the human user:

1. **Issue link:** What is the GitHub issue number for this work? If none exists, the human must open one first.
2. **Maintainer approval:** Does that issue have the **`contributions-welcome`** label applied by a maintainer? If not, the human must comment on the issue and wait for maintainer triage. Do not start implementation for merge-bound work until the label is present.
3. **PR linkage:** The PR description must reference the issue with a recognized keyword (e.g. `Fixes #123` when completing the issue, or `Addresses #123` / `Refs #123` for partial work). Sidebar-only links do not satisfy automation.

This gate is mandatory for all agents **unless the user has write access to the repository** (maintainer bypass below).

### Maintainer bypass

If the authenticated user has **write**, **maintain**, or **admin** access to `unit8co/darts`, they may skip the contribution gate for their own work. Team members manage their own priorities and do not need to wait for `contributions-welcome` on internal PRs.

## Zero-spam and PR authorization

- **Require an issue:** Do not create a pull request unless there is an existing, open issue with the **`contributions-welcome`** label that covers this work. Drive-by PRs, speculative refactors, typo-only PRs, and unsolicited features are not accepted.
- **Respect claimed work:** If someone is assigned to an issue or has stated they are working on it, do not open a competing PR without coordinating in the issue thread.
- **Stay inside scope:** Implement only what the approved issue (or approved sub-issue) describes. Open separate issues for unrelated findings.
- **Multiple PRs per issue:** Allowed when each PR states its slice of the work. Use `Addresses`/`Refs`/`Part of` for partial PRs; use `Fixes`/`Closes` only on the PR that finishes the issue (or on a sub-issue).
- **One PR, one focus:** Do not bundle multiple unrelated issues into one PR without maintainer agreement in the issue.
- **No duplicate bot chores:** Dependency bumps and similar housekeeping belong to existing automation; do not duplicate that work.

## Darts development principles

Follow [CONTRIBUTING.md — Main Development Principles](CONTRIBUTING.md#main-development-principles). In short:

- Simple, clear API; `TimeSeries` as the user-facing data type; avoid future leakage and user foot-guns.
- Clear code over clever code; PEP-8; type hints; docstrings on public API; unit tests for changes.
- Use `uv`, pre-commit/ruff, and `uv run pytest` as described in `CONTRIBUTING.md`.

## AI-assisted contributions

AI assistance is allowed only when a **human** drives the contribution and takes responsibility:

- The human must understand, verify, and be able to explain every change in review.
- Disclose AI use in the PR description (tool name and what it was used for, e.g. tests, docs, refactor).
- Fully automated PRs submitted without meaningful human review may be closed.
- AI-generated code must meet the same bar as any other contribution (CI, style, tests, Darts conventions).

## Technical procedure

See [CONTRIBUTING.md — Technical Procedure](CONTRIBUTING.md#technical-procedure): fork, branch from **master** with `feature/`, `fix/`, or `refactor/` prefixes, changelog under **Unreleased**, PR into **master**.
