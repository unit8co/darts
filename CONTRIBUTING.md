# Darts Contribution Guidelines

## Picking an Issue on Which to Work

The backlog of issues and ongoing work is tracked here: https://github.com/unit8co/darts/projects/1

**External contributions (fork -> pull request):** work must be tied to a GitHub issue that a maintainer has marked with the **`ok`** label. Without that label, the **contribution-gate** check will fail and the PR cannot merge.

1. Open a new issue or pick an existing one from the backlog.
2. Comment on the issue and discuss the approach with core developers.
3. Wait until a maintainer adds the **`ok`** label before you invest in a large change or open a PR for merge.
4. In your PR description, link the approved issue so automation can find it (sidebar-only links do not satisfy the contribution gate):
   - **`Fixes #123`**, **`Closes #123`**, or **`Resolves #123`** when this PR **fully** completes the issue (GitHub will auto-close the issue on merge).
   - **`Addresses #123`**, **`Refs #123`**, or **`Part of #123`** when the PR is **partial** — the issue stays open for follow-up PRs.

If you spot improvements that are not yet in the backlog, open a regular GitHub issue (https://github.com/unit8co/darts/issues) and discuss it with the core team first.

Repository collaborators with write access may open PRs without the label for internal coordination; the gate applies to outside contributors.

AI coding agents should read [AGENTS.md](AGENTS.md) for the same rules in agent-oriented form.

### Maintainer triage

Maintainers (triage/write access) decide when external work is welcome. Label as **`ok`** when the issue is understood, in scope, and you are happy for a contributor (or their PR) to proceed.

### AI-assisted contributions

Use of AI tools (Copilot, Claude, Cursor, etc.) to help write code, tests, or docs is fine when **you** remain in the loop:

- You are responsible for correctness, tests, style, and responding to review feedback.
- AI coding agents must read [AGENTS.md](AGENTS.md) for general instructions.
- State in the PR description which tools you used and for what (e.g. “Cursor for unit test scaffolding”).
- PRs that appear fully automated, without a human who can explain the design, may be closed.

### Security

Do not open public pull requests for undisclosed security vulnerabilities. Discuss sensitive findings with maintainers privately first.


## Main Development Principles

* Focus on simplicity and clarity for end users.
* Pay attention to designing the best possible API, simple by default, but offering control when needed.
* Make it hard for users to make mistakes (e.g., future data leakage).
* `TimeSeries` is the main data type used to interface with Darts because:
    * We do not want users to have to worry about the specifics of Numpy, Torch etc
    (users shouldn't have to know what's used behind the scenes if they don't want to).
    * `TimeSeries` objects provide guarantees that the data represent a well-formed time series.
* It is in general OK to propose breaking changes, if these changes are really genuinely improving the API.
* Embrace functional principles as much as possible (immutability, using pure functions).
* Coding style: better write clear code than fancy code. Follow PEP-8. Use type hints.
* Write good docstrings for public classes and methods.
* Always unit test everything.

#### These principles should prevail even if:
* They mean more code has to be written inside the library.
* They decrease the raw performance / computational speed; although in general we always
  strive to find computationally efficient solutions.


## Technical Procedure

1. Make sure your work is tracked by an issue that has the **`ok`** label (see above)
2. Fork the repository.
3. Clone the forked repository locally.
4. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already: `curl -LsSf https://astral.sh/uv/install.sh | sh`
5. Install all latest development dependencies: `uv sync --group dev-all --upgrade`
6. Set up [automatic code formatting and linting](#code-formatting-and-linting): `uv run pre-commit install`
7. Create a new branch:
    * Branch off from the **master** branch.
    * Prefix the branch with the type of update you are making:
        * `feature/`
        * `fix/`
        * `refactor/`
        * …
    * Work on your update
8. Check that your code passes all the tests and design new unit tests if needed: `uv run pytest`.
9. If your contribution introduces a non-negligible change, add it to `CHANGELOG.md` under the "Unreleased" section.
   You can already refer to the pull request. In addition, for tracking contributions we are happy if you provide
   your full name (if you want to) and link to your Github handle. Example:
   ```
   - Added new feature XYZ. [#001](https://https://github.com/unit8co/darts/pull/001) by [<Your Name>](https://github.com/<your-handle>).
   ```
10. Create a pull request from your new branch into the **master** branch. Reference at least one approved issue in the PR description:
    * `Fixes #123`**, **`Closes #123`**, or **`Resolves #123`** when this PR **fully** completes the issue.
    * `Addresses #123`** or **`Refs #123`** when the PR partially completes the issue.
11. `Codecov` will add a test coverage report in the pull request. Make sure your test cover all changed lines.

### Build the Documentation Locally

You can build the documentation locally using `make`:

```bash
# ensure pandoc is available. If not, install it: https://pandoc.org/installing.html
pandoc --version
# build the docs (uv sync already installed darts in editable mode)
uv run make --directory=./docs build-all-docs
```

After that docs will be available in `./docs/build/html` directory. You can just open `./docs/build/html/index.html` using your favourite browser.

### Code Formatting and Linting

Darts uses [Black via Ruff](https://docs.astral.sh/ruff/formatter/) with default values for automatic code formatting, along with [ruff](https://docs.astral.sh/ruff/).
As part of the checks on pull requests, it is checked whether the code still adheres to the code style.
To ensure you don't need to worry about formatting and linting when contributing, it is recommended to set up at least one of the following:
- Integration in git (recommended):
    1. Install the pre-commit hook using `uv run pre-commit install`
    2. This will install `ruff` linting hooks
    3. The formatters will automatically fix all files and in case of some non-trivial case `ruff` will highlight any remaining problems before committing
- Integration in your editor:
    - For other integrations please look at the documentation for your editor
