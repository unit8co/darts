# Darts Contribution Guidelines

## Picking an Issue on Which to Work

The backlog of issues and ongoing work is tracked here: https://github.com/unit8co/darts/projects/1
Anyone is welcome to pick an issue from the backlog, work on it and submit a pull request.
However, it is strongly recommended to first comment on the issue, and discuss the approach
together with some of the core developers.

If you spot issues or would like to propose new improvements that are not (yet) in the backlog
of issues above, the best procedure is to open a regular Github issue (https://github.com/unit8co/darts/issues),
and discuss it with some of the core team.


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

1. Make sure your work is being tracked (and if possible discussed) by an existing issue on the backlog
2. Fork the repository.
3. Clone the forked repository locally.
4. Create a clean Python env and install requirements with pip: `pip install -r requirements/dev-all.txt`
5. Set up [automatic code formatting and linting](#code-formatting-and-linting)
6. Create a new branch:
    * Branch off from the **master** branch.
    * Prefix the branch with the type of update you are making:
        * `feature/`
        * `fix/`
        * `refactor/`
        * …
    * Work on your update
7. Check that your code passes all the tests and design new unit tests if needed: `pytest`.
8. If your contribution introduces a non-negligible change, add it to `CHANGELOG.md` under the "Unreleased" section.
   You can already refer to the pull request. In addition, for tracking contributions we are happy if you provide
   your full name (if you want to) and link to your Github handle. Example:
   ```
   - Added new feature XYZ. [#001](https://https://github.com/unit8co/darts/pull/001)
     by [<Your Name>](https://github.com/<your-handle>).
   ```
9. Create a pull request from your new branch into the **master** branch.
10. `Codecov` will add a test coverage report in the pull request. Make sure your test cover all changed lines.

### Build the Documentation Locally

You can build the documentation locally using `make`:

```bash
# ensure pandoc is available. If not, install it: https://pandoc.org/installing.html
pandoc --version
# install darts locally in editable mode
pip install -e .
# build the docs
make --directory=./docs build-all-docs
```

After that docs will be available in `./docs/build/html` directory. You can just open `./docs/build/html/index.html` using your favourite browser.

### Code Formatting and Linting

Darts uses [Black via Ruff](https://docs.astral.sh/ruff/formatter/) with default values for automatic code formatting, along with [ruff](https://docs.astral.sh/ruff/).
As part of the checks on pull requests, it is checked whether the code still adheres to the code style.
To ensure you don't need to worry about formatting and linting when contributing, it is recommended to set up at least one of the following:
- Integration in git (recommended):
    1. Install the pre-commit hook using `pre-commit install`
    2. This will install `ruff` linting hooks
    3. The formatters will automatically fix all files and in case of some non-trivial case `ruff` will highlight any remaining problems before committing
- Integration in your editor:
    - For other integrations please look at the documentation for your editor

### Development environment on Mac with Apple Silicon M1 processor (arm64 architecture)

Please follow the procedure described in [INSTALL.md](https://github.com/unit8co/darts/blob/master/INSTALL.md#test-environment-appple-m1-processor)
to set up a x_64 emulated environment. For the development environment, instead of installing Darts with
`pip install darts`, instead go to the darts cloned repo location and install the packages with: `pip install -r requirements/dev-all.txt`.
If necessary, follow the same steps to setup libomp for lightgbm.
Finally, verify your overall environment setup by successfully running all unitTests with `pytest`.
