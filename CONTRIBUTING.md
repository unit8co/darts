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
    (users shouldn’t have to know what’s used behind the scenes if they don’t want to).
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
5. Create a new branch:
    * Branch off from the **master** branch.
    * Prefix the branch with the type of update you are making:
        * `feature/`
        * `fix/`
        * `refactor/`
        * …
    * Work on your update
6. Check that your code passes all the tests and design new unit tests if needed: `./gradlew unitTest_all`.
7. Verify your tests coverage by running `./gradlew coverageTest`
    * Additionally you can generate an xml report and use VSCode Coverage gutter to identify untested 
    lines with `./coverage.sh xml`
8. If your contribution introduces a significant change, add it to `CHANGELOG.md` under the "Unreleased" section.
9. Create a pull request from your new branch into the **master** branch.
