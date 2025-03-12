# Contributing

caveat is an actively maintained and utilised project.

## How to contribute

to report issues, request features, or exchange with our community, just follow the links below.

__Is something not working?__

[:material-bug: Report a bug](https://github.com/fredshone/caveat/issues/new?template=BUG-REPORT.yml "Report a bug in caveat by creating an issue and a reproduction"){ .md-button }

__Missing information in our docs?__

[:material-file-document: Report a docs issue](https://github.com/fredshone/caveat/issues/new?template=DOCS.yml "Report missing information or potential inconsistencies in our documentation"){ .md-button }

__Want to submit an idea?__

[:material-lightbulb-on: Request a change](https://github.com/fredshone/caveat/issues/new?template=FEATURE-REQUEST.yml "Propose a change or feature request or suggest an improvement"){ .md-button }

__Have a question or need help?__

[:material-chat-question: Ask a question](https://github.com/fredshone/caveat/discussions "Ask questions on our discussion board and get in touch with our community"){ .md-button }

## Developing caveat

To find beginner-friendly existing bugs and feature requests you may like to start out with, take a look at our [good first issues](https://github.com/fredshone/caveat/contribute).

### Setting up a development environment

To create a development environment for caveat, with all libraries required for development and quality assurance installed, it is easiest to install caveat using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager, as follows:

1. Install mamba with the [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) executable for your operating system.
2. Open the command line (or the "miniforge prompt" in Windows).
3. Download (a.k.a., clone) the caveat repository: `git clone git@github.com:fredshone/caveat.git`
4. Change into the `caveat` directory: `cd caveat`
5. Create the caveat mamba environment: `mamba create -n caveat -c conda-forge --file requirements/base.txt --file requirements/dev.txt`
6. Activate the caveat mamba environment: `mamba activate caveat`
7. Install the caveat package into the environment, in editable mode and ignoring dependencies (we have dealt with those when creating the mamba environment): `pip install --no-deps -e .`

All together:

--8<-- "README.md:docs-install-dev"

If installing directly with pip, you can install these libraries using the `dev` option, i.e., `pip install -e '.[dev]'`
Either way, you should add your environment as a jupyter kernel, so the example notebooks can run in the tests: `ipython kernel install --user --name=caveat`
If you plan to make changes to the code then please make regular use of the following tools to verify the codebase while you work:

- `pre-commit`: run `pre-commit install` in your command line to load inbuilt checks that will run every time you commit your changes.
The checks are: 1. check no large files have been staged, 2. lint python files for major errors, 3. format python files to conform with the [PEP8 standard](https://peps.python.org/pep-0008/).
You can also run these checks yourself at any time to ensure staged changes are clean by calling `pre-commit`.
- `pytest` - run the unit test suite and check test coverage.
!!! note

    If you already have an environment called `caveat` on your system (e.g., for a stable installation of the package), you will need to [chose a different environment name][choosing-a-different-environment-name].
    You will then need to add this as a pytest argument when running the tests: `pytest --nbmake-kernel=[my-env-name]`.


### Rapid-fire testing
The following options allow you to strip down the test suite to the bare essentials:
1. The test suite includes unit tests and integration tests (in the form of jupyter notebooks found in the `examples` directory, and tests in the `integration_tests` diractory).
The integration tests can be slow, so if you want to avoid them during development, you should run `pytest tests/`.
2. You can avoid generating coverage reports, by adding the `--no-cov` argument: `pytest --no-cov`.
3. By default, the tests run with up to two parallel threads, to increase this to e.g. 4 threads: `pytest -n4`.

All together:

``` shell
pytest tests/ --no-cov -n4
```

!!! note

    You cannot debug failing tests and have your tests run in parallel, you will need to set `-n0` if using the `--pdb` flag


## Submitting changes

--8<-- "CONTRIBUTING.md:docs"