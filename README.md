# u8timeseries

## install

### using conda
`pmdarima` is currently not supported on conda which means we need to reinstall it through pip

    conda env create -f conda_recipe/environment.yml
    conda install -c conda-forge -c alkaline-ml --name u8timeseries-dev --file requirements/main.txt
    # here install any additional dev requirements from the other requirements/*.txt files
    source activate u8timeseries-dev
    pip install "pmdarima>=1.5.3"
    pip install --no-deps -e .
    
### pure pip
This approach most likely requires other non-Python dependencies.

    pip install .
    # install any additional dev requirements, e.g.:
    pip install -r requirements/docs.txt

If Fortran is not installed on the device, the following error message might be printed during the installation:

```
error: library dfftpack has Fortran sources but no Fortran compiler found
```
This can be solved by installing gcc using Homebrew:
```
brew install gcc
```

### docker

Build and run the docker using the following two commands:
```
./build_docker.sh
./run_docker.sh
```
Then copy and paste the URL provided by the docker container into your browser to access jupyter notebook.

## Usage
For now the best documentation is examples.
See: https://github.com/unit8co/u8timeseries/blob/master/examples/Air-passengers-example.ipynb

## Issue with Prophet and Pandas
If you encounter the following error when trying to plot a `TimeSeries`:
```
float() argument must be a string or a number, not 'Period
```
this is likely because [Prophet deregisters the Pandas converters in its code](https://darektidwell.com/typeerror-float-argument-must-be-a-string-or-a-number-not-period-facebook-prophet-and-pandas/). To fix it, just call
```
pd.plotting.register_matplotlib_converters()
```

## unit test status
![ci_workflow](https://github.com/unit8co/u8timeseries/workflows/ci_workflow/badge.svg)
