## macOS install: 

The install happens with `pip`. For `conda` users, do the following first:
```
conda install gcc
conda install -c conda-forge fbprophet
```

Next, from the root of u8timeseries:
```
pip install .
```

If Fortran is not installed on the device, the following error message might be printed during the installation:

```
error: library dfftpack has Fortran sources but no Fortran compiler found
```
This can be solved by installing gcc using Homebrew:
```
brew install gcc
```

## docker install

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

## version bumping
To bump the current version of u8timeseries, please execute the version bumping script from the root:
```
python version_bump.py
```
The user will then be prompted to enter the desired bump settings.
Alternatively, the settings can also be passed as arguments when calling the script:
```
python version_bump.py -b BUMP -r RELEASE
```
BUMP should be an integer between 0 and 3 indicating the type of version bump.
(0 for no version increment, 1 for Major, 2 for Minor, 3 for Patch)
RELEASE is expected to be 'y' or 'n', indicating whether or not the SNAPSHOT suffix should be added.
If either of the arguments is missing, or if an invalid value is given, the user will be prompted
to enter them again.
The versioning of u8timeseries follows the semantic versioning specification (https://semver.org).