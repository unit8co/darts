## Install: 

The install happens with `pip`. For `conda` users, do the following first:
```
conda install gcc
conda install -c conda-forge fbprophet
```

And then:
```
pip install .
```

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
