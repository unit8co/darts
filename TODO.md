# Models

## Auto-Regressive
- Theta (https://robjhyndman.com/papers/Theta.pdf)
- FFT (https://www.youtube.com/watch?v=VYpAodcdFfA)

## Regressive
- vanilla RNN
- LSTM
- Gaussian process (http://www.gaussianprocess.org/gpml/chapters/RW2.pdf)
- Quantile regression forest
- seq2seq (https://www.youtube.com/watch?v=VYpAodcdFfA)

# Backtesting
A different backtesting, to really simulate historical predictions. I.e., with an increment of 1,
and returning a TimeSeries corresponding to forecasts that would have been made.

The function should accept a *list* of time horizons, and return a *list* of corresponding
predicted time series.


# Useful TimeSeries

The following could be useful TimeSeries to regress over:
- Indicator variable for holidays (1 during holidays 0 else)
- TimeSeries filled with values linearly increasing over time
- Constant TimeSeries
- Indicator for quarters

# Utils
- A function such as `checkresiduals()` used here https://otexts.com/fpp2/regression-evaluation.html

# Others
- Logging