import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import CatBoostModel

dates = pd.date_range("2021-02-01", "2021-04-19", freq="W-Mon")
series = TimeSeries.from_times_and_values(dates, np.sin(np.arange(len(dates))))
future_cov = TimeSeries.from_times_and_values(dates, np.cos(np.arange(len(dates))))

model = CatBoostModel(lags_future_covariates=(0, 1), output_chunk_length=10)
model.fit(series, future_covariates=future_cov)


hist_fc = model.historical_forecasts(
    series,
    future_covariates=future_cov,
    start=pd.Timestamp("2021-02-11"),
    retrain=lambda *args: False,
)
print(hist_fc.start_time())
# >> 2021-04-12 00:00:00

hist_fc2 = model.historical_forecasts(
    series,
    future_covariates=future_cov,
    start=pd.Timestamp("2021-02-11"),
    retrain=False,
)

print(hist_fc2.start_time())
# >> 2021-02-15 00:00:00
