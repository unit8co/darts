import pandas as pd
from u8timeseries.models.baselines import KthValueAgoBaseline
from u8timeseries.timeseries import TimeSeries

my_df = pd.DataFrame({
  'timestep': [1, 2, 3],
  'values': [4, 5, 6]
})

series = TimeSeries.from_dataframe(my_df, 'timestep', 'values')
series.plot()

my_model = KthValueAgoBaseline(K=1)
my_model.fit(series)

# n step ahead prediction:
predictions = my_model.predict(n=6)
print(predictions)
