import pandas as pd
from u8timeseries import KthValueAgoBaseline

my_df = pd.DataFrame({
  'timestep': [1, 2, 3],
  'values': [4, 5, 6]
})

my_model = KthValueAgoBaseline(K=1)
my_model.fit(my_df, 'values')

# n step ahead prediction:
predictions =  my_model.predict(n=6)
print(predictions)
