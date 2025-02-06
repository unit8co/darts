import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

from darts.timeseries import TimeSeries

# Suppress all warnings
warnings.filterwarnings("ignore")


def create_random_dataframes(
    num_rows: int = 10,
    num_columns: int = 3,
    index: bool = True,
    start_date: str = "2023-01-01",
    freq: str = "D",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create three pandas DataFrames with random data and dates as the index or as a column.

    Parameters:
    - num_rows (int): The number of rows in the DataFrames.
    - num_columns (int): The number of columns in the DataFrames.
    - index (bool): If True, the date is the index of the DataFrame. If False, the date is a column named 'date'.
    - start_date (str): The start date for the date range (used only if date_format is 'date').
    - freq (str): The frequency of the date range (used only if date_format is 'date').

    Returns:
    - tuple: A tuple containing three DataFrames (df_date, df_numpy, df_integer).
    """
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate a date range or integer list based on the date_format parameter
    date_values = pd.date_range(start=start_date, periods=num_rows, freq=freq)
    integer_values = list(range(1, num_rows + 1))
    numpy_values = np.array(
        pd.date_range(start=start_date, periods=num_rows, freq=freq),
        dtype="datetime64[D]",
    )

    # Create random data for the DataFrames
    data = {f"col_{i}": np.random.randn(num_rows) for i in range(num_columns)}

    # Create the DataFrames
    df_date = pd.DataFrame(data)
    df_numpy = pd.DataFrame(data)
    df_integer = pd.DataFrame(data)

    col_names = df_date.columns.values

    # Set the date as index or as a column based on the index parameter
    if index:
        df_date.index = date_values
        df_numpy.index = numpy_values
        df_integer.index = integer_values
    else:
        df_date["date"] = date_values
        df_numpy["date"] = numpy_values
        df_integer["date"] = integer_values

    if index:
        time_col = None
    else:
        time_col = "date"

    return [
        [df_date, col_names, time_col],
        [df_numpy, col_names, time_col],
        [df_integer, col_names, time_col],
    ]


def test_dataframes() -> list:
    test_config = product(
        [10, 100, 1000],
        [10, 100, 500],
        [True, False],
    )

    dataframes_list = [
        create_random_dataframes(
            num_rows=num_rows, num_columns=num_columns, index=index
        )
        for num_rows, num_columns, index in test_config
    ]

    return dataframes_list


df_list = test_dataframes()

num_iter = 5
pandas_global_timer = 0
narwhals_global_timer = 0

for _ in range(num_iter):
    pandas_timer = 0
    narwhals_timer = 0
    for df_config in df_list:
        for df, col_names, time_col in df_config:
            for i in range(2):
                # on the second run we shuffle the data
                if i == 1:
                    df = df.sample(frac=1)

                # pandas processing time
                begin = time.time()
                pandas_timeseries = TimeSeries.from_dataframe(
                    df, value_cols=col_names, time_col=time_col, freq=None
                )
                end = time.time()
                pandas_timer += end - begin

                # narwhals processing time
                begin_nw = time.time()
                narwhals_timeseries = TimeSeries.from_narwhals_dataframe(
                    df, value_cols=col_names, time_col=time_col, freq=None
                )
                end_nw = time.time()
                narwhals_timer += end_nw - begin_nw

                # Check if the TimeSeries objects are equal
                try:
                    assert pandas_timeseries.time_index.equals(
                        narwhals_timeseries.time_index
                    )
                except AssertionError as e:
                    print(
                        f"Index assertion failed for DataFrame with columns {col_names} and time_col {time_col}: {e}"
                    )
                try:
                    np.testing.assert_array_almost_equal(
                        pandas_timeseries.all_values(), narwhals_timeseries.all_values()
                    )
                except AssertionError as e:
                    print(
                        f"Equal assertion failed for DataFrame with columns {col_names} and time_col {time_col}: {e}"
                    )

    print("pandas processing time: ", pandas_timer)
    print("narwhals processing time: ", narwhals_timer, "\n")
    pandas_global_timer += pandas_timer
    narwhals_global_timer += narwhals_timer

pandas_global_timer /= num_iter
narwhals_global_timer /= num_iter

print("Average pandas processing time: ", pandas_global_timer)
print("Average narwhals processing time: ", narwhals_global_timer)

diff_in_fraction = (-pandas_global_timer + narwhals_global_timer) / pandas_global_timer
print(f"Average processing time difference: {diff_in_fraction:.2%}")
