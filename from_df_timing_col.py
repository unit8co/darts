import argparse
import json
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from darts.timeseries import TimeSeries

# Suppress all warnings
warnings.filterwarnings("ignore")


def test_from_dataframe(f_name: str):
    return getattr(TimeSeries, f_name)


def create_random_dataframes(
    num_rows: int = 10,
    num_columns: int = 3,
    index: bool = True,
    col_names_given: bool = True,
    start_date: str = "1900-01-01",
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

    if col_names_given:
        col_names = df_date.columns.values
    else:
        col_names = None

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
        [1000],
        [1, 10, 100, 1000],
        [True, False],
        [True, False],
    )

    dataframes_list = [
        create_random_dataframes(
            num_rows=num_rows,
            num_columns=num_columns,
            index=index,
            col_names_given=col_names_given,
        )
        for num_rows, num_columns, index, col_names_given in test_config
    ]

    return dataframes_list


def calculate_processing_time(
    f_name: str,
    num_iter: int,
    save_path="/Users/julesauthier/Documents/darts/from_df_times/data/",
):
    df_list = test_dataframes()
    df_func = test_from_dataframe(f_name)

    # Initialize dictionaries to store processing times
    times = {}

    # Initialize the progress bar
    total_iterations = (
        len(df_list) * 2 * 3
    )  # 2 iterations per dataframe configuration, 3 df per config
    progress_bar = tqdm(total=total_iterations, desc="Processing DataFrames")

    for df_config in df_list:
        for df, col_names, time_col in df_config:
            num_cols = df.shape[1]
            if num_cols > 1 and (num_cols % 2 == 1 or num_cols == 2):
                num_cols -= 1
            dict_entry = str(num_cols)

            for i in range(2):
                # on the second run we shuffle the data
                if i == 1:
                    df = df.sample(frac=1)
                    dict_entry += "_shuffled"

                begin = time.time()
                for _ in range(num_iter):
                    _ = df_func(df, value_cols=col_names, time_col=time_col, freq=None)
                end = time.time()
                timer = (end - begin) / num_iter

                if dict_entry not in times:
                    times[dict_entry] = timer
                else:
                    times[dict_entry] += timer

                # Update the progress bar
                progress_bar.update(1)

    file_name = f_name + "_avg_time_cols_" + str(num_iter) + "_iter.json"

    # Store the average times in separate JSON files
    with open(save_path + file_name, "w") as f:
        json.dump(times, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The function to test and the number of iter can "
    )
    parser.add_argument(
        "--f_name", type=str, default="from_dataframe", help="method to time"
    )
    parser.add_argument(
        "--n_iter", type=int, default=100, help="number of function call"
    )

    args = parser.parse_args()

    f_name = args.f_name
    n_iter = args.n_iter

    calculate_processing_time(f_name, n_iter)
