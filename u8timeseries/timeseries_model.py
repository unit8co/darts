from abc import ABC, abstractmethod
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from .utils import add_time_delta_to_datetime, fill_dates_between
from .backtesting import backtest


ALLOWED_STEP_DURATIONS = {'second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'}


def assert_step_duration(stepduration_str):
    assert stepduration_str in ALLOWED_STEP_DURATIONS, \
        'step duration must be in {}'.format(ALLOWED_STEP_DURATIONS)


class TimeseriesModel(ABC):
    """
    A simple interface to fit models (on some training set), and predict the n next data points
    """

    @abstractmethod
    def __init__(self):
        # Stores training date information (if provided):
        self.training_dates = None
        self.time_column = None
        self.stepduration_str = None

        # state
        self.fit_called = False

    @abstractmethod
    def fit(self, df, target_column, time_column, stepduration_str):
        """
        :param df: A Pandas DataFrame that contains at least one column with time series values
        :param target_column: the column containing time series values
        :param time_column: optionally, the column containing corresponding timestamps (can be None)
        :param stepduration_str: if [time_column] is provided, this must provide the duration of time steps,
               among following values: {'second', 'minute', 'hour', 'day', 'week', 'month', 'year'}

        TODO: Interpolate missing values
        """
        if time_column is not None:
            assert_step_duration(stepduration_str)

            self.time_column = time_column
            self.training_dates = df[time_column].values
            self.stepduration_str = stepduration_str

        self.fit_called = True

    @abstractmethod
    def predict(self, n):
        """
        :return: A Pandas DataFrame with following columns:
                 - y_hat: contains the point-predictions of the model
                 - [time_column] (optional): a time column (if a [time_column] is provided to fit())
                 - y_lower (optional): lower confidence bound
                 - y_upper (optional): upper confidence bound
        """
        assert self.fit_called, 'predict() method called before fit()'

    def backtest(self, df, target_column, time_column, stepduration_str,
                 start_dt, n, eval_fun, nr_steps_iter=1, predict_nth_only=False):

        # Prepare generic fit() and predict() calls to be used from backtest()
        def fit_fn(*args):
            return self.fit(*args)

        def predict_fn(_, _n):
            return self.predict(_n)

        return backtest(df, target_column, time_column, stepduration_str, start_dt, n, eval_fun, fit_fn,
                        predict_fn, nr_steps_iter, predict_nth_only)

    def _get_new_dates(self, n):
        """
        This function creates a list of the n new dates (after the end of training set)
        :param n: number of dates after training set to generate
        """
        return [add_time_delta_to_datetime(self.training_dates[-1], i, self.stepduration_str)
                for i in range(1, n + 1)]

    def _build_forecast_df(self, point_preds, lower_bound=None, upper_bound=None):
        """
        Builds the pandas DataFrame to be returned by predict() method
        The column names are inspired from Prophet

        :param point_preds: a list or array of n point-predictions
        :param lower_bound: optionally, a list or array of lower bounds
        :param upper_bound:optionally, a list or array of upper bounds
        :return: a dataframe nicely formatted
        """

        columns = {
            'yhat': pd.Series(point_preds)
        }

        if self.time_column is not None:
            n = len(point_preds)
            new_dates = self._get_new_dates(n)
            columns[self.time_column] = pd.Series(new_dates)

        if lower_bound is not None:
            assert len(point_preds) == len(lower_bound), 'bounds should be same size as point predictions'
            columns['yhat_lower'] = lower_bound

        if upper_bound is not None:
            assert len(point_preds) == len(upper_bound), 'bounds should be same size as point predictions'
            columns['yhat_upper'] = upper_bound

        return pd.DataFrame(columns)


# TODO: move?
class SupervisedTimeSeriesModel:

    def __init__(self, model=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)):
        """
        :param model: The regression model to use. It can be any sklearn model with fit() and predict()
        """
        self.model = model
        self.target_column = None
        self.time_column = None
        self.training_dates = None
        self.fit_called = False
        self.stepduration_str = None

    def __str__(self):
        return 'supervised ({})'.format(self.model)

    def fit(self, df, target_column, time_column, stepduration_str=None, feature_columns=None):
        """
        Supports missing time steps

        :param df: A Pandas DataFrame containing the training set. It has to contain at least one time column and one
                   column with time series values (targets). Optionally, it can also contain extra features columns.
        :param target_column: The name of the column containing the target values
        :param time_column: The name of the column containing the timestamps. The timestamps can be of any type
                            that is hashable and comparable. They will be encoded (using ordinal encoding) before
                            being treated as a feature column by the regressor.
                            If not provided, the time is assumed to be part of feature columns, and encoded
                            by the user.
        :param stepduration_str: the duration of a time step
        :param feature_columns: A list of numerical columns to use as features. Should not contain the time column,
                                unless if [time_column] is not provided and the time has already been encoded in a
                                numerical type. If None, use all remaining columns.
        """

        assert len(df) >= 2, 'Need at least 2 data points'

        if target_column in df.columns:
            X_train = df.drop([target_column], axis=1)
        else:
            X_train = df

        # TODO: retain only numerical columns; or support automatic encoding of categorical features as well
        X_train = X_train if feature_columns is None else X_train.copy()[feature_columns]

        self.target_column = target_column

        if time_column is not None:
            assert_step_duration(stepduration_str)

            # Encode time column using the same ordinal encoding used during training;
            # possibly interpolating if some time steps are missing
            le = LabelEncoder()
            first_dt = min(df[time_column])
            last_dt = max(df[time_column])
            all_dts = fill_dates_between(first_dt, last_dt, stepduration_str)
            le.fit(all_dts)

            self.training_dates = [pd.Timestamp(d) for d in df[time_column].values]
            time_codes = le.transform(self.training_dates)
            X_train.loc[:, time_column + '-u8ts_codes'] = time_codes
            if self.time_column in X_train:
                X_train = X_train.drop([time_column], axis=1)

            self.time_column = time_column
            self.stepduration_str = stepduration_str

        y_train = df[target_column].values
        self.fit_called = True
        self.model.fit(X_train, y_train)

    def predict(self, df, feature_columns=None):
        """
        :param df: A Pandas DataFrame containing features columns
        :param feature_columns: A list of columns to use as features. If None, use all columns.
        :return: A Pandas Series containing the predictions for all rows in df

        Note: it is the caller responsibility to ensure all columns used as feature contain only
        data available at the corresponding values of the time column.
        """

        assert self.fit_called, 'predict() method called before fit()'

        if self.target_column in df.columns:
            X_test = df.drop([self.target_column], axis=1)
        else:
            X_test = df

        X_test = X_test if feature_columns is None else df.copy()[feature_columns]

        if self.time_column is not None:
            # Apply an ordinal encoding consistent with training set
            start_test_date = min(df[self.time_column])
            start_train_date = min(self.training_dates)
            end_test_date = max(df[self.time_column])
            end_train_date = max(self.training_dates)

            # If the test set comes before the training set, we raise an exception
            if start_test_date < end_train_date:
                raise ValueError('Some test dates provided to predict() are earlier than dates of training set.')

            # Interpolate dates in between and use ordinal encoding consistent with training set
            all_dates = fill_dates_between(start_train_date, end_test_date, self.stepduration_str)
            le = LabelEncoder()
            le.fit(all_dates)
            test_time_codes = le.transform([pd.Timestamp(d) for d in df[self.time_column].values])
            X_test.loc[:, self.time_column + '-u8ts_codes'] = test_time_codes

            if self.time_column in X_test:
                X_test = X_test.drop([self.time_column], axis=1)

        predictions = self.model.predict(X_test)
        to_return = df.copy()
        to_return.loc[:, 'yhat'] = predictions
        # TODO: perhaps also confidence intervals, depending what is supported by underlying sklearn model
        return to_return

    def backtest(self, df, target_column, time_column, stepduration_str,
                 start_dt, n, eval_fun, nr_steps_iter=1, predict_nth_only=False, feature_columns=None):

        # Prepare generic fit() and predict() calls to be used from backtest()
        def fit_fn(*args):
            return self.fit(*args, feature_columns)

        def predict_fn(val_set, _):
            return self.predict(val_set, feature_columns)

        return backtest(df, target_column, time_column, stepduration_str, start_dt, n, eval_fun, fit_fn,
                        predict_fn, nr_steps_iter, predict_nth_only)
