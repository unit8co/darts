from abc import ABC, abstractmethod
from dateutils import relativedelta
import pandas as pd
import logging


class TimeseriesModel(ABC):
    """
    A simple interface to fit models (on some training set), and predict the n next data points
    """

    @abstractmethod
    def __init__(self):
        self.allowed_stepduration_str = {'second', 'minute', 'hour', 'day', 'week', 'month', 'year'}

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
            assert stepduration_str in self.allowed_stepduration_str, \
                   'periodicity_str argument must be in {}'.format(self.allowed_stepduration_str)

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
        """
        Performs backtesting and returns the outputs of a user-provided evaluation function.

        This function builds several validation sets, by iterating a pointer over the dataframe,
        starting at [start_dt] and every [nr_steps_iter] time slot. The validation sets are the
        [n] data points after the pointer. The function trains the model on the train set and emits
        predictions for the [n] points of the validation set. It then calls [eval_fun()] on the
        predicted values (and the targets), and returns a list containing the outputs of [eval_fun()]
        on all validation sets.

        :param df: a Pandas DataFrame that contains at least one column with time series values
        :param target_column: the column containing time series values
        :param time_column: *mandatory for backtesting*, the column containing corresponding timestamps
        :param stepduration_str: this must provide the duration of time steps,
               among following values: {'second', 'minute', 'hour', 'day', 'week', 'month', 'year'}
        :param start_dt: the datetime corresponding to the beginning of the first validation set
        :param n: number of points in each validation sets
        :param eval_fun: a function of following form: eval_fun(true_values, predicted_values),
                         which returns some evaluation of the prediction (typically, an error value).
                         The arguments (true_values, predicted_values) are lists.
        :param nr_steps_iter: the number of time steps to elapse between each validation set
        :param predict_nth_only: If true, evaluates the predictions only on the n-th point of each validation set

        :return: A list containing the outputs of [eval_fun()] on all validation sets.
        """

        # TODO: at some point we can get rid of this requirement
        assert time_column is not None, 'argument "time_column" must be provided for backtesting'

        df_to_use = df[[time_column, target_column]]

        def _update_pointer_and_val_set(old_pointer, nr_steps=0):
            new_pointer = self._add_time_delta_to_datetime(old_pointer, nr_steps, stepduration_str)
            new_val_df = df_to_use[df_to_use[time_column] > new_pointer].iloc[:n, :]
            return new_pointer, new_val_df

        current_pointer, current_val_df = _update_pointer_and_val_set(start_dt)
        results = []

        while len(current_val_df) == n:
            train_df = df_to_use[df_to_use[time_column] <= current_pointer]

            if len(train_df) == 0:
                logging.warning('Skipping validation set starting on {}, as it yields empty training set.'
                                .format(current_pointer))
                current_pointer, current_val_df = _update_pointer_and_val_set(current_pointer, nr_steps_iter)
                continue

            try:
                self.fit(train_df, target_column, time_column, stepduration_str)
                preds = list(self.predict(n=n)['yhat'])
                self.__init__()  # re-init model, as fit() stores some state

                y_true = list(current_val_df[target_column])

                if predict_nth_only:
                    # We only care about the last point
                    preds = list([preds[-1]])
                    y_true = list([y_true[-1]])

                results.append(eval_fun(y_true, preds))
            except:
                # TODO: proper handling
                logging.warning('Something went wrong when training, for validation set starting on {}'
                                .format(current_pointer))
            finally:
                # update pointer and validation set
                current_pointer, current_val_df = _update_pointer_and_val_set(current_pointer, nr_steps_iter)

        if len(results) == 0:
            logging.warning('Empty backtesting results: did you specify datetimes allowing at least 1 validation set?')

        return results

    def _add_time_delta_to_datetime(self, np_dt, i, stepduration_str):
        """
        Adds [i] time-steps (whose length is given by [stepduration_str]) to the numpy datetime [np_dt]
        """
        datet = pd.Timestamp(np_dt)

        switch = {
            'second': lambda dt: dt + relativedelta(seconds=+i),
            'minute': lambda dt: dt + relativedelta(minutes=+i),
            'hour': lambda dt: dt + relativedelta(hours=+i),
            'day': lambda dt: dt + relativedelta(days=+i),
            'week': lambda dt: dt + relativedelta(weeks=+i),
            'month': lambda dt: dt + relativedelta(months=+i),
            'year': lambda dt: dt + relativedelta(years=+i)
        }
        return switch[stepduration_str](datet)

    def _get_new_dates(self, n):
        """
        This function creates a list of the n new dates (after the end of training set)
        :param n: number of dates after training set to generate
        """
        return [self._add_time_delta_to_datetime(self.training_dates[-1], i, self.stepduration_str)
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


class SupervisedTimeSeriesModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, df, target_column, feature_columns=None):
        """
        :param df: A Pandas DataFrame containing at least one feature column and one column with time series values (targets)
        :param feature_columns: A list of columns to use as features. If None, use all columns.
        """
        pass

    @abstractmethod
    def predict(self, df, feature_columns=None):
        """
        :param df: A Pandas DataFrame containing features columns
        :param feature_columns: A list of columns to use as features. If None, use all columns.
        :return: A Pandas Series containing the predictions for all rows in df
        """
        pass
