import logging
from .utils import add_time_delta_to_datetime


def backtest(df, target_column, time_column, stepduration_str, start_dt, n, eval_fun, fit_fn,
             predict_fn, nr_steps_iter=1, predict_nth_only=False):
    """
    Performs backtesting and returns the outputs of a user-provided evaluation function.

    This function builds several validation sets, by iterating a pointer over the dataframe,
    starting at [start_dt] and every [nr_steps_iter] time slot. The validation sets are the
    data sets containing the n time steps after the training set.

    The function trains the model on the train set and emits predictions for the [n] points
    of the validation set. It then calls [eval_fun()] on the predicted values (and the targets),
    and returns a list containing the outputs of [eval_fun()] on all validation sets.

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

    # df_to_use = df.copy() if feature_columns is None else df[feature_columns + [time_column]]

    def _update_pointer_and_val_set(old_pointer, nr_steps=0):
        new_pointer = add_time_delta_to_datetime(old_pointer, nr_steps, stepduration_str)
        end_val_set_dt = add_time_delta_to_datetime(new_pointer, n, stepduration_str)
        mask = (new_pointer < df[time_column]) & (df[time_column] <= end_val_set_dt)
        new_val_df = df.loc[mask]
        return new_pointer, new_val_df

    current_pointer, current_val_df = _update_pointer_and_val_set(start_dt)
    results = []

    # TODO: check duration of validation set instead of nr of data points here
    while len(current_val_df) >= n:
        # TODO sliding or expanding window
        train_df = df[df[time_column] <= current_pointer]

        if len(train_df) == 0:
            logging.warning('Skipping validation set starting on {}, as it yields empty training set.'
                            .format(current_pointer))
            current_pointer, current_val_df = _update_pointer_and_val_set(current_pointer, nr_steps_iter)
            continue

        try:
            fit_fn(train_df, target_column, time_column, stepduration_str)
            preds = predict_fn(current_val_df, n)['yhat']  # TODO: is this kind of ugly?

            y_true = list(current_val_df[target_column])

            if predict_nth_only:
                # We only care about the last point
                preds = list([preds[-1]])
                y_true = list([y_true[-1]])

            results.append(eval_fun(y_true, preds))
        except Exception as e:
            # TODO: proper handling
            logging.warning('Something went wrong when training, for validation set starting on {}'
                            .format(current_pointer) + ':\n' + str(e))
        finally:
            # update pointer and validation set
            current_pointer, current_val_df = _update_pointer_and_val_set(current_pointer, nr_steps_iter)

    if len(results) == 0:
        logging.warning('Empty backtesting results: did you specify datetimes allowing at least 1 validation set?')

    return results
