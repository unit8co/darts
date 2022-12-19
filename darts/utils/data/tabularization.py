from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from darts.logging import raise_if
from darts.timeseries import TimeSeries
from darts.utils.utils import series2seq


def _create_lagged_data(
    target_series: Union[TimeSeries, Sequence[TimeSeries]],
    output_chunk_length: int,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    lags: Optional[Sequence[int]] = None,
    lags_past_covariates: Optional[Sequence[int]] = None,
    lags_future_covariates: Optional[Sequence[int]] = None,
    max_samples_per_ts: Optional[int] = None,
    is_training: Optional[bool] = True,  # other option: 'inference
    multi_models: Optional[bool] = True,
):
    """
    Helper function that creates training/validation matrices (X and y as required in sklearn), given series and
    max_samples_per_ts.

    X has the following structure:
    lags_target | lags_past_covariates | lags_future_covariates

    Where each lags_X has the following structure (lags_X=[-2,-1] and X has 2 components):
    lag_-2_comp_1_X | lag_-2_comp_2_X | lag_-1_comp_1_X | lag_-1_comp_2_X

    y has the following structure (output_chunk_length=4 and target has 2 components):
    lag_+0_comp_1_target | lag_+0_comp_2_target | ... | lag_+3_comp_1_target | lag_+3_comp_2_target

    Parameters
    ----------
    target_series
        The target series of the regression model.
    output_chunk_length
        The output_chunk_length of the regression model.
    past_covariates
        Optionally, the past covariates of the regression model.
    future_covariates
        Optionally, the future covariates of the regression model.
    lags
        Optionally, the lags of the target series to be used as features.
    lags_past_covariates
        Optionally, the lags of the past covariates to be used as features.
    lags_future_covariates
        Optionally, the lags of the future covariates to be used as features.
    max_samples_per_ts
        Optionally, the maximum number of samples to be drawn for training/validation
        The kept samples are the most recent ones.
    is_training
        Optionally, whether the data is used for training or inference.
        If inference, the rows where the future_target_lags are NaN are not removed from X,
        as we are only interested in the X matrix to infer the future target values.
    """

    # ensure list of TimeSeries format
    if isinstance(target_series, TimeSeries):
        target_series = [target_series]
        past_covariates = [past_covariates] if past_covariates else None
        future_covariates = [future_covariates] if future_covariates else None

    Xs, ys, Ts = [], [], []

    # iterate over series
    for idx, target_ts in enumerate(target_series):
        covariates = [
            (
                past_covariates[idx].pd_dataframe(copy=False)
                if past_covariates
                else None,
                lags_past_covariates,
            ),
            (
                future_covariates[idx].pd_dataframe(copy=False)
                if future_covariates
                else None,
                lags_future_covariates,
            ),
        ]

        df_X = []
        df_y = []
        df_target = target_ts.pd_dataframe(copy=False)

        # y: output chunk length lags of target
        if multi_models:
            for future_target_lag in range(output_chunk_length):
                df_y.append(
                    df_target.shift(-future_target_lag).rename(
                        columns=lambda x: f"{x}_horizon_lag{future_target_lag}"
                    )
                )
        else:
            df_y.append(
                df_target.shift(-output_chunk_length + 1).rename(
                    columns=lambda x: f"{x}_horizon_lag{output_chunk_length-1}"
                )
            )

        if lags:
            for lag in lags:
                df_X.append(
                    df_target.shift(-lag).rename(
                        columns=lambda x: f"{x}_target_lag{lag}"
                    )
                )

        # X: covariate lags
        for covariate_name, (df_cov, lags_cov) in zip(["past", "future"], covariates):
            if lags_cov:
                if not is_training:
                    # We extend the covariates dataframe
                    # so that when we create the lags with shifts
                    # we don't have nan on the last (or first) rows. Only useful for inference.
                    df_cov = df_cov.reindex(df_target.index.union(df_cov.index))

                for lag in lags_cov:
                    df_X.append(
                        df_cov.shift(-lag).rename(
                            columns=lambda x: f"{x}_{covariate_name}_cov_lag{lag}"
                        )
                    )

        # combine lags
        df_X = pd.concat(df_X, axis=1)
        df_y = pd.concat(df_y, axis=1)
        df_X_y = pd.concat([df_X, df_y], axis=1)

        if is_training:
            df_X_y = df_X_y.dropna()
        # We don't need to drop where y are none for inference, as we just care for X
        else:
            df_X_y = df_X_y.dropna(subset=df_X.columns)

        Ts.append(df_X_y.index)
        X_y = df_X_y.values

        # keep most recent max_samples_per_ts samples
        if max_samples_per_ts:
            X_y = X_y[-max_samples_per_ts:]
            Ts[-1] = Ts[-1][-max_samples_per_ts:]

        raise_if(
            X_y.shape[0] == 0,
            "Unable to build any training samples of the target series "
            + (f"at index {idx} " if len(target_series) > 1 else "")
            + "and the corresponding covariate series; "
            "There is no time step for which all required lags are available and are not NaN values.",
        )

        X, y = np.split(X_y, [df_X.shape[1]], axis=1)

        Xs.append(X)
        ys.append(y)

    # combine samples from all series
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, Ts


def _add_static_covariates(model, series, features):
    """
    Add static covariates to the features. Accounts for series with potentially different static covariates
    by padding with 0 to accomodate for the maximum number of available static_covariates in any of the given
    series in the sequence. If no static covariates are provided for a given series, its corresponding features
    are padded with 0.
    """

    series = series2seq(series)
    reps = features.shape[0] // len(series)
    # collect static covariates info
    map = {"covs_width": [], "values": []}
    for ts in series:
        if ts.static_covariates is not None:
            # reshape with order="F" to ensure that the covariates are read column wise
            scovs = ts.static_covariates_values(copy=False).reshape(1, -1, order="F")
            map["covs_width"].append(scovs.shape[1])
            map["values"].append(scovs)
        else:
            map["covs_width"].append(0)
            map["values"].append(np.array([]))

    max_width = max(map["covs_width"])

    if max_width == 0:
        if (
            hasattr(model, "n_features_in_")
            and model.n_features_in_ is not None
            and model.n_features_in_ > features.shape[1]
        ):
            # for when series in prediction do not have static covariates but some of the training series did
            pad_zeros = np.zeros((1, model.n_features_in_ - features.shape[1]))
            return np.concatenate(
                [features, np.tile(pad_zeros, reps=(reps, 1))], axis=1
            )
        else:
            return features

    else:
        # at least one series in the sequence has static covariates
        static_covs = []

        # build static covariates array
        for i in range(len(series)):
            pad_zeros = np.zeros((1, max_width - map["covs_width"][i]))
            scovs = (
                np.concatenate((map["values"][i], pad_zeros), axis=1)
                if map["covs_width"][i] > 0
                else pad_zeros
            )
            static_covs.append(np.tile(scovs, reps=(reps, 1)))
        static_covs = np.concatenate(static_covs, axis=0)

        # concatenate static covariates to features
        return np.concatenate([features, static_covs], axis=1)
