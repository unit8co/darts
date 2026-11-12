"""
Layer Norm Variants
-------------------

MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn

from darts.logging import raise_log

_RIN_GROUP_KEYS = ("series", "past_covariates", "future_covariates")
_RIN_PARAM_KEYS = ("affine", "eps")


class RMSNorm(nn.Module):
    """An alternate to layer normalization, without mean centering and the learned bias [1]_

    References
    ----------
    .. [1] Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." Advances in Neural Information
           Processing Systems 32 (2019).
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class LayerNormNoBias(nn.LayerNorm):
    def __init__(self, input_size, **kwargs):
        super().__init__(input_size, elementwise_affine=False, **kwargs)


class LayerNorm(nn.LayerNorm):
    def __init__(self, input_size, **kwargs) -> None:
        super().__init__(input_size, **kwargs)


class RINormHelper(nn.Module):
    """Normalizes a single group of components with Reversible Instance Normalization.

    Used internally by :class:`RINorm`, which owns one independent instance of this class per
    active group (`series`, `past_covariates`, and/or `future_covariates`).
    """

    def __init__(
        self,
        input_dim: int | None = None,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Parameters
        ----------
        input_dim
            The number of components to normalize.
        eps
            The epsilon value for numerical stability.
        affine
            Whether to apply a learned affine transformation after normalization.
        """
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

        if self.affine and self.input_dim:
            self.affine_weight = nn.Parameter(torch.ones(self.input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes and stores `mean`/`stdev` from `x` (to be reused by :meth:`transform` /
        :meth:`inverse`), then normalizes `x` with them.

        Parameters
        ----------
        x
            Tensor with last dimension `input_dim`. Shape: ``(batch_size, seq_len, input_dim)``.
        """
        # TL;DR: calculate mean and variance over all dimensions except batch and input_dim
        calc_dims = tuple(range(1, x.ndim - 1))

        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # explicit class reference rather than `self.transform(x)`: `RINorm` subclasses
        # `RINormHelper` and overrides `transform` with a different signature/return shape, so
        # `self.transform` would incorrectly re-dispatch to that override when `self` is a `RINorm`
        # instance normalizing its `series` group (see `RINorm._apply_groups`).
        return RINormHelper.transform(self, x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` using statistics previously computed by :meth:`forward`.

        Unlike :meth:`forward`, this does **not** recompute ``mean`` and ``stdev``
        from ``x``; it reuses the values stored during the last :meth:`forward` call.
        This is useful for normalizing auxiliary inputs (e.g. teacher-forcing
        targets) that should share the same normalization statistics as the
        primary input.

        Parameters
        ----------
        x
            Tensor with the same last dimension as the original input to
            :meth:`forward`. Shape: ``(batch_size, seq_len, input_dim)``.
        """
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize `x` using the `mean`/`stdev` stored by the last :meth:`forward` call.

        Parameters
        ----------
        x
            Tensor to denormalize. `x` is assumed to be the output of
            `PLForecastingModule.forward()`, and has shape
            ``(batch_size, output_chunk_length, input_dim, nr_params)``.
        """
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (
                self.affine_weight.view(self.affine_weight.shape + (1,))
                + self.eps * self.eps
            )
        x = x * self.stdev.view(self.stdev.shape + (1,))
        x = x + self.mean.view(self.mean.shape + (1,))
        return x


class RINorm(RINormHelper):
    def __init__(
        self,
        input_dim: int | None = None,
        eps: float = 1e-5,
        affine: bool = True,
        past_cov_dim: int | None = None,
        future_cov_dim: int | None = None,
    ):
        """Reversible Instance Normalization based on [1]_

        Normalizes the target `series` and, optionally, subsets of `past_covariates` and/or
        `future_covariates` components. Each of the three groups (`series`, `past_covariates`,
        `future_covariates`) is normalized independently, with its own mean/standard deviation and
        (if `affine`) its own affine parameters. Since statistics are always computed per channel
        (independently of other channels), grouping different feature types together for a single
        :class:`RINorm` instance does not cause any cross-contamination between groups.

        Parameters
        ----------
        input_dim
            The number of target `series` components to normalize. ``None`` (default) disables
            series normalization.
        eps
            The epsilon value for numerical stability. Shared across all active groups.
        affine
            Whether to apply an affine transformation after normalization. Shared across all active
            groups, but a separate set of affine parameters is created per active group, each sized
            to that group's number of components.
        past_cov_dim
            The number of `past_covariates` components to normalize. ``None`` (default) disables
            past covariates normalization.
        future_cov_dim
            The number of `future_covariates` components to normalize. ``None`` (default) disables
            future covariates normalization.

        References
        ----------
        .. [1] Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift" International Conference on Learning Representations (2022)
        """
        has_series = input_dim is not None and input_dim > 0
        has_past_cov = past_cov_dim is not None and past_cov_dim > 0
        has_future_cov = future_cov_dim is not None and future_cov_dim > 0

        if not (has_series or has_past_cov or has_future_cov):
            raise_log(
                ValueError(
                    "`RINorm` requires at least one of `input_dim`, `past_cov_dim`, or "
                    "`future_cov_dim` to be a positive integer."
                )
            )

        # `series` is normalized directly on `self` (i.e. via the parent `RINormHelper`'s
        # `affine_weight`/`affine_bias`/`mean`/`stdev`), rather than through a `series_norm`
        # sub-module, so that checkpoints saved before per-group RIN (where `RINorm` itself held
        # `affine_weight`/`affine_bias`) remain loadable: those parameters must stay at
        # `rin.affine_weight`/`rin.affine_bias`, not move to `rin.series_norm.affine_weight`/
        # `rin.series_norm.affine_bias`. `past_covariates`/`future_covariates` are new groups with
        # no such backward-compatibility constraint, so they are handled by dedicated
        # `RINormHelper` sub-modules.
        super().__init__(
            input_dim=input_dim if has_series else None, eps=eps, affine=affine
        )
        self.has_series = has_series
        self.has_past_cov = has_past_cov
        self.has_future_cov = has_future_cov
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim

        if has_past_cov:
            self.past_cov_norm = RINormHelper(past_cov_dim, eps, affine)
        if has_future_cov:
            self.future_cov_norm = RINormHelper(future_cov_dim, eps, affine)

    def _apply_groups(
        self,
        op: str,
        x: torch.Tensor | None,
        past_cov: torch.Tensor | None,
        future_cov: torch.Tensor | None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    ):
        """Applies `RINormHelper` method `op` (``"forward"``, ``"transform"``, or ``"inverse"``) to
        each active, given group, and shapes the result per :meth:`forward`'s `Returns` section.
        """
        # `series` is handled by `RINormHelper`'s own implementation of `op`, applied to `self`
        # (see `__init__`); it cannot be called through `getattr(self, op)`, as that would
        # re-dispatch to this class's own (overridden) `forward`/`transform`/`inverse`.
        x_out = (
            getattr(RINormHelper, op)(self, x)
            if self.has_series and x is not None
            else None
        )
        past_cov_out = (
            getattr(self.past_cov_norm, op)(past_cov)
            if self.has_past_cov and past_cov is not None
            else None
        )
        future_cov_out = (
            getattr(self.future_cov_norm, op)(future_cov)
            if self.has_future_cov and future_cov is not None
            else None
        )
        # backward compatible with the pre-groups, series-only API: unwrap to a bare tensor only
        # for instances that don't have any covariates group active
        if x is not None and not (self.has_past_cov or self.has_future_cov):
            return x_out
        else:
            return x_out, past_cov_out, future_cov_out

    def forward(
        self,
        x: torch.Tensor | None = None,
        past_cov: torch.Tensor | None = None,
        future_cov: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Normalizes each of the (active) `x` (series), `past_cov`, and `future_cov` tensors
        independently, computing and storing per-group statistics to later be reused by
        :meth:`transform` / :meth:`inverse`.

        Each input tensor is expected to have shape ``(batch_size, seq_len, n_components)``, where
        ``n_components`` matches `input_dim` / `past_cov_dim` / `future_cov_dim` respectively.
        A group's tensor may be left as ``None`` if that group is not active (e.g. `past_cov=None`
        when `past_cov_dim` was not set at construction) or not provided for a given forward call.

        Returns
        -------
        tuple
            ``(x_out, past_cov_out, future_cov_out)``, with `None` in place of any group that
            wasn't normalized.
        """
        return self._apply_groups("forward", x, past_cov, future_cov)

    def transform(
        self,
        x: torch.Tensor | None = None,
        past_cov: torch.Tensor | None = None,
        future_cov: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    ):
        """Normalize `x` (series), `past_cov`, and/or `future_cov` using statistics previously
        computed by :meth:`forward` for the corresponding group(s).

        Unlike :meth:`forward`, this does **not** recompute the mean/standard deviation; it reuses
        the values stored during the last :meth:`forward` call for each group. This is useful for
        normalizing auxiliary inputs that should share the same normalization statistics as a
        group's primary input (e.g. teacher-forcing targets sharing `"series"` statistics, or the
        future window of a future covariate sharing statistics with its historic/past window).

        Parameters
        ----------
        x
            Series tensor to normalize, or `None` to skip. Shape: ``(batch_size, seq_len, input_dim)``.
        past_cov
            Past covariates tensor to normalize, or `None` to skip.
            Shape: ``(batch_size, seq_len, past_cov_dim)``.
        future_cov
            Future covariates tensor to normalize, or `None` to skip.
            Shape: ``(batch_size, seq_len, future_cov_dim)``.

        Returns
        -------
        torch.Tensor | tuple
            If `x` is given and this instance has no `past_covariates`/`future_covariates` group
            active (i.e. it was constructed with only `input_dim`), returns the normalized series
            tensor directly (backward compatible with the pre-groups, series-only call pattern).
            Otherwise, returns a 3-tuple ``(x_out, past_cov_out, future_cov_out)``, with `None` in
            place of any argument that was not given.
        """
        return self._apply_groups("transform", x, past_cov, future_cov)

    def inverse(
        self,
        x: torch.Tensor | None = None,
        past_cov: torch.Tensor | None = None,
        future_cov: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    ):
        """Denormalize `x` (series), `past_cov`, and/or `future_cov` using statistics previously
        computed by :meth:`forward` for the corresponding group(s).

        Parameters
        ----------
        x
            Series tensor to denormalize, or `None` to skip. `x` is assumed to be the output of
            `PLForecastingModule.forward()`, and has shape
            ``(batch_size, output_chunk_length, input_dim, nr_params)``.
        past_cov
            Past covariates tensor to denormalize, or `None` to skip.
        future_cov
            Future covariates tensor to denormalize, or `None` to skip.

        Returns
        -------
        torch.Tensor | tuple
            If `x` is given and this instance has no `past_covariates`/`future_covariates` group
            active (i.e. it was constructed with only `input_dim`), returns the denormalized series
            tensor directly (backward compatible with the pre-groups, series-only call pattern).
            Otherwise, returns a 3-tuple ``(x_out, past_cov_out, future_cov_out)``, with `None` in
            place of any argument that was not given.
        """
        return self._apply_groups("inverse", x, past_cov, future_cov)

    @staticmethod
    def group_is_active(group_cfg: bool | list[str]) -> bool:
        """Whether a single, already-parsed `use_reversible_instance_norm` group config (as
        returned per-key by :meth:`parse_config`) is active: `True`, or a non-empty list of
        component names.
        """
        return group_cfg is True or (isinstance(group_cfg, list) and len(group_cfg) > 0)

    @classmethod
    def parse_config(cls, config: bool | dict | None) -> dict | None:
        """Validates and normalizes a `use_reversible_instance_norm` value.

        Parameters
        ----------
        config
            Either:

            - ``False``/``None``: RIN is disabled.
            - ``True``: RIN is applied to all `series` components with default `RINorm`
              hyperparameters.
            - a legacy flat dict with keys in ``{"affine", "eps"}``: RIN is applied to all `series`
              components, constructing `RINorm` with the given hyperparameters.
            - a dict with keys in ``{"params", "series", "past_covariates", "future_covariates"}``:
              ``"params"`` is a dict with keys in ``{"affine", "eps"}`` used to construct `RINorm`;
              ``"series"``, ``"past_covariates"``, ``"future_covariates"`` are each either a `bool`
              (apply to all/none of that group's components) or a `list` of component name strings
              (apply only to the named components). Missing keys default to ``False``.

        Returns
        -------
        dict | None
            ``None`` if RIN is disabled, otherwise a normalized dict with keys ``"params"``,
            ``"series"``, ``"past_covariates"``, ``"future_covariates"`` (the latter three always
            either a `bool` or a `list[str]`).

        Notes
        -----
        Enabling a `past_covariates`/`future_covariates` group (`True` or a non-empty list) on a
        model that isn't actually given that covariate type at fit time raises a `ValueError`,
        regardless of whether the group was enabled with `True` or a list of component names. If a
        model is refit with varying covariates across calls, only enable a group unconditionally
        (`True`) for covariate types that are always provided.
        """
        if config is None or config is False:
            return None
        if config is True:
            return {
                "params": {},
                "series": True,
                "past_covariates": False,
                "future_covariates": False,
            }
        if not isinstance(config, dict):
            raise_log(
                ValueError(
                    "`use_reversible_instance_norm` must be a `bool` or a `dict`, received "
                    f"type `{type(config)}`."
                )
            )

        def _raise_unknown_keys(
            allowed_keys: tuple[str, ...], extra_hint: str = ""
        ) -> None:
            unknown_keys = set(config) - set(allowed_keys)
            if unknown_keys:
                raise_log(
                    ValueError(
                        f"Invalid `use_reversible_instance_norm` dict keys `{unknown_keys}`. "
                        f"Supported keys are `{allowed_keys}`{extra_hint}."
                    )
                )

        has_group_keys = any(key in config for key in _RIN_GROUP_KEYS)
        if not has_group_keys and "params" not in config:
            # legacy flat format, e.g. `{"affine": False}` / `{"eps": 1e-3}` -> series-only
            _raise_unknown_keys(
                _RIN_PARAM_KEYS,
                extra_hint=f" (legacy format), or `{('params',) + _RIN_GROUP_KEYS}`",
            )
            return {
                "params": dict(config),
                "series": True,
                "past_covariates": False,
                "future_covariates": False,
            }

        _raise_unknown_keys(("params",) + _RIN_GROUP_KEYS)

        params = config.get("params", {})
        if not isinstance(params, dict) or not set(params).issubset(_RIN_PARAM_KEYS):
            raise_log(
                ValueError(
                    "`use_reversible_instance_norm['params']` must be a `dict` with keys in "
                    f"`{_RIN_PARAM_KEYS}`, received `{params}`."
                )
            )

        parsed: dict = {"params": dict(params)}
        for key in _RIN_GROUP_KEYS:
            val = config.get(key, False)
            if isinstance(val, bool):
                parsed[key] = val
            elif (isinstance(val, list) or isinstance(val, tuple)) and all(
                isinstance(v, str) for v in val
            ):
                parsed[key] = list(val)
            else:
                raise_log(
                    ValueError(
                        f"`use_reversible_instance_norm['{key}']` must be a `bool` or a `list` "
                        f"of component name strings, received `{val}`."
                    )
                )

        if not any(cls.group_is_active(parsed[key]) for key in _RIN_GROUP_KEYS):
            raise_log(
                ValueError(
                    "At least one of `series`, `past_covariates`, `future_covariates` must be "
                    "enabled (`True`, or a non-empty list of component names) in "
                    "`use_reversible_instance_norm`."
                )
            )
        return parsed
