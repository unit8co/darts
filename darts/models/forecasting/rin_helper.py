"""RIN runtime helper utilities.

Encapsulates parser-driven RIN configuration, module construction,
component-index handling, and runtime normalization/inverse transforms.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from darts.logging import raise_log
from darts.models.components.layer_norm_variants import RINorm
from darts.utils.data.torch_datasets.utils import PLModuleInput


@dataclass(frozen=True)
class RINParams:
    affine: bool = True
    eps: float = 1e-5


@dataclass(frozen=True)
class RINGroupConfig:
    enabled: bool = False
    components: list[str] | None = None
    params: RINParams | None = None


@dataclass(frozen=True)
class RINConfig:
    params: RINParams
    series: RINGroupConfig
    past_covariates: RINGroupConfig
    future_covariates: RINGroupConfig
    legacy_target_only: bool = False


class RINParser:
    @staticmethod
    def _parse_rin_params(source: dict[str, Any], key_path: str) -> RINParams:
        valid_keys = {"affine", "eps"}
        invalid_keys = sorted(set(source) - valid_keys)
        if invalid_keys:
            raise_log(
                ValueError(
                    f"Invalid key(s) in `{key_path}`: {invalid_keys}. Supported keys are `affine` and `eps`."
                ),
            )

        affine = source.get("affine", True)
        if not isinstance(affine, bool):
            raise_log(
                ValueError(f"`{key_path}['affine']` must be a boolean."),
            )

        eps = source.get("eps", 1e-5)
        if not isinstance(eps, (int | float)) or isinstance(eps, bool) or eps <= 0:
            raise_log(
                ValueError(f"`{key_path}['eps']` must be a positive number (> 0)."),
            )

        return RINParams(affine=affine, eps=float(eps))

    @classmethod
    def _parse_rin_group(
        cls,
        group_name: str,
        value: bool | list[str] | dict[str, Any],
    ) -> RINGroupConfig:
        key_path = f"use_reversible_instance_norm['{group_name}']"
        if isinstance(value, bool):
            return RINGroupConfig(enabled=value)

        if isinstance(value, list):
            if not value or not all(isinstance(item, str) for item in value):
                raise_log(
                    ValueError(
                        f"`{key_path}` must be one of: boolean, list of component "
                        "names, or a dictionary with keys `components` and optional "
                        "`params`."
                    ),
                )
            duplicates = sorted({item for item in value if value.count(item) > 1})
            if duplicates:
                raise_log(
                    ValueError(
                        f"`{key_path}['components']` contains duplicate component names: {duplicates}."
                    ),
                )
            return RINGroupConfig(enabled=True, components=value)

        if not isinstance(value, dict):
            raise_log(
                ValueError(
                    f"`{key_path}` must be one of: boolean, list of component names, "
                    "or a dictionary with keys `components` and optional `params`."
                ),
            )

        valid_keys = {"components", "params"}
        invalid_keys = sorted(set(value) - valid_keys)
        if invalid_keys:
            raise_log(
                ValueError(
                    f"Invalid key(s) in `{key_path}`: {invalid_keys}. Supported keys are `components` and `params`."
                ),
            )

        if "components" not in value:
            raise_log(
                ValueError(
                    f"`{key_path}['components']` is required when `{group_name}` is a dictionary."
                ),
            )

        components = value["components"]
        if components is True:
            selected_components: list[str] | None = None
            enabled = True
        elif (
            isinstance(components, list)
            and components
            and all(isinstance(item, str) for item in components)
        ):
            duplicates = sorted({
                item for item in components if components.count(item) > 1
            })
            if duplicates:
                raise_log(
                    ValueError(
                        f"`{key_path}['components']` contains duplicate component names: {duplicates}."
                    ),
                )
            selected_components = components
            enabled = True
        else:
            raise_log(
                ValueError(
                    f"`{key_path}['components']` must be `True` or a non-empty list of strings."
                ),
            )

        params_value = value.get("params")
        if params_value is None:
            params = None
        else:
            if not isinstance(params_value, dict):
                raise_log(
                    ValueError(f"`{key_path}['params']` must be a dictionary."),
                )
            params = cls._parse_rin_params(params_value, f"{key_path}['params']")

        return RINGroupConfig(
            enabled=enabled,
            components=selected_components,
            params=params,
        )

    @classmethod
    def parse_use_reversible_instance_norm(
        cls, value: bool | dict
    ) -> tuple[RINConfig, bool]:
        if isinstance(value, bool):
            if not value:
                cfg = RINConfig(
                    params=RINParams(),
                    series=RINGroupConfig(enabled=False),
                    past_covariates=RINGroupConfig(enabled=False),
                    future_covariates=RINGroupConfig(enabled=False),
                    legacy_target_only=False,
                )
                return cfg, False

            cfg = RINConfig(
                params=RINParams(),
                series=RINGroupConfig(enabled=True),
                past_covariates=RINGroupConfig(enabled=False),
                future_covariates=RINGroupConfig(enabled=False),
                legacy_target_only=True,
            )
            return cfg, True

        if not isinstance(value, dict):
            raise_log(
                ValueError(
                    "`use_reversible_instance_norm` must be a boolean or a dictionary."
                ),
            )

        legacy_keys = {"affine", "eps"}
        selector_keys = {"params", "series", "past_covariates", "future_covariates"}
        value_keys = set(value)

        if value_keys & legacy_keys and value_keys & selector_keys:
            raise_log(
                ValueError(
                    "`use_reversible_instance_norm` cannot mix legacy keys (`affine`, `eps`) with selector keys "
                    "(`params`, `series`, `past_covariates`, `future_covariates`). Use `params` in selector mode."
                ),
            )

        # legacy dict mode; empty dict must stay legacy-compatible
        if value_keys.issubset(legacy_keys):
            params = cls._parse_rin_params(value, "use_reversible_instance_norm")
            cfg = RINConfig(
                params=params,
                series=RINGroupConfig(enabled=True),
                past_covariates=RINGroupConfig(enabled=False),
                future_covariates=RINGroupConfig(enabled=False),
                legacy_target_only=True,
            )
            return cfg, True

        valid_top_keys = selector_keys
        invalid_keys = sorted(value_keys - valid_top_keys)
        if invalid_keys:
            raise_log(
                ValueError(
                    "Invalid key(s) in `use_reversible_instance_norm`: "
                    f"{invalid_keys}. Supported top-level keys are `affine`, `eps` (legacy mode) or "
                    "`params`, `series`, `past_covariates`, `future_covariates` (selector mode)."
                ),
            )

        top_params = value.get("params")
        if top_params is None:
            params = RINParams()
        else:
            if not isinstance(top_params, dict):
                raise_log(
                    ValueError(
                        "`use_reversible_instance_norm['params']` must be a dictionary."
                    ),
                )
            params = cls._parse_rin_params(
                top_params, "use_reversible_instance_norm['params']"
            )

        group_values = {
            "series": value.get("series", False),
            "past_covariates": value.get("past_covariates", False),
            "future_covariates": value.get("future_covariates", False),
        }

        series_cfg = cls._parse_rin_group("series", group_values["series"])
        past_cov_cfg = cls._parse_rin_group(
            "past_covariates", group_values["past_covariates"]
        )
        future_cov_cfg = cls._parse_rin_group(
            "future_covariates", group_values["future_covariates"]
        )

        cfg = RINConfig(
            params=params,
            series=series_cfg,
            past_covariates=past_cov_cfg,
            future_covariates=future_cov_cfg,
            legacy_target_only=False,
        )
        active = any([
            cfg.series.enabled,
            cfg.past_covariates.enabled,
            cfg.future_covariates.enabled,
        ])
        return cfg, active

    @staticmethod
    def _get_component_names(series: Sequence[Any]) -> list[str]:
        return [str(component) for component in series[0].components]

    @classmethod
    def _resolve_group_component_indices(
        cls,
        group_name: str,
        requested_components: list[str],
        series: Sequence[Any],
    ) -> list[int]:
        available = cls._get_component_names(series)
        for ts in series[1:]:
            names = [str(component) for component in ts.components]
            if names != available:
                raise_log(
                    ValueError(
                        f"`use_reversible_instance_norm['{group_name}']` uses name-based selection, but "
                        "component names differ across training series. Please align component names or use "
                        "boolean selection."
                    ),
                )

        missing = [name for name in requested_components if name not in available]
        if missing:
            raise_log(
                ValueError(
                    f"For `use_reversible_instance_norm['{group_name}']`, unknown component name(s): {missing}. "
                    f"Available components are: {available}."
                ),
            )

        return [available.index(name) for name in requested_components]

    @classmethod
    def resolve_component_indices_for_fit(
        cls,
        use_reversible_instance_norm: bool | dict,
        series: Sequence[Any],
        past_covariates: Sequence[Any] | None,
        future_covariates: Sequence[Any] | None,
    ) -> dict[str, list[int] | None] | None:
        rin_cfg, active = cls.parse_use_reversible_instance_norm(
            use_reversible_instance_norm
        )
        if not active:
            return None

        resolved: dict[str, list[int] | None] = {
            "series": None,
            "past_covariates": None,
            "future_covariates": None,
        }

        if rin_cfg.series.components is not None:
            resolved["series"] = cls._resolve_group_component_indices(
                "series", rin_cfg.series.components, series
            )

        if rin_cfg.past_covariates.components is not None:
            if past_covariates is None:
                raise_log(
                    ValueError(
                        "For `use_reversible_instance_norm['past_covariates']`, name-based selection requires "
                        "`past_covariates` to be provided to `fit()`."
                    ),
                )
            resolved["past_covariates"] = cls._resolve_group_component_indices(
                "past_covariates",
                rin_cfg.past_covariates.components,
                past_covariates,
            )

        if rin_cfg.future_covariates.components is not None:
            if future_covariates is None:
                raise_log(
                    ValueError(
                        "For `use_reversible_instance_norm['future_covariates']`, name-based selection requires "
                        "`future_covariates` to be provided to `fit()`."
                    ),
                )
            resolved["future_covariates"] = cls._resolve_group_component_indices(
                "future_covariates",
                rin_cfg.future_covariates.components,
                future_covariates,
            )

        return resolved

    @classmethod
    def validate_fit_from_dataset_config(
        cls,
        use_reversible_instance_norm: bool | dict,
    ) -> None:
        rin_cfg, _ = cls.parse_use_reversible_instance_norm(
            use_reversible_instance_norm
        )
        for group_name, group_cfg in [
            ("series", rin_cfg.series),
            ("past_covariates", rin_cfg.past_covariates),
            ("future_covariates", rin_cfg.future_covariates),
        ]:
            if group_cfg.components is not None:
                raise_log(
                    ValueError(
                        f"`fit_from_dataset()` does not support name-based RIN component selection for "
                        f"`{group_name}` because dataset samples do not expose component names. Use boolean "
                        "selection or call `fit()` with TimeSeries inputs."
                    ),
                )

    @classmethod
    def is_target_series_rin_active(
        cls,
        use_reversible_instance_norm: bool | dict,
    ) -> bool:
        rin_cfg, active = cls.parse_use_reversible_instance_norm(
            use_reversible_instance_norm
        )
        return active and rin_cfg.series.enabled

    @classmethod
    def override_affine_for_foundation_model(
        cls,
        use_reversible_instance_norm: bool | dict,
    ) -> tuple[bool | dict, bool]:
        rin_cfg, active = cls.parse_use_reversible_instance_norm(
            use_reversible_instance_norm
        )
        series_params = rin_cfg.series.params or rin_cfg.params
        requires_affine_override = (
            active and rin_cfg.series.enabled and series_params.affine
        )
        if not requires_affine_override:
            return use_reversible_instance_norm, False

        if use_reversible_instance_norm is True:
            return {"affine": False}, True

        if not isinstance(use_reversible_instance_norm, dict):
            return use_reversible_instance_norm, False

        if set(use_reversible_instance_norm).issubset({"affine", "eps"}):
            return {
                **use_reversible_instance_norm,
                "affine": False,
            }, True

        updated = {**use_reversible_instance_norm}
        if rin_cfg.series.components is None:
            params = updated.get("params")
            updated["params"] = (
                {**params, "affine": False} if params else {"affine": False}
            )
            return updated, True

        series_cfg = updated.get("series")
        if isinstance(series_cfg, dict):
            series_cfg = {**series_cfg}
            group_params = series_cfg.get("params")
            series_cfg["params"] = (
                {**group_params, "affine": False} if group_params else {"affine": False}
            )
            updated["series"] = series_cfg
            return updated, True

        params = updated.get("params")
        updated["params"] = {**params, "affine": False} if params else {"affine": False}
        return updated, True


@dataclass(frozen=True)
class RINComponentIndices:
    series: list[int] | None
    past_covariates: list[int] | None
    future_covariates: list[int] | None


class RINHelper(nn.Module):
    def __init__(
        self,
        rin_config: RINConfig,
        active: bool,
        n_targets: int,
        n_past_covariates: int,
        n_future_covariates: int,
    ) -> None:
        super().__init__()
        self.rin_config = rin_config
        self.active = active
        self.n_targets = n_targets
        self.n_past_covariates = n_past_covariates
        self.n_future_covariates = n_future_covariates

        self._rin_series_indices: list[int] | None = (
            None if self.rin_config.series.components is None else []
        )
        self._rin_past_cov_indices: list[int] | None = (
            None if self.rin_config.past_covariates.components is None else []
        )
        self._rin_future_cov_indices: list[int] | None = (
            None if self.rin_config.future_covariates.components is None else []
        )

        self.rin = self._build_group_rin_module(self.rin_config.series, self.n_targets)
        self.rin_past_cov = self._build_group_rin_module(
            self.rin_config.past_covariates,
            self.n_past_covariates,
        )
        self.rin_future_cov = self._build_group_rin_module(
            self.rin_config.future_covariates,
            self.n_future_covariates,
        )

    @classmethod
    def _extract_optional_component_width(
        cls,
        train_sample_shape: tuple | None,
        idx: int,
    ) -> int:
        if train_sample_shape is None or idx >= len(train_sample_shape):
            return 0

        shape_at_idx = train_sample_shape[idx]
        if shape_at_idx is None or len(shape_at_idx) < 2:
            return 0

        return shape_at_idx[1]

    @classmethod
    def from_user_config(
        cls,
        use_reversible_instance_norm: bool | dict,
        n_targets: int,
        train_sample_shape: tuple | None,
    ) -> "RINHelper":
        rin_config, active = RINParser.parse_use_reversible_instance_norm(
            use_reversible_instance_norm
        )

        n_past_covariates = cls._extract_optional_component_width(
            train_sample_shape=train_sample_shape,
            idx=1,
        )
        n_future_covariates = cls._extract_optional_component_width(
            train_sample_shape=train_sample_shape,
            idx=3,
        )

        return cls(
            rin_config=rin_config,
            active=active,
            n_targets=n_targets,
            n_past_covariates=n_past_covariates,
            n_future_covariates=n_future_covariates,
        )

    @property
    def series_indices(self) -> list[int] | None:
        return self._rin_series_indices

    @property
    def past_covariate_indices(self) -> list[int] | None:
        return self._rin_past_cov_indices

    @property
    def future_covariate_indices(self) -> list[int] | None:
        return self._rin_future_cov_indices

    def get_component_indices(self) -> RINComponentIndices:
        return RINComponentIndices(
            series=self._rin_series_indices,
            past_covariates=self._rin_past_cov_indices,
            future_covariates=self._rin_future_cov_indices,
        )

    def set_component_indices(
        self,
        series_indices: list[int] | None = None,
        past_covariate_indices: list[int] | None = None,
        future_covariate_indices: list[int] | None = None,
    ) -> None:
        if self.rin_config.series.components is not None and series_indices is not None:
            self._rin_series_indices = series_indices
        if (
            self.rin_config.past_covariates.components is not None
            and past_covariate_indices is not None
        ):
            self._rin_past_cov_indices = past_covariate_indices
        if (
            self.rin_config.future_covariates.components is not None
            and future_covariate_indices is not None
        ):
            self._rin_future_cov_indices = future_covariate_indices

    def _effective_rin_params(
        self, group_cfg: RINGroupConfig
    ) -> dict[str, float | bool]:
        group_params = group_cfg.params
        if group_params is None:
            group_params = self.rin_config.params
        return {"affine": group_params.affine, "eps": group_params.eps}

    def _get_rin_input_dim(
        self,
        group_cfg: RINGroupConfig,
        total_dim: int,
    ) -> int:
        if group_cfg.components is None:
            return total_dim
        return len(group_cfg.components)

    def _build_group_rin_module(
        self,
        group_cfg: RINGroupConfig,
        total_dim: int,
    ) -> RINorm | None:
        if not group_cfg.enabled:
            return None

        input_dim = self._get_rin_input_dim(group_cfg, total_dim)
        if input_dim <= 0:
            return None

        group_params = self._effective_rin_params(group_cfg)
        return RINorm(input_dim=input_dim, **group_params)

    def _apply_group_rin(
        self,
        tensor: torch.Tensor,
        left: int,
        right: int,
        group_rin: RINorm | None,
        group_cfg: RINGroupConfig,
        group_indices: list[int] | None,
    ) -> None:
        if not group_cfg.enabled or group_rin is None or right <= left:
            return

        if group_cfg.components is None:
            tensor[:, :, left:right] = group_rin(tensor[:, :, left:right])
            return

        if group_indices:
            block = tensor[:, :, left:right]
            block[:, :, group_indices] = group_rin(block[:, :, group_indices])
            tensor[:, :, left:right] = block

    def forward(self, x_in: PLModuleInput) -> PLModuleInput:
        if not self.active:
            return x_in

        x_past, x_future, x_static, future_target = x_in
        past_features = x_past.clone()

        # normalize target features
        self._apply_group_rin(
            tensor=past_features,
            left=0,
            right=self.n_targets,
            group_rin=self.rin,
            group_cfg=self.rin_config.series,
            group_indices=self._rin_series_indices,
        )

        # normalize optional past covariate features inside x_past
        past_cov_left = self.n_targets
        past_cov_right = self.n_targets + self.n_past_covariates
        self._apply_group_rin(
            tensor=past_features,
            left=past_cov_left,
            right=past_cov_right,
            group_rin=self.rin_past_cov,
            group_cfg=self.rin_config.past_covariates,
            group_indices=self._rin_past_cov_indices,
        )

        # normalize historic future covariates inside x_past
        future_cov_left = self.n_targets + self.n_past_covariates
        future_cov_right = future_cov_left + self.n_future_covariates
        self._apply_group_rin(
            tensor=past_features,
            left=future_cov_left,
            right=future_cov_right,
            group_rin=self.rin_future_cov,
            group_cfg=self.rin_config.future_covariates,
            group_indices=self._rin_future_cov_indices,
        )

        # normalize future covariates tuple element
        if (
            self.rin_config.future_covariates.enabled
            and self.rin_future_cov is not None
            and x_future is not None
        ):
            x_future = x_future.clone()
            if self.rin_config.future_covariates.components is None:
                x_future = self.rin_future_cov(x_future)
            elif self._rin_future_cov_indices:
                x_future[:, :, self._rin_future_cov_indices] = self.rin_future_cov(
                    x_future[:, :, self._rin_future_cov_indices]
                )

        return past_features, x_future, x_static, future_target

    def inverse(self, out):
        if not self.active or not self.rin_config.series.enabled or self.rin is None:
            return out

        if self.rin_config.series.components is None:
            if isinstance(out, tuple):
                pred, *rest = out
                return self.rin.inverse(pred), *rest
            return self.rin.inverse(out)

        if isinstance(out, tuple):
            pred, *rest = out
            pred = pred.clone()
            if self._rin_series_indices:
                pred[:, :, self._rin_series_indices, :] = self.rin.inverse(
                    pred[:, :, self._rin_series_indices, :]
                )
            return pred, *rest

        pred = out.clone()
        if self._rin_series_indices:
            pred[:, :, self._rin_series_indices, :] = self.rin.inverse(
                pred[:, :, self._rin_series_indices, :]
            )
        return pred
