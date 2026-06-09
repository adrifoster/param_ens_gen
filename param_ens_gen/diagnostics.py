"""Diagnostic utility functions for param_ens_gen

Functions for analyzing and understanding parameter ranges and default values,
useful for plotting and ensemble design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .parameter import Parameter
from .parameter_dataset import ParameterDataset
from .utils import read_param_list


def _expand_normalized(
    param: Parameter, normalized: float | np.ndarray | list
) -> list[tuple[str, float]]:
    """Expand a normalized value into (name, value) pairs, one per index."""
    param_names = (
        param.spec.base_params if param.spec.base_params else [param.spec.name]
    )
    normalized_list = normalized if isinstance(normalized, list) else [normalized]

    rows = []
    for pname, norm_val in zip(param_names, normalized_list):
        arr = np.asarray(norm_val)
        if arr.ndim == 0:
            rows.append((pname, float(arr)))
        else:
            for i, v in enumerate(arr.flat):
                rows.append((f"{pname}_{i}", float(v)))
    return rows


def normalize_defaults(
    param_dir: Path,
    default_param_file: Path,
    posterior_sources: Path | None = None,
    param_list: list[str] | None = None,
) -> pd.DataFrame:
    """Return normalized default values for all parameters in param_dir.

    For each parameter, computes where the default value sits in [0, 1]
    relative to the parameter's prior bounds. Useful for understanding
    whether defaults are centered, skewed toward min or max, or outside
    the specified range.

    Args:
        param_dir (Path): Path to the parameter metadata directory containing
            main.csv and optional per-PFT bound files.
        default_param_file (Path): Path to the default parameter file
            (NetCDF or JSON).
        posterior_sources (Path | None, optional): Path to a posterior sources
            YAML file. Required for parameters with strategy='posterior'.
            Defaults to None.
        param_list (list[str] | None, optional): Subset of parameters to
            normalize. If None, all parameters in main.csv are used.
            Defaults to None.

    Returns:
        pd.DataFrame: One row per parameter with columns:
            - 'parameter': parameter name
            - 'normalized_value': where the default sits in [0, 1]
    """
    param_dir = Path(param_dir)
    default_param_file = Path(default_param_file)

    main, pft_sheets = read_param_list(param_dir)

    if param_list is not None:
        missing = set(param_list) - set(main.parameter_name)
        if missing:
            raise ValueError(
                f"param_list contains parameters not found in main.csv: "
                f"{sorted(missing)}. "
                f"Available parameters: {sorted(main.parameter_name)}"
            )
        main = main[main.parameter_name.isin(param_list)].copy()

    posterior_config: dict = {}
    if posterior_sources is not None:
        posterior_sources = Path(posterior_sources)
        if not posterior_sources.exists():
            raise FileNotFoundError(
                f"Posterior sources file '{posterior_sources}' does not exist."
            )
        with open(posterior_sources, "r", encoding="utf-8") as f:
            posterior_config = yaml.safe_load(f)

    default_ds = ParameterDataset.from_path(default_param_file)

    params = [
        Parameter.from_row(
            row,
            pft_sheet=pft_sheets.get(row["parameter_name"]),
            default_ds=default_ds,
            posterior_config=(
                posterior_config.get(row["parameter_name"])
                if posterior_config
                else None
            ),
        )
        for _, row in main.iterrows()
    ]

    rows = []
    for param in params:
        default_value = param.get_default(default_ds)
        normalized = param.normalize(default_value, default_ds)
        rows.extend(_expand_normalized(param, normalized))
    return pd.DataFrame(rows, columns=["parameter", "normalized_value"])
