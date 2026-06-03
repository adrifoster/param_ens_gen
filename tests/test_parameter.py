"""Tests for Parameter class"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


from param_ens_gen.parameter import (
    DimIndex,
    DefaultParameter,
    JointParameter,
    Parameter,
    ScaleFromRootParameter,
    SlicedParameter,
)
from param_ens_gen.sampler import (
    UniformSampler,
    PosteriorSampler,
)

from param_ens_gen.distribution_stat import (
    FixedStat,
    PercentStat,
    PFTStat,
)

# ===========================================================================
# DimIndex
# ===========================================================================


def test_dimindex_stores_dim_and_index():
    """DimIndex correctly stores dim name and 0-based index."""
    di = DimIndex(dim="fates_pft", index=2)
    assert di.dim == "fates_pft"
    assert di.index == 2


def test_dimindex_is_named_tuple():
    """DimIndex is a named tuple"""
    di = DimIndex("fates_pft", 0)
    assert di[0] == "fates_pft"
    assert di[1] == 0


# ===========================================================================
# Parameter registry / from_row
# ===========================================================================


def test_from_row_returns_default_parameter(default_row, default_ds):
    """from_row returns a DefaultParameter for param_type='default'."""
    param = Parameter.from_row(default_row, default_ds)
    assert isinstance(param, DefaultParameter)


def test_from_row_returns_sliced_parameter(sliced_row, default_ds):
    """from_row returns a SlicedParameter for param_type='sliced'."""
    param = Parameter.from_row(sliced_row, default_ds)
    assert isinstance(param, SlicedParameter)


def test_from_row_returns_scale_from_root_parameter(scale_from_root_row, default_ds):
    """from_row returns a ScaleFromRootParameter for param_type='scale_from_root'."""
    param = Parameter.from_row(scale_from_root_row, default_ds)
    assert isinstance(param, ScaleFromRootParameter)


def test_from_row_returns_joint_parameter(
    joint_param_row, default_ds, posterior_config
):
    """from_row returns a JointParameter for param_type='joint'."""
    param = Parameter.from_row(
        joint_param_row, default_ds, posterior_config=posterior_config
    )
    assert isinstance(param, JointParameter)


def test_from_row_raises_for_unknown_param_type(default_row, default_ds):
    """from_row raises ValueError for an unregistered param_type."""
    default_row["param_type"] = "unknown_type"
    with pytest.raises(ValueError, match="Unknown param_type"):
        Parameter.from_row(default_row, default_ds)


def test_from_row_empty_param_type_raises(default_row, default_ds):
    """from_row raises ValueError for a missing param_type cell"""
    default_row["param_type"] = ""
    with pytest.raises(ValueError, match="Unknown param_type"):
        Parameter.from_row(default_row, default_ds)


def test_from_row_whitespace_param_type_stripped(default_row, default_ds):
    """from_row raises ValueError for a missing param_type cell"""
    default_row["param_type"] = "  default  "
    param = Parameter.from_row(default_row, default_ds)
    assert isinstance(param, DefaultParameter)


# ===========================================================================
# Parameter from_row / __init__
# ===========================================================================


def test_from_row_default_grabs_paramspec(default_row, default_ds):
    """from_row correctly constructs the param spec data"""
    param = Parameter.from_row(default_row, default_ds)
    assert param.spec is not None
    spec = param.spec
    assert spec.name == "fates_leaf_slatop"
    assert spec.param_type == "default"
    assert spec.dims == ["fates_pft"]
    assert spec.slice_dim is None
    assert spec.slice_index is None
    assert spec.root_param is None
    assert not spec.base_params


def test_from_row_uniform_parameter_has_uniform_sampler(default_row, default_ds):
    """A uniform strategy parameter has a UniformSampler with correct DistributionStat"""
    param = Parameter.from_row(default_row, default_ds)
    assert param.sampler is not None
    assert isinstance(param.sampler, UniformSampler)
    assert isinstance(param.sampler.min_stat, FixedStat)
    assert isinstance(param.sampler.max_stat, FixedStat)


def test_from_row_posterior_parameter_has_posterior_sampler(
    joint_param_row, default_ds, posterior_config
):
    """A posterior strategy parameter has a PosteriorSampler"""
    param = Parameter.from_row(
        joint_param_row, default_ds, posterior_config=posterior_config
    )
    assert param.sampler is not None
    assert isinstance(param.sampler, PosteriorSampler)
    assert param.sampler.parameters == [
        "fates_leafn_vert_scaler_coeff1",
        "fates_leafn_vert_scaler_coeff2",
    ]
    assert len(param.sampler.sources) == 1


def test_from_row_percent_parameter_has_percent_stat(percent_row, default_ds):
    """from_row correctly constructs a parameter with percent bounds."""
    param = Parameter.from_row(percent_row, default_ds)
    assert param.sampler is not None
    assert isinstance(param.sampler, UniformSampler)
    assert isinstance(param.sampler.min_stat, PercentStat)
    assert isinstance(param.sampler.max_stat, PercentStat)


def test_from_row_pft_parameter_has_pft_stat(pft_row, default_ds, pft_sheet):
    """from_row correctly constructs a parameter with pft bounds."""
    param = Parameter.from_row(pft_row, default_ds, pft_sheet=pft_sheet)
    assert param.sampler is not None
    assert isinstance(param.sampler, UniformSampler)
    assert isinstance(param.sampler.min_stat, PFTStat)
    assert isinstance(param.sampler.max_stat, PFTStat)


def test_from_row_active_index_is_none_on_construction(default_row, default_ds):
    """active_index is None on a freshly constructed Parameter."""
    param = Parameter.from_row(default_row, default_ds)
    assert param.active_index is None


# ===========================================================================
# Parameter from_row / validate_params
# ===========================================================================


def test_from_row_raises_for_missing_variable(default_row, default_ds):
    """from_row raises ValueError when the variable is missing from the dataset."""
    default_row["parameter_name"] = "nonexistent_param"
    with pytest.raises(ValueError, match="not found in default dataset"):
        Parameter.from_row(default_row, default_ds)


def test_from_row_raises_for_wrong_dims(default_row, default_ds):
    """from_row raises ValueError when spec.dims does not match the dataset dims."""
    # Give the spec the wrong dims — variable exists but dims won't match
    default_row["coord"] = "['fates_leafage_class', 'fates_pft']"
    with pytest.raises(ValueError, match="Dimensions must match exactly"):
        Parameter.from_row(default_row, default_ds)


def test_from_row_raises_for_missing_base_param(sliced_row, default_ds):
    """from_row raises ValueError when a base_param variable is missing."""
    sliced_row["base_params"] = "['nonexistent_param']"
    with pytest.raises(ValueError, match="not found in default dataset"):
        Parameter.from_row(sliced_row, default_ds)


def test_from_row_raises_for_missing_root_param(scale_from_root_row, default_ds):
    """from_row raises ValueError when root_param is missing from the dataset."""
    scale_from_root_row["root_param"] = "nonexistent_root"
    with pytest.raises(ValueError, match="not found in default dataset"):
        Parameter.from_row(scale_from_root_row, default_ds)


# ===========================================================================
# Parameter from_row / validate_posterior
# ===========================================================================


def test_from_row_non_posterior_sampler_passes_through(default_row, default_ds):
    """A uniform sampler must not trigger any posterior validation."""
    param = DefaultParameter(default_row, default_ds)
    assert param.sampler is not None


def test_from_row_posterior_parameters_mismatch_raises(
    joint_param_row, default_ds, posterior_file
):
    """source.parameters must match _variables_to_validate()."""
    # joint param targets coeff1 and coeff2, but config names something else
    config = {
        "parameters": ["wrong_param_a", "wrong_param_b"],
        "files": [{"path": str(posterior_file), "array_indices": "all"}],
    }
    with pytest.raises(ValueError, match="mismatch"):
        Parameter.from_row(joint_param_row, default_ds, posterior_config=config)


def test_from_row_posterior_partial_parameters_mismatch_raises(
    joint_param_row, default_ds, posterior_file
):
    """source.parameters must match _variables_to_validate()."""
    # joint param targets coeff1 and coeff2, but config names something else
    config = {
        "parameters": ["fates_leafn_vert_scaler_coeff1", "wrong_param_b"],
        "files": [{"path": str(posterior_file), "array_indices": "all"}],
    }
    with pytest.raises(ValueError, match="mismatch"):
        Parameter.from_row(joint_param_row, default_ds, posterior_config=config)
