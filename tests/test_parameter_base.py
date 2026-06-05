"""Tests for Parameter class"""

from __future__ import annotations

import numpy as np
import pytest


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
    """from_row with a uniform sampler must not trigger any posterior validation."""
    param = DefaultParameter(default_row, default_ds)
    assert param.sampler is not None


def test_from_row_posterior_parameters_mismatch_raises(
    joint_param_row, default_ds, posterior_file
):
    """from_row with posterior parameters that don't match the config raises a ValueError."""
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
    """from_row with posterior parameters that don't match the config raises a ValueError."""
    # joint param targets coeff1 and coeff2, but config names something else
    config = {
        "parameters": ["fates_leafn_vert_scaler_coeff1", "wrong_param_b"],
        "files": [{"path": str(posterior_file), "array_indices": "all"}],
    }
    with pytest.raises(ValueError, match="mismatch"):
        Parameter.from_row(joint_param_row, default_ds, posterior_config=config)


def test_from_row_non_broadcast_multi_dim_raises(
    default_row, default_ds, posterior_file
):
    """from_row with multi-dimensional posterior parameters without broadcast raises"""
    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["coord"] = "['fates_leafage_class', 'fates_pft']"
    default_row["strategy"] = "posterior"
    default_row["param_min"] = ""
    default_row["param_max"] = ""
    default_row["base_params"] = "fates_leaf_vcmax25top"
    print(default_row)
    config = {
        "parameters": ["fates_leaf_vcmax25top"],
        "files": [
            # non-broadcast: specific indices only
            {"path": str(posterior_file), "array_indices": [0, 1]}
        ],
    }
    with pytest.raises(ValueError, match="broadcast mode"):
        Parameter.from_row(default_row, default_ds, posterior_config=config)


def test_from_row_array_indices_out_of_bounds_raises(
    joint_param_row, default_ds, posterior_file
):
    """from_row raises if the array_indices on the posterior config doesn't match the parameter"""
    config = {
        "parameters": [
            "fates_leafn_vert_scaler_coeff1",
            "fates_leafn_vert_scaler_coeff2",
        ],
        "files": [
            # n_indices=3 (N_PFTS); index 3 is out of bounds
            {"path": str(posterior_file), "array_indices": [0, 1, 3]}
        ],
    }
    with pytest.raises(ValueError, match="dimension mismatch"):
        Parameter.from_row(joint_param_row, default_ds, posterior_config=config)


# ===========================================================================
# Parameter free dims
# ===========================================================================


def test_free_dims_is_none_on_scalar_param(scalar_param):
    """free_dims on a scalar param is None"""
    assert scalar_param.free_dims is None


def test_free_dims_on_1d_param(default_param):
    """free_dims for a 1-d param is correct"""
    assert default_param.free_dims == ["fates_pft"]


def test_free_dims_returns_list(default_param):
    """free_dims returns a list"""
    assert isinstance(default_param.free_dims, list)


def test_free_dims_excludes_slice_dim(sliced_param):
    """free_dims on a sliced parameter should exclude the slice"""
    assert sliced_param.free_dims == ["fates_pft"]


def test_free_dims_includes_all_dims(multi_dim_param):
    """free_dims on a multi-dimensional parameter should include all dimensions"""
    assert multi_dim_param.free_dims == ["dim_1", "dim_2"]


# ===========================================================================
# Parameter n_indices
# ===========================================================================


def test_n_indices_scalar(scalar_param):
    """n_indices of a scalar parameter is [1]"""
    assert scalar_param.n_indices == [1]


def test_n_indices_1d(default_param):
    """n_indices of a 1d parameter is correct"""
    assert default_param.n_indices == [3]


def test_n_indices_sliced(sliced_param):
    """n_indices of a sliced parameter is just the slice"""
    assert sliced_param.n_indices == [3]


def test_n_indices_multi_param(multi_dim_param):
    """n_indices of a multi-parameter has all dims"""
    assert multi_dim_param.n_indices == [2, 3]


# =============================================================================
# set_value dispatch
# =============================================================================


def test_set_value_dispatches_to_write_at_index_when_expanded(
    default_param, working_ds, default_ds, mocker
):
    """set_value dispatches to write_at_index when expanded dims are used"""
    default_param.active_index = DimIndex("fates_pft", 1)
    mock_at_index = mocker.patch.object(default_param, "_write_at_index")
    mock_full = mocker.patch.object(default_param, "_write_full")

    default_param.set_value(working_ds, default_ds, 9.0)

    mock_at_index.assert_called_once_with(working_ds, DimIndex("fates_pft", 1), 9.0)
    mock_full.assert_not_called()


def test_set_value_dispatches_to_write_full_when_not_expanded(
    default_param, working_ds, default_ds, mocker
):
    """set_value dispatches to write_full when expanded dims are not being used"""
    assert default_param.active_index is None
    mock_at_index = mocker.patch.object(default_param, "_write_at_index")
    mock_full = mocker.patch.object(default_param, "_write_full")

    default_param.set_value(working_ds, default_ds, 9.0)

    mock_full.assert_called_once_with(working_ds, default_ds, 9.0, {})
    mock_at_index.assert_not_called()


def test_set_value_none_fixed_indices_becomes_empty_dict(
    default_param, working_ds, default_ds, monkeypatch
):
    """fixed is set to an empty dict when it is None"""
    received = {}

    def capture(ds, default_ds, value, fixed_indices):
        received["fixed_indices"] = fixed_indices

    monkeypatch.setattr(default_param, "_write_full", capture)
    default_param.set_value(working_ds, default_ds, 9.0, fixed_indices=None)
    assert received["fixed_indices"] == {}


# =============================================================================
# Sample
# =============================================================================


def test_default_parameter_sample(default_param, default_ds):
    """sample() returns a value with the expected value."""
    result = default_param.sample(0.5, default_ds)
    assert result == pytest.approx(0.0275)


def test_default_parameter_sample_bounds(default_param, default_ds):
    """sample() at 0.0 and 1.0 returns min and max."""
    assert default_param.sample(0.0, default_ds) == pytest.approx(0.005)
    assert default_param.sample(1.0, default_ds) == pytest.approx(0.05)


def test_scalar_parameter_sample(scalar_param, default_ds):
    """sample() returns a value with the expected value."""
    result = scalar_param.sample(0.5, default_ds)
    assert result == pytest.approx(0.8)


def test_scalar_parameter_sample_bounds(scalar_param, default_ds):
    """sample() at 0.0 and 1.0 returns min and max."""
    assert scalar_param.sample(0.0, default_ds) == pytest.approx(0.7)
    assert scalar_param.sample(1.0, default_ds) == pytest.approx(0.9)


def test_percent_parameter_sample(percent_row, default_ds):
    """sample() returns a value with the expected value."""
    param = Parameter.from_row(percent_row, default_ds)
    result = param.sample(0.5, default_ds)
    expected = default_ds["fates_leaf_vcmax25top"].values
    np.testing.assert_allclose(result, expected)


def test_joint_parameter_sample(joint_param, default_ds):
    """sample() returns a list of arrays, one per base_param."""
    result = joint_param.sample(0.5, default_ds)

    # should be a list with one entry per base_param
    assert len(result) == 2

    # each entry should be an array with one value per PFT
    assert result[0].shape == (3,)
    assert result[1].shape == (3,)

    # values should be within the posterior range
    assert np.all(result[0] >= 0.0) and np.all(result[0] <= 0.95)  # coeff1 range
    assert np.all(result[1] >= 10.0) and np.all(result[1] <= 19.5)  # coeff2 range
