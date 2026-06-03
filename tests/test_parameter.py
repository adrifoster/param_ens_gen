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


# ===========================================================================
# DefaultParameter
# ===========================================================================


def test_default_validate_specs_passes(default_row, default_ds):
    """_validate_specs() does not raise for a correctly configured DefaultParameter."""
    Parameter.from_row(default_row, default_ds)


def test_default_missing_variable_raises(default_row, default_ds):
    """DefaultParameter with a missing parameter raises"""
    default_row["parameter_name"] = "nonexistent_var"
    with pytest.raises(ValueError, match="not found"):
        Parameter.from_row(default_row, default_ds)


def test_dim_mismatch_raises(default_row, default_ds):
    """DefaultParameter with a dimension mismatch raises"""
    default_row["parameter_name"] = "fates_leaf_slatop"
    default_row["coord"] = "[]"
    with pytest.raises(ValueError, match="dims"):
        Parameter.from_row(default_row, default_ds)


def test_default_get_default_pft(default_row, default_ds):
    """DefaultParameter.get_default returns the full PFT array."""
    param = Parameter.from_row(default_row, default_ds)
    result = param.get_default(default_ds)
    np.testing.assert_allclose(result, [0.010, 0.020, 0.030])


def test_default_get_default_scalar(scalar_row, default_ds):
    """DefaultParameter.get_default returns a scalar for scalar parameters."""
    param = Parameter.from_row(scalar_row, default_ds)
    result = param.get_default(default_ds)
    assert float(result) == pytest.approx(0.5)


def test_default_set_value_uniform_no_fixed(default_row, default_ds, working_ds):
    """DefaultParameter.set_value writes value to all array positions with a scalar."""
    param = Parameter.from_row(default_row, default_ds)
    param.set_value(working_ds, default_ds, value=0.025)
    np.testing.assert_allclose(
        working_ds["fates_leaf_slatop"].values,
        [0.025, 0.025, 0.025],
    )


def test_default_set_value_with_fixed_indices(default_row, default_ds, working_ds):
    """DefaultParameter.set_value leaves fixed PFTs at their default values."""
    param = Parameter.from_row(default_row, default_ds)
    param.set_value(
        working_ds,
        default_ds,
        value=0.025,
        fixed_indices={"fates_pft": [2]},  # fix PFT index 2 (0-based)
    )
    assert working_ds["fates_leaf_slatop"].values[0] == pytest.approx(0.025)
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(0.025)
    assert working_ds["fates_leaf_slatop"].values[2] == pytest.approx(
        0.030
    )  # unchanged


def test_default_set_value_sets_single_slot(default_param, working_ds, default_ds):
    """DefaultParameter.set_value writes to active_index position"""
    default_param.active_index = DimIndex(dim="fates_pft", index=1)
    default_param.set_value(working_ds, default_ds, 0.5)
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(0.5)


def test_default_set_value_with_active_index(default_row, default_ds, working_ds):
    """DefaultParameter.set_value writes only to active_index position."""
    param = Parameter.from_row(default_row, default_ds)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    param.set_value(working_ds, default_ds, 0.099)

    assert working_ds["fates_leaf_slatop"].values[0] == pytest.approx(
        0.010
    )  # unchanged
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(0.099)  # written
    assert working_ds["fates_leaf_slatop"].values[2] == pytest.approx(
        0.030
    )  # unchanged


def test_default_set_value_write_at_index_other_slots_unchanged(
    default_param, working_ds, default_ds
):
    """DefaultParameter.set_value doesn't write to other positions when active_index is on"""
    default_param.active_index = DimIndex(dim="fates_pft", index=1)
    default_param.set_value(working_ds, default_ds, 0.5)
    assert working_ds["fates_leaf_slatop"].values[0] == pytest.approx(
        default_ds["fates_leaf_slatop"].values[0]
    )
    assert working_ds["fates_leaf_slatop"].values[2] == pytest.approx(
        default_ds["fates_leaf_slatop"].values[2]
    )


def test_default_set_value_write_at_index_rejects_array_value(
    default_param, working_ds, default_ds
):
    """Test that set_value with an active_index raises with an array"""
    default_param.active_index = DimIndex(dim="fates_pft", index=1)
    with pytest.raises(ValueError, match="scalar"):
        default_param.set_value(working_ds, default_ds, np.array([0.01, 0.02]))


def test_default_set_value_scalar_value_broadcasts(
    default_param, working_ds, default_ds
):
    default_param.set_value(working_ds, default_ds, 0.025, None)
    np.testing.assert_array_equal(
        working_ds["fates_leaf_slatop"].values, [0.025, 0.025, 0.025]
    )


def test_default_set_value_array_value(default_param, working_ds, default_ds):
    """DefaultParameter.set_value can set a whole array"""
    value = np.array([0.011, 0.022, 0.033])
    default_param.set_value(working_ds, default_ds, value, None)
    np.testing.assert_array_equal(working_ds["fates_leaf_slatop"].values, value)


def test_default_set_value_fixed_indices_held_at_default(
    default_param, working_ds, default_ds
):
    """DefaultValue.set_value holds fixed_indices at default"""
    default_param.set_value(working_ds, default_ds, 0.025, {"fates_pft": [1]})
    result = working_ds["fates_leaf_slatop"].values
    assert result[0] == pytest.approx(0.025)
    assert result[1] == pytest.approx(default_ds["fates_leaf_slatop"].values[1])
    assert result[2] == pytest.approx(0.025)


def test_default_set_value_scalar_param(scalar_param, working_ds, default_ds):
    """DefaultValue.set_value correctly writes a scalar parameter"""
    scalar_param.set_value(working_ds, default_ds, 0.8, None)
    assert float(working_ds["fates_canopy_closure_thresh"].values) == pytest.approx(0.8)


def test_default_set_value_ds_not_mutated(default_param, working_ds, default_ds):
    """DefaultValue.set_value doesn't mutate the default dataset"""
    original = default_ds["fates_leaf_slatop"].values.copy()
    default_param.set_value(working_ds, default_ds, 0.025, {"fates_pft": [0]})
    np.testing.assert_array_equal(default_ds["fates_leaf_slatop"].values, original)


def test_default_set_value_scalar_fills_all_positions(
    default_row, default_ds, working_ds
):
    """DefaultParameter.set_value writes value to all array positions with a scalar."""
    param = Parameter.from_row(default_row, default_ds)
    working_ds["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(working_ds, default_ds, value=9.0)
    np.testing.assert_equal(
        working_ds["fates_leaf_slatop"].values,
        [9.0, 9.0, 9.0],
    )


def test_default_set_value_array_passthrough(default_row, default_ds, working_ds):
    """DefaultParameter.set_value writes can pass an array"""
    param = Parameter.from_row(default_row, default_ds)
    working_ds["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(working_ds, default_ds, value=np.array([4.0, 5.0, 6.0]))
    np.testing.assert_equal(
        working_ds["fates_leaf_slatop"].values,
        [4.0, 5.0, 6.0],
    )


def test_default_set_value_scalar_fixed_positions_held(
    default_row, default_ds, working_ds
):
    """DefaultParameter.set_value correctly holds indices fixed with an input scalar value"""
    param = Parameter.from_row(default_row, default_ds)
    working_ds["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(working_ds, default_ds, 9.0, {"fates_pft": [1]})
    np.testing.assert_equal(
        working_ds["fates_leaf_slatop"].values,
        [9.0, 2.0, 9.0],
    )


def test_default_set_value_array_fixed_positions_held(
    default_row, default_ds, working_ds
):
    """DefaultParameter.set_value correctly holds indices fixed with an input array value"""
    param = Parameter.from_row(default_row, default_ds)
    working_ds["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(
        working_ds, default_ds, np.array([7.0, 8.0, 9.0]), {"fates_pft": [0]}
    )
    np.testing.assert_equal(
        working_ds["fates_leaf_slatop"].values,
        [1.0, 8.0, 9.0],
    )


def test_default_set_value_all_positions_fixed_returns_original(
    default_row, default_ds, working_ds
):
    """DefaultParameter.set_value correctly can hold all indices fixed"""
    param = Parameter.from_row(default_row, default_ds)
    working_ds["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(
        working_ds, default_ds, np.array([7.0, 8.0, 9.0]), {"fates_pft": [0, 1, 2]}
    )
    np.testing.assert_equal(
        working_ds["fates_leaf_slatop"].values,
        np.array([1.0, 2.0, 3.0]),
    )


def test_default_set_value_shape_mismatch_raises(default_row, default_ds, working_ds):
    """DefaultParameter.set_value raises for a shape mismatch between value and param"""
    param = Parameter.from_row(default_row, default_ds)
    working_ds["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        param.set_value(working_ds, default_ds, np.ones(5), None)


def test_default_set_value_0d_array_input(scalar_row, default_ds, working_ds):
    """DefaultParameter.set_value can take a 0d array"""
    param = Parameter.from_row(scalar_row, default_ds)
    working_ds["fates_canopy_closure_thresh"].values = 1.0
    param.set_value(working_ds, default_ds, np.float64(9.0), None)
    assert float(working_ds["fates_canopy_closure_thresh"]) == pytest.approx(9.0)


def test_default_set_value_float_input(default_param, default_ds, working_ds):
    """DefaultParameter.set_value can take a float input with an active_index"""
    default_param.active_index = DimIndex("fates_pft", 1)
    default_param.set_value(working_ds, default_ds, 3.14)
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(3.14)


def test_default_set_value_0d_input(default_param, default_ds, working_ds):
    """DefaultParameter.set_value can take a 0d input with an active_index"""
    default_param.active_index = DimIndex("fates_pft", 1)
    default_param.set_value(working_ds, default_ds, np.float64(7.0))
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(7.0)


def test_default_set_value_single_element_array_input(
    default_param, default_ds, working_ds
):
    """DefaultParameter.set_value can take a single_element_array input with an active_index"""
    default_param.active_index = DimIndex("fates_pft", 1)
    default_param.set_value(working_ds, default_ds, np.array([2.5]))
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(2.5)


def test_default_set_value_multi_element_array_raises(
    default_param, default_ds, working_ds
):
    """DefaultParameter.set_value with active_index and a multi-element array raises"""
    default_param.active_index = DimIndex("fates_pft", 1)
    with pytest.raises(ValueError, match="scalar"):
        default_param.set_value(working_ds, default_ds, np.array([1.0, 2.0]))


# =============================================================================
# SlicedParameter
# =============================================================================


def test_sliced_param_validate_specs_passes(sliced_row, default_ds):
    """_validate_specs() does not raise for a correctly configured SlicedParameter."""
    Parameter.from_row(sliced_row, default_ds)


def test_sliced_param_validate_specs_missing_slice_dim_raises(default_row, default_ds):
    """_validate_specs() raises if slice_dim is missing"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "sliced"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["slice_dim"] = ""
    default_row["slice_index"] = "0"
    default_row["base_params"] = "fates_leaf_vcmax25top"

    with pytest.raises(ValueError, match="slice_dim"):
        Parameter.from_row(default_row, default_ds)


def test_sliced_param_validate_specs_missing_slice_index_raises(
    default_row, default_ds
):
    """_validate_specs() raises if slice_index is missing"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "sliced"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["slice_dim"] = "fates_leafage_class"
    default_row["slice_index"] = None
    default_row["base_params"] = "fates_leaf_vcmax25top"

    with pytest.raises(ValueError, match="slice_index"):
        Parameter.from_row(default_row, default_ds)


def test_sliced_param_validate_multiple_base_params_raises(default_row, default_ds):
    """_validate_specs() raises for multiple base_params"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "sliced"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["slice_dim"] = "fates_leafage_class"
    default_row["slice_index"] = "0"
    default_row["base_params"] = "['fates_leaf_vcmax25top', 'fates_leaf_slatop']"

    with pytest.raises(ValueError, match="Too many base_params"):
        Parameter.from_row(default_row, default_ds)


def test_sliced_get_default(sliced_row, default_ds):
    """SlicedParameter.get_default returns the slice at slice_index."""
    param = Parameter.from_row(sliced_row, default_ds)
    result = param.get_default(default_ds)
    np.testing.assert_allclose(result, [50.0, 60.0, 70.0])


def test_sliced_param_set_value_active_index_sets_correct_element(
    sliced_param, working_ds, default_ds
):
    """SlicedParameter.set_value with an active index sets the correct element"""
    sliced_param.active_index = DimIndex("fates_pft", 1)
    sliced_param.set_value(working_ds, default_ds, 99.0, None)
    result = working_ds["fates_leaf_vcmax25top"].values
    assert result[0, 1] == pytest.approx(99.0)


def test_sliced_param_set_value_active_index_leaves_other_indices_unchanged(
    sliced_param, working_ds, default_ds
):
    """SlicedParameter.set_value with an active index doesn't modify other indices"""
    sliced_param.active_index = DimIndex("fates_pft", 1)
    sliced_param.set_value(working_ds, default_ds, 99.0, None)
    result = working_ds["fates_leaf_vcmax25top"].values
    assert result[0, 0] == pytest.approx(
        default_ds["fates_leaf_vcmax25top"].values[0, 0]
    )
    assert result[0, 2] == pytest.approx(
        default_ds["fates_leaf_vcmax25top"].values[0, 2]
    )
    # other leafage_class row entirely untouched
    np.testing.assert_array_equal(
        result[1, :], default_ds["fates_leaf_vcmax25top"].values[1, :]
    )


def test_sliced_param_set_value_only_modifies_target_slice(
    sliced_param, working_ds, default_ds
):
    """SlicedParameter.set_value without active index modifies the correct slice"""
    sliced_param.set_value(working_ds, default_ds, 75.0, None)
    result = working_ds["fates_leaf_vcmax25top"].values
    np.testing.assert_array_equal(result[0, :], [75.0, 75.0, 75.0])
    # leafage_class row 1 untouched
    np.testing.assert_array_equal(
        result[1, :], default_ds["fates_leaf_vcmax25top"].values[1, :]
    )


def test_sliced_param_set_value_fixed_indices_respected(
    sliced_param, working_ds, default_ds
):
    """SlicedParameter.set_value correctly sets with fixed indices"""
    sliced_param.set_value(working_ds, default_ds, 75.0, {"fates_pft": [1]})
    result = working_ds["fates_leaf_vcmax25top"].values
    assert result[0, 0] == pytest.approx(75.0)
    assert result[0, 1] == pytest.approx(
        default_ds["fates_leaf_vcmax25top"].values[0, 1]
    )
    assert result[0, 2] == pytest.approx(75.0)


def test_sliced_param_set_value_array_value(sliced_param, working_ds, default_ds):
    """SlicedParameter.set_value can set a whole array"""
    value = np.array([21.0, 32.0, 43.0])
    sliced_param.set_value(working_ds, default_ds, value, None)
    np.testing.assert_array_equal(
        working_ds["fates_leaf_vcmax25top"].values[0, :], value
    )


def test_sliced_param_set_value_ds_not_mutated(sliced_param, working_ds, default_ds):
    """SlicedParameter.set_value doesn't modify the default ds"""
    original = default_ds["fates_leaf_vcmax25top"].values.copy()
    sliced_param.set_value(working_ds, default_ds, 75.0, {"fates_pft": [0]})
    np.testing.assert_array_equal(default_ds["fates_leaf_vcmax25top"].values, original)


def test_sliced_set_value_no_fixed(sliced_row, default_ds, working_ds):
    """SlicedParameter.set_value writes value across all indices at the slice."""
    param = Parameter.from_row(sliced_row, default_ds)
    param.set_value(working_ds, default_ds, value=80.0)

    arr = working_ds["fates_leaf_vcmax25top"].values
    # slice_index=0 (leafage_class dim) should be updated
    np.testing.assert_allclose(arr[0, :], [80.0, 80.0, 80.0])
    # other leafage_class rows unchanged
    np.testing.assert_allclose(arr[1, :], [40.0, 50.0, 60.0])


def test_sliced_set_value_with_fixed_indices(sliced_row, default_ds, working_ds):
    """SlicedParameter.set_value leaves fixed PFTs at their default values."""
    param = Parameter.from_row(sliced_row, default_ds)
    param.set_value(
        working_ds,
        default_ds,
        value=80.0,
        fixed_indices={"fates_pft": [0]},
    )
    arr = working_ds["fates_leaf_vcmax25top"].values
    assert arr[0, 0] == pytest.approx(50.0)  # fixed — unchanged
    assert arr[0, 1] == pytest.approx(80.0)  # written
    assert arr[0, 2] == pytest.approx(80.0)  # written


def test_sliced_set_value_with_active_index(sliced_row, default_ds, working_ds):
    """SlicedParameter.set_value writes only to (slice_index, active_index) cell."""
    param = Parameter.from_row(sliced_row, default_ds)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    param.set_value(working_ds, default_ds, value=99.0)

    arr = working_ds["fates_leaf_vcmax25top"].values
    assert arr[0, 0] == pytest.approx(50.0)  # unchanged
    assert arr[0, 1] == pytest.approx(99.0)  # written
    assert arr[0, 2] == pytest.approx(70.0)  # unchanged
    np.testing.assert_allclose(arr[1, :], [40.0, 50.0, 60.0])  # other slice unchanged


# =============================================================================
# ScaleFromRootParameter
# =============================================================================


def test_validate_specs_passes_for_scale(scale_from_root_row, default_ds):
    """_validate_specs() does not raise for a correctly configured ScaleFromRootParameter."""
    Parameter.from_row(scale_from_root_row, default_ds)


def test_scale_from_root_missing_root_param_raises(default_row, default_ds):
    """ScaleFromRootParam raises if missing the root parameter"""
    default_row["parameter_name"] = "smpsc_delta"
    default_row["coord"] = "['fates_pft']"
    default_row["param_type"] = "scale_from_root"
    default_row["param_min"] = "-600000"
    default_row["param_max"] = "-20000"
    default_row["root_param"] = ""
    default_row["base_params"] = "fates_nonhydro_smpsc"

    with pytest.raises(ValueError, match="root_param"):
        Parameter.from_row(default_row, default_ds)


def test_scale_from_root__missing_base_param_raises(default_row, default_ds):
    """ScaleFromRootParam raises if missing the base parameter"""
    default_row["parameter_name"] = "smpsc_delta"
    default_row["coord"] = "['fates_pft']"
    default_row["param_type"] = "scale_from_root"
    default_row["param_min"] = "-600000"
    default_row["param_max"] = "-20000"
    default_row["root_param"] = "fates_nonhydro_smpso"
    default_row["base_params"] = ""

    with pytest.raises(ValueError, match="base_param"):
        Parameter.from_row(default_row, default_ds)


def test_scale_from_root_get_default(scale_from_root_row, default_ds):
    """ScaleFromRootParameter.get_default returns root parameter values."""
    param = Parameter.from_row(scale_from_root_row, default_ds)
    result = param.get_default(default_ds)
    np.testing.assert_allclose(result, [-100000.0, -110000.0, -120000.0])


def test_scale_from_root_set_value_no_fixed(
    scale_from_root_row, default_ds, working_ds
):
    """ScaleFromRootParameter.set_value adds delta to root values for all indices."""
    param = Parameter.from_row(scale_from_root_row, default_ds)
    delta = -10000.0
    param.set_value(working_ds, default_ds, value=delta)

    expected = np.array([-50000.0, -60000.0, -70000.0]) + delta
    np.testing.assert_allclose(working_ds["fates_nonhydro_smpsc"].values, expected)


def test_scale_from_root_set_value_with_fixed(
    scale_from_root_row, default_ds, working_ds
):
    """ScaleFromRootParameter.set_value leaves fixed indices at their default values."""
    param = Parameter.from_row(scale_from_root_row, default_ds)
    param.set_value(
        working_ds,
        default_ds,
        value=-10000.0,
        fixed_indices={"fates_pft": [0]},
    )
    arr = working_ds["fates_nonhydro_smpsc"].values
    assert arr[0] == pytest.approx(-100000.0)  # fixed — default value
    assert arr[1] == pytest.approx(-70000.0)  # root + delta
    assert arr[2] == pytest.approx(-80000.0)  # root + delta


def test_scale_from_root_set_value_with_active_index(
    scale_from_root_row, default_ds, working_ds
):
    """ScaleFromRootParameter.set_value writes delta only at active_index."""
    param = Parameter.from_row(scale_from_root_row, default_ds)
    param.active_index = DimIndex(dim="fates_pft", index=0)
    param.set_value(working_ds, default_ds, value=-5000.0)

    arr = working_ds["fates_nonhydro_smpsc"].values
    assert arr[0] == pytest.approx(-55000.0)  # root[0] + delta
    assert arr[1] == pytest.approx(-110000.0)  # unchanged
    assert arr[2] == pytest.approx(-120000.0)  # unchanged


def test_scale_from_root_set_value_adds_delta_to_root(
    scale_param, working_ds, default_ds
):
    """ScaleFromRootParameter.set_value correctly adds delta to the root param"""
    # simulate driver having written root already
    working_ds["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.active_index = DimIndex("fates_pft", 1)
    scale_param.set_value(working_ds, default_ds, -50000.0)
    # index 1: root[1] + delta = -65000 + -50000 = -115000
    assert working_ds["fates_nonhydro_smpsc"].values[1] == pytest.approx(-115000.0)


def test_scale_from_root_set_value_other_slots_unchanged(
    scale_param, working_ds, default_ds
):
    """ScaleFromRootParameter.set_value correctly writes only to the active index"""
    working_ds["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.active_index = DimIndex("fates_pft", 1)
    scale_param.set_value(working_ds, default_ds, -50000.0)
    assert working_ds["fates_nonhydro_smpsc"].values[0] == pytest.approx(
        default_ds["fates_nonhydro_smpsc"].values[0]
    )
    assert working_ds["fates_nonhydro_smpsc"].values[2] == pytest.approx(
        default_ds["fates_nonhydro_smpsc"].values[2]
    )


def test_scale_from_root_set_value_reads_written_root_not_default(
    scale_param, working_ds, default_ds
):
    """Root must be read from working_ds, not default_ds."""
    working_ds["fates_nonhydro_smpso"].values[:] = -999000.0
    scale_param.active_index = DimIndex("fates_pft", 1)
    scale_param.set_value(working_ds, default_ds, -1000.0)
    assert working_ds["fates_nonhydro_smpsc"].values[1] == pytest.approx(-1000000.0)


def test_scale_from_root_set_value_scalar_delta(scale_param, working_ds, default_ds):
    """ScaleFromRootParameter.set_value can set a scalar delta"""
    working_ds["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.set_value(working_ds, default_ds, -50000.0, None)
    np.testing.assert_allclose(
        working_ds["fates_nonhydro_smpsc"].values,
        [-105000.0, -115000.0, -125000.0],
    )


def test_write_full_array_delta(scale_param, working_ds, default_ds):
    """ScaleFromRoot parameter.set_value can set a whole array"""
    working_ds["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    delta = np.array([-10000.0, -20000.0, -30000.0])
    scale_param.set_value(working_ds, default_ds, delta, None)
    np.testing.assert_allclose(
        working_ds["fates_nonhydro_smpsc"].values,
        [-65000.0, -85000.0, -105000.0],
    )


def test_write_full_fixed_indices_held_at_default(scale_param, working_ds, default_ds):
    """ScaleFromRoot.set_value can set an array with fixed indices held"""
    working_ds["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.set_value(working_ds, default_ds, -50000.0, {"fates_pft": [1]})
    assert working_ds["fates_nonhydro_smpsc"].values[0] == pytest.approx(-105000.0)
    assert working_ds["fates_nonhydro_smpsc"].values[1] == pytest.approx(
        default_ds["fates_nonhydro_smpsc"].values[1]
    )
    assert working_ds["fates_nonhydro_smpsc"].values[2] == pytest.approx(-125000.0)


def test_write_full_reads_written_root_not_default(scale_param, working_ds, default_ds):
    """Root must be read from working_ds, not default_ds."""
    working_ds["fates_nonhydro_smpso"].values[:] = -999000.0
    scale_param.set_value(working_ds, default_ds, -1000.0, None)
    np.testing.assert_allclose(
        working_ds["fates_nonhydro_smpsc"].values,
        [-1000000.0, -1000000.0, -1000000.0],
    )


def test_default_ds_not_mutated(scale_param, working_ds, default_ds):
    """ScaleFromRootParameter.set_value does not mutate default_ds"""
    original_smpso = default_ds["fates_nonhydro_smpso"].values.copy()
    original_smpsc = default_ds["fates_nonhydro_smpsc"].values.copy()
    working_ds["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.set_value(working_ds, default_ds, -50000.0, {"fates_pft": [0]})
    np.testing.assert_array_equal(
        default_ds["fates_nonhydro_smpso"].values, original_smpso
    )
    np.testing.assert_array_equal(
        default_ds["fates_nonhydro_smpsc"].values, original_smpsc
    )


# =============================================================================
# JointParameter
# =============================================================================


def test_validate_specs_passes_for_joint_parameter(
    joint_param_row, default_ds, posterior_config
):
    """_validate_specs() does not raise for a correctly configured JointParameter."""
    Parameter.from_row(joint_param_row, default_ds, posterior_config=posterior_config)


def test_joint_param_empty_base_params_raises(
    default_row, default_ds, posterior_config
):
    default_row["parameter_name"] = "fates_leafn_vert_scaler"
    default_row["coord"] = "['fates_pft']"
    default_row["param_type"] = "joint"
    default_row["strategy"] = "posterior"
    default_row["param_min"] = ""
    default_row["param_max"] = ""
    default_row["base_params"] = "[]"
    with pytest.raises(ValueError, match="base_params"):
        Parameter.from_row(default_row, default_ds, posterior_config=posterior_config)


def test_joint_get_default(joint_param_row, default_ds, posterior_config):
    """JointParameter.get_default returns a list of arrays, one per base_param."""
    param = Parameter.from_row(
        joint_param_row, default_ds, posterior_config=posterior_config
    )
    result = param.get_default(default_ds)
    assert isinstance(result, list)
    assert len(result) == 2
    np.testing.assert_allclose(result[0], [0.012, 0.015, 0.005])
    np.testing.assert_allclose(result[1], [2.1, 2.5, 2.6])


def test_joint_set_value(joint_param_row, default_ds, working_ds, posterior_config):
    """JointParameter.set_value writes each array to its corresponding variable."""
    param = Parameter.from_row(
        joint_param_row, default_ds, posterior_config=posterior_config
    )
    new_values = [
        np.array([0.5, 0.6, 0.7]),
        np.array([5.0, 6.0, 7.0]),
    ]
    param.set_value(working_ds, default_ds, value=new_values)
    np.testing.assert_allclose(
        working_ds["fates_leafn_vert_scaler_coeff1"].values, [0.5, 0.6, 0.7]
    )
    np.testing.assert_allclose(
        working_ds["fates_leafn_vert_scaler_coeff2"].values, [5.0, 6.0, 7.0]
    )


def test_joint_set_value_wrong_length_raises(
    joint_param_row, default_ds, working_ds, posterior_config
):
    """JointParameter.set_value raises ValueError if value length mismatches base_params."""
    param = Parameter.from_row(
        joint_param_row, default_ds, posterior_config=posterior_config
    )
    with pytest.raises(ValueError, match="expected 2 arrays"):
        param.set_value(working_ds, default_ds, value=[np.array([0.5, 0.6, 0.7])])


def test_joint_set_value_with_index_sets_single_slot_in_all_params(
    joint_param, working_ds, default_ds
):
    """JointParameter.set_value with active_index just sets a single index"""
    joint_param.active_index = DimIndex("fates_pft", 1)
    joint_param.set_value(
        working_ds,
        default_ds,
        [0.99, 3.5],
        None,
    )
    assert working_ds["fates_leafn_vert_scaler_coeff1"].values[1] == pytest.approx(0.99)
    assert working_ds["fates_leafn_vert_scaler_coeff2"].values[1] == pytest.approx(3.5)
    # other indices untouched
    for i in [0, 2]:
        assert working_ds["fates_leafn_vert_scaler_coeff1"].values[i] == pytest.approx(
            default_ds["fates_leafn_vert_scaler_coeff1"].values[i]
        )
        assert working_ds["fates_leafn_vert_scaler_coeff2"].values[i] == pytest.approx(
            default_ds["fates_leafn_vert_scaler_coeff2"].values[i]
        )


def test_joint_set_value_broadcasts_arrays_to_all_params(
    joint_param, working_ds, default_ds
):
    """JointParameter.set_value can broadcast arrays to all indices"""
    coeff1 = np.array([0.010, 0.011, 0.012])
    coeff2 = np.array([2.2, 2.3, 2.4])
    joint_param.set_value(working_ds, default_ds, [coeff1, coeff2], None)
    np.testing.assert_array_equal(
        working_ds["fates_leafn_vert_scaler_coeff1"].values, coeff1
    )
    np.testing.assert_array_equal(
        working_ds["fates_leafn_vert_scaler_coeff2"].values, coeff2
    )


def test_joint_set_value_fixed_indices_respected_across_all_params(
    joint_param, working_ds, default_ds
):
    """JointParameter.set_value can hold fixed_indices at default"""
    coeff1 = np.array([0.010, 0.011, 0.012])
    coeff2 = np.array([2.2, 2.3, 2.4])
    joint_param.set_value(working_ds, default_ds, [coeff1, coeff2], {"fates_pft": [1]})
    # index 1 held at default for both
    assert working_ds["fates_leafn_vert_scaler_coeff1"].values[1] == pytest.approx(
        default_ds["fates_leafn_vert_scaler_coeff1"].values[1]
    )
    assert working_ds["fates_leafn_vert_scaler_coeff2"].values[1] == pytest.approx(
        default_ds["fates_leafn_vert_scaler_coeff2"].values[1]
    )
    # other indices written
    assert working_ds["fates_leafn_vert_scaler_coeff1"].values[0] == pytest.approx(
        0.010
    )
    assert working_ds["fates_leafn_vert_scaler_coeff2"].values[2] == pytest.approx(2.4)


def test_joint_set_value_default_ds_not_mutated(joint_param, working_ds, default_ds):
    """JointParameter.set_value does not change the default_ds"""
    original_c1 = default_ds["fates_leafn_vert_scaler_coeff1"].values.copy()
    original_c2 = default_ds["fates_leafn_vert_scaler_coeff2"].values.copy()
    joint_param.set_value(
        working_ds, default_ds, [np.ones(3), np.ones(3)], {"fates_pft": [0]}
    )
    np.testing.assert_array_equal(
        default_ds["fates_leafn_vert_scaler_coeff1"].values, original_c1
    )
    np.testing.assert_array_equal(
        default_ds["fates_leafn_vert_scaler_coeff2"].values, original_c2
    )
