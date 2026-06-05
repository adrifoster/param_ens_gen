"""Tests for DefaultParameter class"""

from __future__ import annotations

import numpy as np
import pytest

from param_ens_gen.parameter import DimIndex, Parameter


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
    """Test that set_value with a scalar broadcasts to the whole array"""
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
