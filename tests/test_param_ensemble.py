"""Tests for ParamEnsemble"""

import pytest
import xarray as xr
from pathlib import Path

import numpy as np
import pandas as pd

from param_ens_gen.param_ensemble import (
    ParamEnsemble,
    LatinHypercubeEnsemble,
    OneAtATimeParameterEnsemble,
    EnsembleMemberSample,
    ParameterSample,
)

from param_ens_gen.ensemble_config import LatinHypercubeConfig, OneAtATimeConfig


def test_from_dict_missing_ensemble_type(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    with pytest.raises(ValueError, match="ensemble_type.*required"):
        ParamEnsemble.from_dict(
            {
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_unknown_ensemble_type(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    with pytest.raises(ValueError, match="Unknown ensemble_type"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "BadType",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_bad_key(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    with pytest.raises(TypeError, match="Invalid config key"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "LatinHypercube",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "not_a_real_key": 42,
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_valid_latin_hypercube(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": str(posterior_config_file),
        }
    )
    assert isinstance(ensemble, LatinHypercubeEnsemble)


def test_from_dict_valid_oat(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "OAT",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "posterior_sources": posterior_config_file,
        }
    )
    assert isinstance(ensemble, OneAtATimeParameterEnsemble)


def test_from_dict_invalid_oat(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    with pytest.raises(TypeError, match="Invalid config key"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "OAT",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "ensemble_members": 5,
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_valid_latin_hypercube_creates_ensemble_dir(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
        }
    )
    assert isinstance(ensemble.ensemble_dir, Path)
    assert ensemble.ensemble_dir.exists()


def test_from_dict_valid_latin_hypercube_correctly_sets_attributes(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
            "fixed_indices": {"fates_pft": [1]},
        }
    )
    assert ensemble.file_prefix == "test"
    assert isinstance(ensemble.default_ds, xr.Dataset)
    assert ensemble.fixed_indices == {"fates_pft": [1]}


def test_from_dict_valid_latin_hypercube_fixed_indices_empty_if_unset(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
        }
    )
    assert ensemble.fixed_indices == {}


def test_param_list_unknown_parameter_raises(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        posterior_sources=posterior_config_file,
        param_list=["fates_leaf_slatop", "not_a_real_param"],
    )
    with pytest.raises(ValueError, match="not found in main.csv"):
        LatinHypercubeEnsemble(config)


def test_missing_default_param_file_raises(
    ensemble_param_dir, tmp_path, posterior_config_file
):
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=tmp_path / "nonexistent.nc",
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LatinHypercubeEnsemble(config)


def test_missing_posterior_sources_file_raises(
    ensemble_param_dir, tmp_path, default_param_file
):
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=tmp_path / "nonexistant.yaml",
        default_param_file=default_param_file,
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LatinHypercubeEnsemble(config)


def test_from_dict_params_correct(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
        }
    )
    assert len(ensemble.params) == 6


def test_from_dict_param_list_subsets_correctly(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    param_list = ["fates_leaf_slatop", "fates_leaf_vcmax25top"]
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
            "param_list": param_list,
        }
    )
    names = [p.spec.name for p in ensemble.params]
    assert len(ensemble.params) == 2
    assert set(param_list) == set(names)


def test_ensemble_params_types(lh_ensemble):
    """All params are Parameter instances."""
    from param_ens_gen.parameter import Parameter

    assert all(isinstance(p, Parameter) for p in lh_ensemble.params)


def test_ensemble_params_names(lh_ensemble, tmp_path):
    """Parameter names match what's in main.csv."""
    param_data = pd.read_csv(tmp_path / "main.csv")
    param_names = param_data.parameter_name.unique()
    names = [p.spec.name for p in lh_ensemble.params]
    assert set(param_names) == set(names)


def test_sort_params_root_before_dependent(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """sort_params ensures root parameter is written before scale_from_root dependent."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
    )
    ensemble = LatinHypercubeEnsemble(config)
    names = [p.spec.name for p in ensemble.params]
    assert names.index("fates_nonhydro_smpso") < names.index("smpsc_delta")


def test_fixed_indices_invalid_dimension_raises(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """fixed_indices with an unknown dimension raises ValueError."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"not_a_dim": [0, 1]},
    )
    with pytest.raises(ValueError, match="not found in default dataset"):
        LatinHypercubeEnsemble(config)


def test_fixed_indices_out_of_range_raises(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """fixed_indices with out-of-range indices raises ValueError."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"fates_pft": [99]},  # default_ds has N_PFTS=3
    )
    with pytest.raises(ValueError, match="out-of-range indices"):
        LatinHypercubeEnsemble(config)


def test_fixed_indices_valid(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """Valid fixed_indices constructs successfully."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"fates_pft": [0, 1]},
    )
    ensemble = LatinHypercubeEnsemble(config)
    assert ensemble.fixed_indices == {"fates_pft": [0, 1]}
    
def test_fixed_indices_negative_index_raises(ensemble_param_dir, default_param_file, 
                                             posterior_config_file, tmp_path):
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"fates_pft": [-1]},
    )
    with pytest.raises(ValueError, match="out-of-range indices"):
        LatinHypercubeEnsemble(config)


def test_build_lh_shape(lh_ensemble):
    lh = lh_ensemble.build_lh()
    assert lh.shape == (5, lh_ensemble.num_params)
    assert np.all(lh >= 0.0) and np.all(lh <= 1.0)


def test_build_lh_prebuilt(lh_ensemble):
    prebuilt = np.random.default_rng(0).random((5, lh_ensemble.num_params))
    lh = lh_ensemble.build_lh(prebuilt=prebuilt)
    np.testing.assert_array_equal(lh, prebuilt)


def test_build_lh_prebuilt_wrong_samples(lh_ensemble):
    bad = np.random.default_rng(0).random((10, lh_ensemble.num_params))
    with pytest.raises(ValueError, match="shape"):
        lh_ensemble.build_lh(prebuilt=bad)
        

def test_build_lh_prebuilt_wrong_params(lh_ensemble):
    bad = np.random.default_rng(0).random((5, 10))
    with pytest.raises(ValueError, match="shape"):
        lh_ensemble.build_lh(prebuilt=bad)


def test_build_lh_zero_params(lh_ensemble):
    lh_ensemble.num_params = 0
    lh = lh_ensemble.build_lh()
    assert lh.shape == (5, 0)


def test_lh_create_samples_length(lh_ensemble):
    samples = lh_ensemble.create_samples()
    assert len(samples) == 5


def test_lh_create_samples_structure(lh_ensemble):
    samples = lh_ensemble.create_samples()
    for sample in samples:
        assert len(sample) == lh_ensemble.num_params
        for ps in sample:
            assert 0.0 <= ps.normalized_value <= 1.0


def test_oat_create_samples_length(oat_ensemble):
    """OAT produces 2 samples per parameter (min and max)."""
    samples = oat_ensemble.create_samples()
    assert len(samples) == 2 * oat_ensemble.num_params


def test_oat_create_samples_values(oat_ensemble):
    """OAT samples are always 0.0 or 1.0."""
    samples = oat_ensemble.create_samples()
    values = [s.parameter_samples[0].normalized_value for s in samples]
    assert set(values) == {0.0, 1.0}


def test_lh_create_ensemble_member(lh_ensemble):
    samples = lh_ensemble.create_samples()
    ds = lh_ensemble.create_ensemble_member(samples[0])
    assert "fates_leaf_slatop" in ds


def test_oat_create_ensemble_member(oat_ensemble):
    samples = oat_ensemble.create_samples()
    ds = oat_ensemble.create_ensemble_member(samples[0])
    assert "fates_leaf_slatop" in ds


def test_oat_create_ensemble_member_wrong_length(oat_ensemble, lh_ensemble):
    """OAT raises if sample has more than one ParameterSample."""
    lh_samples = lh_ensemble.create_samples()
    with pytest.raises(ValueError, match="exactly one"):
        oat_ensemble.create_ensemble_member(lh_samples[0])
        
def test_lh_create_ensemble_key(lh_ensemble):
    samples = lh_ensemble.create_samples()
    key = lh_ensemble.create_ensemble_key(samples)
    assert "ensemble" in key.columns
    assert len(key) == 5


def test_oat_create_ensemble_key(oat_ensemble):
    samples = oat_ensemble.create_samples()
    key = oat_ensemble.create_ensemble_key(samples)
    assert "ensemble" in key.columns
    assert "direction" in key.columns
    assert set(key["direction"]) == {"minimum", "maximum"}
    
def test_oat_create_ensemble_key_wrong_length_raises(oat_ensemble, lh_ensemble):
    """create_ensemble_key raises if any sample has more than one ParameterSample."""
    lh_samples = lh_ensemble.create_samples()
    with pytest.raises(ValueError, match="exactly one"):
        oat_ensemble.create_ensemble_key(lh_samples)

def test_oat_create_ensemble_key_bad_direction_raises(oat_ensemble):
    """create_ensemble_key raises if normalized_value is not 0.0 or 1.0."""
    param = oat_ensemble.params[0]
    bad_sample = EnsembleMemberSample([ParameterSample(param, 0.5)])
    with pytest.raises(ValueError, match="expects only 0.0 or 1.0"):
        oat_ensemble.create_ensemble_key([bad_sample])
    
def test_create_ensemble_writes_files(lh_ensemble, tmp_path):
    lh_ensemble.create_ensemble()
    output_files = list(lh_ensemble.ensemble_dir.glob("test_*.nc"))
    assert len(output_files) == 5
    key_file = lh_ensemble.ensemble_dir / "test_key.csv"
    assert key_file.exists()
    list_file = lh_ensemble.ensemble_dir / "test.txt"
    assert list_file.exists()