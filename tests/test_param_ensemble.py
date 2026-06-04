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


def test_missing_default_param_file_raises(ensemble_param_dir, tmp_path,
                                           posterior_config_file):
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=tmp_path / "nonexistent.nc",
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LatinHypercubeEnsemble(config)


def test_missing_posterior_sources_file_raises(ensemble_param_dir, tmp_path,
                                           default_param_file):
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=tmp_path/ "nonexistant.yaml",
        default_param_file=default_param_file,
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LatinHypercubeEnsemble(config)
        