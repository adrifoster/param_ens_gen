"""Tests for ParamEnsemble"""

import pytest
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

def test_from_dict_missing_ensemble_type(ensemble_param_dir, default_param_file, tmp_path):
    with pytest.raises(ValueError, match="ensemble_type.*required"):
        ParamEnsemble.from_dict({
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
        })
        
def test_from_dict_unknown_ensemble_type(ensemble_param_dir, default_param_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown ensemble_type"):
        ParamEnsemble.from_dict({
            "ensemble_type": "BadType",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
        })


def test_from_dict_bad_key(ensemble_param_dir, default_param_file, tmp_path):
    with pytest.raises(TypeError, match="Invalid config key"):
        ParamEnsemble.from_dict({
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "not_a_real_key": 42,
        })


def test_from_dict_valid_latin_hypercube(ensemble_param_dir, default_param_file, tmp_path):
    ensemble = ParamEnsemble.from_dict({
        "ensemble_type": "LatinHypercube",
        "param_dir": str(ensemble_param_dir),
        "ensemble_dir": str(tmp_path / "ensemble"),
        "file_prefix": "test",
        "default_param_file": str(default_param_file),
        "ensemble_members": 5,
    })
    assert isinstance(ensemble, LatinHypercubeEnsemble)
    
    
def test_from_dict_valid_oat(ensemble_param_dir, default_param_file, tmp_path):
    ensemble = ParamEnsemble.from_dict({
        "ensemble_type": "OAT",
        "param_dir": str(ensemble_param_dir),
        "ensemble_dir": str(tmp_path / "ensemble"),
        "file_prefix": "test",
        "default_param_file": str(default_param_file),
    })
    assert isinstance(ensemble, OneAtATimeParameterEnsemble)
    
    
def test_from_dict_invalid_oat(ensemble_param_dir, default_param_file, tmp_path):
    with pytest.raises(TypeError, match="Invalid config key"):
        ParamEnsemble.from_dict({
            "ensemble_type": "OAT",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
        })
        

def test_from_dict_valid_latin_hypercube_reads_param_dir(ensemble_param_dir, default_param_file, tmp_path):
    ensemble = ParamEnsemble.from_dict({
        "ensemble_type": "LatinHypercube",
        "param_dir": str(ensemble_param_dir),
        "ensemble_dir": str(tmp_path / "ensemble"),
        "file_prefix": "test",
        "default_param_file": str(default_param_file),
        "ensemble_members": 5,
    })
    
    