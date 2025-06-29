import pytest
from unittest.mock import patch, MagicMock
from src.ts_val_shuffle.validate import Validator
from src.ts_val_shuffle.utils import _validate_json_keys, _validate_json_values
import pandas as pd

@pytest.fixture(autouse=True)
def patch_validation_functions(monkeypatch):
    monkeypatch.setattr('src.ts_val_shuffle.utils._validate_json_keys', _validate_json_keys)
    monkeypatch.setattr('src.ts_val_shuffle.utils._validate_json_values', _validate_json_values)


@pytest.fixture
def validator_instance():
    return Validator()

class TestValidator:
    def test_validate_init_params_missing_key(self, validator_instance):
        validator_instance._Validator__init_params = {
            "params": {}
        }
        with pytest.raises(KeyError):
            validator_instance._Validator__validate_init_params()

    def test_validate_init_params_wrong_type(self, validator_instance):
        validator_instance._Validator__init_params = {
            "model_name": 123,
            "params": {}
        }
        with pytest.raises(TypeError):
            validator_instance._Validator__validate_init_params()

    def test_validate_split_params_valid(self, validator_instance):
        validator_instance._Validator__split_params = {
            "method": "rolling",
            "n_splits": 5,
            "test_size": 10
        }
        validator_instance._Validator__validate_split_params()

    def test_validate_split_params_missing_key(self, validator_instance):
        validator_instance._Validator__split_params = {
            "method": "rolling",
            "test_size": 10
        }
        with pytest.raises(KeyError):
            validator_instance._Validator__validate_split_params()

    def test_validate_split_params_wrong_type(self, validator_instance):
        validator_instance._Validator__split_params = {
            "method": "rolling",
            "n_splits": "five",
            "test_size": 10
        }
        with pytest.raises(TypeError):
            validator_instance._Validator__validate_split_params()

    def test_validate_validation_params_default_values(self, validator_instance):
        validator_instance._Validator__validation_params = {
            "metric": "WAPE",
            "target_feature": "y",
            "time_feature": "ds"
        }
        validator_instance._Validator__validate_validation_params()
        assert validator_instance._Validator__validation_params["shuffling"] is False
        assert validator_instance._Validator__validation_params["extra_fit_params"] == {}
        assert validator_instance._Validator__validation_params["add_feature_params"] == {}

    def test_validate_validation_params_all_fields(self, validator_instance):
        validator_instance._Validator__validation_params = {
            "metric": "WAPE",
            "target_feature": "y",
            "time_feature": "ds",
            "shuffling": True,
            "extra_fit_params": {"param": 1},
            "add_feature_params": {"param": 2}
        }
        validator_instance._Validator__validate_validation_params()

    def test_validate_validation_params_wrong_type(self, validator_instance):
        validator_instance._Validator__validation_params = {
            "metric": "WAPE",
            "target_feature": "y",
            "time_feature": "ds",
            "shuffling": "yes"
        }
        with pytest.raises(TypeError):
            validator_instance._Validator__validate_validation_params()

    def test_validate_raises_if_generated_ts_none(self, validator_instance):
        validator_instance.generated_ts = None
        validator_instance.adapter = type('Adapter', (), {'adapter_config': {}})()
        validator_instance.split_handler = object()
        with pytest.raises(RuntimeError, match=r"Field \[generated_ts\] is None"):
            validator_instance._Validator__validate("MAPE", "target", "time")

    def test_validate_raises_if_generated_ts_empty(self, validator_instance):
        validator_instance.generated_ts = pd.DataFrame()
        validator_instance.adapter = type('Adapter', (), {'adapter_config': {}})()
        validator_instance.split_handler = object()
        with pytest.raises(RuntimeError, match=r"Field \[generated_ts\] is None"):
            validator_instance._Validator__validate("MAPE", "target", "time")

    def test_validate_raises_if_adapter_config_none(self, validator_instance):
        validator_instance.generated_ts = pd.DataFrame({'ds': [1], 'y': [1]})
        validator_instance.adapter = type('Adapter', (), {'adapter_config': None})()
        validator_instance.split_handler = object()
        with pytest.raises(RuntimeError, match=r"Field \[adapter\] is None"):
            validator_instance._Validator__validate("MAPE", "target", "time")

    def test_validate_raises_if_split_handler_none(self, validator_instance):
        validator_instance.generated_ts = pd.DataFrame({'ds': [1], 'y': [1]})
        validator_instance.adapter = type('Adapter', (), {'adapter_config': {}})()
        validator_instance.split_handler = None
        with pytest.raises(RuntimeError, match=r"Field \[split_handler\] is None"):
            validator_instance._Validator__validate("MAPE", "target", "time")

    def test_set_model_calls_adapter_set_model(self, validator_instance):
        mock_adapter = MagicMock()
        validator_instance.adapter = mock_adapter

        model_name = "ARIMA"
        params = {"order": (1, 0, 1)}

        result = validator_instance._Validator__set_model(model_name, params)

        mock_adapter.set_model.assert_called_once_with(model_name, params)
        assert validator_instance._Validator__init_params == params
        assert result is validator_instance 

    def test_set_data_sets_ts_and_returns_self(self, validator_instance):
        ts = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=3), "y": [1, 2, 3]})

        result = validator_instance.set_data(ts)

        assert validator_instance.ts.equals(ts)
        assert result is validator_instance

    def test_set_generator_raises_if_ts_none(self, validator_instance):
        validator_instance.ts = None
        with pytest.raises(RuntimeError, match=r"Field \[ts\] is None"):
            validator_instance.set_generator("config.json")

    def test_set_generator_raises_if_ts_empty(self, validator_instance):
        validator_instance.ts = pd.DataFrame()
        with pytest.raises(RuntimeError, match=r"Field \[ts\] is None"):
            validator_instance.set_generator("config.json")