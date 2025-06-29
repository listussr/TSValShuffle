import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import mock_open, patch
from src.ts_val_shuffle.features_generation import FeaturesGenerator

VALID_CONFIG = {
    "features": {
        "lags": [],
        "shift": [],
        "rolling": [],
        "absolute_time": [],
        "relative_time": []
    },
    "options": {
        "drop_na": True
    } 
}

class TestClassInitialization:
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(VALID_CONFIG))
    def test_init_success(self, mock_file):
        instance = FeaturesGenerator("dummy_path.json")
        assert instance.path == "dummy_path.json"
        assert instance.config == VALID_CONFIG

class TestLoadConfig:
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(VALID_CONFIG))
    def test_load_config_success(self, mock_file):
        instance = FeaturesGenerator("dummy_path.json")
        assert instance.config == VALID_CONFIG

class TestCheckConfig:
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "features": {
            "lags": [],
            "shift": [],
            "rolling": [],
            "absolute_time": [], 
            "relative_time": []
        },
        "options": {"drop_na": True}
    }))
    def test_check_config_valid(self, mock_file):
        instance = FeaturesGenerator("valid_config.json")
        assert instance.config["options"]["drop_na"] is True
    

class TestAggregationFunctions:
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(VALID_CONFIG))
    def test_agg_functions(self, mock_file):
        instance = FeaturesGenerator("dummy_path.json")
        test_data = [1, 2, 3]
        assert instance.__agg_functions["mean"](test_data) == 2.0
        assert instance.__agg_functions["sum"](test_data) == 6

class TestTimeUnits:
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(VALID_CONFIG))
    def test_time_units(self, mock_file):
        instance = FeaturesGenerator("dummy_path.json")
        test_date = pd.Series([datetime(2023, 1, 1)])
        assert instance.__time_units["day"](test_date)[0] == 1

class TestRelativeTimeUnits:
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(VALID_CONFIG))
    def test_relative_time_units(self, mock_file):
        instance = FeaturesGenerator("dummy_path.json")
        date1 = pd.Series([datetime(2023, 1, 2)])
        date2 = pd.Series([datetime(2023, 1, 1)])
        assert instance.__relative_time_units["day"](date1, date2)[0] == 1.0

class TestCycleFunctions: 
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(VALID_CONFIG))
    def test_cycle_functions(self, mock_file):
        instance = FeaturesGenerator("dummy_path.json")
        assert instance.__cycle_function["sin"](0) == 0.0


@pytest.fixture
def sample_time_series():
    idx = pd.date_range("2023-01-01", periods=5)
    return pd.DataFrame({
        'value': [10, 20, 30, 40, 50],
        'temperature': [22, 23, 24, 25, 26],
        'event_time': [datetime(2023,1,1), datetime(2023,1,2), 
                      datetime(2023,1,3), datetime(2023,1,4), datetime(2023,1,5)]
    }, index=idx)

@pytest.fixture
def valid_config():
    return {
        "features": {
            "lags": [
                {"name": "value_lag_1", "source": "value", "window": 1},
                {"name": "temp_lag_2", "source": "temperature", "window": 2}
            ],
            "shift": [],
            "rolling": [],
            "absolute_time": [],
            "relative_time": []
        },
        "options": {"drop_na": True}
    }

@pytest.fixture
def generator_with_lags(valid_config):
    with patch("builtins.open", mock_open(read_data=json.dumps(valid_config))):
        return FeaturesGenerator("dummy.json")

class TestGenerateLagFeatures:    
    @pytest.fixture
    def lag_config(self):
        return {
            "features": {
                "lags": [
                    {"name": "value_lag_1", "source": "value", "window": 1},
                    {"name": "temp_lag_2", "source": "temperature", "window": 2}
                ],
                "shift": [],
                "rolling": [],
                "absolute_time": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
    
    def test_generate_lags_basic(self, lag_config, sample_time_series):
        with patch("builtins.open", mock_open(read_data=json.dumps(lag_config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_lag_features(sample_time_series)
            
            expected = pd.DataFrame({
                'value_lag_1': [None, 10, 20, 30, 40],
                'temp_lag_2': [None, None, 22, 23, 24]
            }, index=sample_time_series.index)
            
            pd.testing.assert_frame_equal(result, expected)

    def test_generate_lags_empty_config(self, sample_time_series):
        config = {
            "features": {
                "lags": None,
                "shift": [],
                "rolling": [],
                "absolute_time": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_lag_features(sample_time_series)
            assert result.empty

    def test_generate_lags_invalid_config(self, sample_time_series):
        config = {
            "features": {
                "lags": [{"name": "lag", "source": "value"}],
                "shift": [],
                "rolling": [],
                "absolute_time": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            generator = FeaturesGenerator("dummy.json")
            with pytest.raises((KeyError, ValueError)):
                generator._FeaturesGenerator__generate_lag_features(sample_time_series)


class TestGenerateShiftFeatures:    
    def test_generate_lags_basic(self, generator_with_lags, sample_time_series):
        test_method = generator_with_lags._FeaturesGenerator__generate_lag_features
        result = test_method(sample_time_series)
        
        expected = pd.DataFrame({
            'value_lag_1': [None, 10, 20, 30],
            'temp_lag_2': [None, None, 22, 23]
        }, index=sample_time_series.index)
        
        pd.testing.assert_frame_equal(result, expected)

    def test_generate_lags_empty_config(self, sample_time_series):
        config = {
            "features": {
                "lags": None,
                "shift": [],
                "rolling": [],
                "absolute_time": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_lag_features(sample_time_series)
            assert result.empty

    def test_generate_lags_invalid_config(self, sample_time_series):
        config = {
            "features": {
                "lags": [{"name": "lag", "source": "value"}],
                "shift": [],
                "rolling": [],
                "absolute_time": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            generator = FeaturesGenerator("dummy.json")
            with pytest.raises((KeyError, ValueError)):
                generator._FeaturesGenerator__generate_lag_features(sample_time_series)

class TestGenerateShiftFeatures:
    @pytest.fixture
    def shift_config(self):
        return {
            "features": {
                "shift": [
                    {
                        "name": "value_diff_1",
                        "source": "value",
                        "window": 1,
                        "operation": "diff"
                    }
                ],
                "lags": [],
                "rolling": [],
                "absolute_time": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
    
    def test_generate_shift_basic(self, shift_config, sample_time_series):
        with patch("builtins.open", mock_open(read_data=json.dumps(shift_config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_shift_features(sample_time_series)
            
            expected = pd.DataFrame({
                'value_diff_1': [None, 10, 10, 10, 10]
            }, index=sample_time_series.index)
            
            pd.testing.assert_frame_equal(result, expected)

    def test_generate_shift_empty_config(self, sample_time_series):
        config = {
            "features": {"shift": None},
            "options": {"drop_na": True}
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_shift_features(sample_time_series)
            assert result.empty

class TestGenerateRolling:    
    @pytest.fixture
    def rolling_config(self):
        return {
            "features": {
                "rolling": [
                    {
                        "name": "mean_3",
                        "source": "value",
                        "window": 3,
                        "agg": "mean"
                    }
                ],
                "shift": [],
                "lags": [],
                "absolute_time": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
    
    def test_generate_rolling_mean(self, rolling_config, sample_time_series):
        with patch("builtins.open", mock_open(read_data=json.dumps(rolling_config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_rolling(sample_time_series)
            
            expected = pd.DataFrame({
                'mean_3': [None, None, 20.0, 30.0, 40.0]
            }, index=sample_time_series.index)
            
            pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_generate_rolling_with_lag(self, sample_time_series):
        config = {
            "features": {
                "rolling": [
                    {
                        "name": "mean_3_lag1",
                        "source": "value",
                        "window": 3,
                        "agg": "mean",
                        "lag_window": 1
                    }
                ]
            },
            "options": {"drop_na": True}
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_rolling(sample_time_series)
            expected = pd.DataFrame({
                'mean_3_lag1': [None, None, None, 20.0, 30.0]
            }, index=sample_time_series.index)
            
            pd.testing.assert_frame_equal(result, expected, check_dtype=False)

class TestGenerateCalendar:  
    @pytest.fixture
    def calendar_config(self):
        return {
            "features": {
                "absolute_time": [{
                    "name": "day_of_week",
                    "source": "event_time",
                    "time_unit": "dayofweek"
                }],
                "shift": [],
                "lags": [],
                "rolling": [],
                "relative_time": []
            },
            "options": {"drop_na": True}
        }
    
    def test_generate_calendar_basic(self, calendar_config, sample_time_series):
        with patch("builtins.open", mock_open(read_data=json.dumps(calendar_config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_calendar(sample_time_series)
            
            expected = pd.DataFrame({
                'day_of_week': [6, 0, 1, 2, 3]
            }, index=sample_time_series.index)
            
            expected['day_of_week'] = expected['day_of_week'].astype('int64')
            result['day_of_week'] = result['day_of_week'].astype('int64')
            
            pd.testing.assert_frame_equal(result, expected, check_dtype=False)

class TestGenerateRelativeTime:
    @pytest.fixture
    def relative_time_config(self):
        return {
            "features": {
                "relative_time": [
                    {
                        "name": "days_since_start",
                        "source": "event_time",
                        "time_unit": "day"
                    }
                ],
                "shift": [],
                "lags": [],
                "rolling": [],
                "absolute_time": []
            },
            "options": {"drop_na": True}
        }
    
    def test_generate_relative_time(self, relative_time_config, sample_time_series):
        with patch("builtins.open", mock_open(read_data=json.dumps(relative_time_config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_relative_time(sample_time_series)
            
            expected = pd.DataFrame({
                'days_since_start': [0.0, 1.0, 2.0, 3.0, 4.0]
            }, index=sample_time_series.index)
            
            pd.testing.assert_frame_equal(result, expected)

    def test_generate_relative_time_scaled(self, sample_time_series):
        config = {
            "features": {
                "relative_time": [
                    {
                        "name": "norm_days",
                        "source": "event_time",
                        "time_unit": "day",
                        "range": [0, 1]
                    }
                ]
            },
            "options": {"drop_na": True}
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(config))):
            generator = FeaturesGenerator("dummy.json")
            result = generator._FeaturesGenerator__generate_relative_time(sample_time_series)
            
            assert result['norm_days'].min() == 0.0
            assert result['norm_days'].max() == 1.0