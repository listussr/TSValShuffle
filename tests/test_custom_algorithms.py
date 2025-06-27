import pytest
import pandas as pd
import numpy as np
from src.ts_val_shuffle.custom_algorithms import (
    _ConstPrediction,
    _RollingMean,
    _CatBoostRegressor_,
    _SARIMAX_,
    _Holt_,
    _SimpleExpSmoothing_,
    _ExponentialSmoothing_,
    _FourierModel
)

class TestConstPrediction:
    @pytest.fixture
    def sample_series(self):
        return pd.Series([1, 2, 2, 3, 5])
    
    @pytest.mark.parametrize("type_name,expected_value", [
        ('Moda', 2),
        ('Mean', 2.6),
        ('Min', 1),
        ('Max', 5),
        ('Median', 2)
    ])
    def test_fit_predict(self, sample_series, type_name, expected_value):
        model = _ConstPrediction(type=type_name)
        model.fit(sample_series)
        predictions, value = model.predict(pd.Series([1, 2, 3]))
        assert value == expected_value
        assert all(predictions == expected_value)
        assert len(predictions) == 3

    
    def test_nan_values(self):
        model = _ConstPrediction(type='Median')
        model.fit(pd.Series([1, np.nan, 3]))
        predictions, value = model.predict(pd.Series([1, 2]))
        assert value == 2.0
    
class TestRollingMean:
    @pytest.fixture
    def sample_series(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7])
    
    def test_fit_predict(self, sample_series):
        model = _RollingMean(window_size=3, weights_coeffs=[], weights_type='new')
        model.fit(sample_series[:-2])
        predictions = model.predict(sample_series[-2:])
        assert len(predictions) == 2
        assert round(predictions[0], 2) == 1.55
        assert round(predictions[1], 2) == 0.90

class TestCatBoostRegressor_Initialization:
    def test_initialization_with_parameters(self):
        params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6
        }
        model = _CatBoostRegressor_(**params)
        model.fit(pd.DataFrame(np.random.rand(10, 5)), pd.Series(np.random.rand(10)))
        predict_params = {
            'X': pd.DataFrame(np.random.rand(10, 5)),
        }
        model.predict(**predict_params)

class TestFourierModel:
    def test_init_missing_order(self):
        with pytest.raises(ValueError, match="Missed argument \[order\]"):
            _FourierModel(some_param=1)
    
    def test_init_incorrect_order_type(self):
        with pytest.raises(ValueError, match="Incorrect \[order\] type. \[order\] must be an integer"):
            _FourierModel(order="not_an_integer")
    
    @pytest.fixture
    def initialized_model(self):
        return _FourierModel(order=3)
    
    @pytest.fixture
    def valid_data(self):
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = np.array([0.5, 0.8, 1.2])
        return X, y
    
    def test_fit_invalid_X_type(self, initialized_model, valid_data):
        _, y = valid_data
        with pytest.raises(ValueError, match="X must be a pandas DataFrame"):
            initialized_model.fit(X=np.array([[1], [2]]), y=y)
    
    def test_fit_invalid_y_type(self, initialized_model, valid_data):
        X, _ = valid_data
        with pytest.raises(ValueError, match="y must be numpy array or pandas Series"):
            initialized_model.fit(X=X, y=[1, 2, 3])
    
    def test_fit_shape_mismatch(self, initialized_model):
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = np.array([0.5, 0.8])
        with pytest.raises(ValueError, match="Length mismatch between X and y"):
            initialized_model.fit(X=X, y=y)
    
    @pytest.fixture
    def fitted_model(self, initialized_model, valid_data):
        X, y = valid_data
        initialized_model.fit(X=X, y=y)
        return initialized_model
    
    def test_predict_before_fit(self, initialized_model):
        X = pd.DataFrame({'feature': [1, 2, 3]})
        with pytest.raises(RuntimeError, match="Model is not fitted yet"):
            initialized_model.predict(X)
    
    def test_predict_invalid_X_type(self, fitted_model):
        with pytest.raises(ValueError, match="X must be a pandas DataFrame"):
            fitted_model.predict(X=np.array([[1], [2]]))
    
    def test_predict_features_mismatch(self, fitted_model):
        X = pd.DataFrame({'different_feature': [1, 2, 3]})
        with pytest.raises(ValueError, match="Features mismatch with training data"):
            fitted_model.predict(X)
    
    def test_fit_empty_data(self, initialized_model):
        X = pd.DataFrame({'feature': []})
        y = np.array([])
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            initialized_model.fit(X=X, y=y)
    
    def test_predict_empty_data(self, fitted_model):
        X = pd.DataFrame({'feature': []})
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            fitted_model.predict(X)

class TestSARIMAX:    
    @pytest.fixture
    def sample_data(self):
        return pd.Series(np.random.randn(100))
    
    @pytest.mark.parametrize("order,seasonal_order,error", [
        (("1",0,0), (0,0,0,0), "order must be a tuple of 3 integers"),
        ((1,0,0), ("0",0,0,0), "seasonal_order must be a tuple of 4 integers"),
        ((1,-1,0), (0,0,0,0), "order values must be non-negative"),
        ((1,0,0), (0,-1,0,0), "seasonal_order values must be non-negative"),
    ])
    def test_init_validation(self, order, seasonal_order, error):
        with pytest.raises(ValueError, match=error):
            _SARIMAX_(order=order, seasonal_order=seasonal_order)
    
    def test_fit_invalid_endog_type(self):
        model = _SARIMAX_()
        with pytest.raises(ValueError, match="endog must be array-like"):
            model.fit(endog="invalid_type")
    
    def test_fit_empty_data(self):
        model = _SARIMAX_()
        with pytest.raises(ValueError, match="endog cannot be empty"):
            model.fit(endog=[])
    
    @pytest.mark.parametrize("method", ["predict", "forecast", "summary"])
    def test_methods_before_fit(self, method):
        model = _SARIMAX_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            getattr(model, method)()
    
    def test_predict_before_fit(self):
        model = _SARIMAX_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.predict()

    def test_forecast_before_fit(self):
        model = _SARIMAX_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.forecast(steps=1)

class TestSimpleExpSmoothingErrors:   
    @pytest.fixture
    def sample_data(self):
        return pd.Series(np.random.randn(100))
    
    def test_fit_invalid_endog_type(self):
        model = _SimpleExpSmoothing_()
        with pytest.raises(ValueError, match="endog must be array-like"):
            model.fit(endog="invalid_type")
    
    def test_fit_empty_data(self):
        model = _SimpleExpSmoothing_()
        with pytest.raises(ValueError, match="endog cannot be empty"):
            model.fit(endog=[])
    
    def test_predict_before_fit(self):
        model = _SimpleExpSmoothing_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.predict()
    
    def test_forecast_before_fit(self):
        model = _SimpleExpSmoothing_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.forecast(steps=1)
    
    def test_summary_before_fit(self):
        model = _SimpleExpSmoothing_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.summary()

class TestHoltErrors:
    @pytest.fixture
    def sample_data(self):
        return pd.Series(np.random.randn(100))
    
    def test_fit_invalid_endog_type(self):
        model = _Holt_()
        with pytest.raises(ValueError, match="endog must be array-like"):
            model.fit(endog="invalid_type")
    
    def test_fit_empty_data(self):
        model = _Holt_()
        with pytest.raises(ValueError, match="endog cannot be empty"):
            model.fit(endog=pd.Series([]))
    
    def test_predict_before_fit(self):
        model = _Holt_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.predict()
    
    def test_forecast_before_fit(self):
        model = _Holt_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.forecast(steps=1)
    
    def test_summary_before_fit(self):
        model = _Holt_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.summary()

class TestExponentialSmoothingErrors:
    @pytest.fixture
    def sample_data(self):
        return pd.Series(np.random.randn(100))
    
    def test_fit_invalid_endog_type(self):
        model = _ExponentialSmoothing_()
        with pytest.raises(ValueError, match="endog must be array-like"):
            model.fit(endog="invalid_type")
    
    def test_fit_empty_data(self):
        model = _ExponentialSmoothing_()
        with pytest.raises(ValueError, match="endog cannot be empty"):
            model.fit(endog=pd.Series([]))
    
    def test_predict_before_fit(self):
        model = _ExponentialSmoothing_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.predict()
    
    def test_forecast_before_fit(self):
        model = _ExponentialSmoothing_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.forecast(steps=5)
    
    def test_summary_before_fit(self):
        model = _ExponentialSmoothing_()
        with pytest.raises(ValueError, match="Model is not fitted yet"):
            model.summary()