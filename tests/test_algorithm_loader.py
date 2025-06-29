import pytest
from unittest.mock import MagicMock
from src.ts_val_shuffle.algorithm_loader import _ModelAdapter
import pandas as pd
import numpy as np

@pytest.fixture
def mock_model():
    return MagicMock()

@pytest.fixture
def wrapper(mock_model):
    wrapper = _ModelAdapter()
    wrapper.model = mock_model
    return wrapper

@pytest.fixture
def sample_df():
    return pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=3), 'y': [1, 2, 3]})

class Test_ModelAdapterMethods:
    def test_add_regressor_calls_model_method(self, wrapper, mock_model):
        wrapper.add_regressor('feature', {'prior_scale': 0.5})
        mock_model.add_regressor.assert_called_once_with('feature', prior_scale=0.5)

    def test_add_regressor_raises_if_no_method(self, wrapper, mock_model):
        del mock_model.add_regressor
        with pytest.raises(TypeError):
            wrapper.add_regressor('feature', {})

    def test_add_seasonality_calls_model_method(self, wrapper, mock_model):
        wrapper.add_seasonality({'name': 'monthly', 'period': 30.5})
        mock_model.add_seasonality.assert_called_once_with(name='monthly', period=30.5)

    def test_add_seasonality_raises_if_no_method(self, wrapper, mock_model):
        del mock_model.add_seasonality
        with pytest.raises(TypeError):
            wrapper.add_seasonality({})

    def test_make_future_dataframe_returns_result(self, wrapper, mock_model, sample_df):
        expected_df = sample_df
        mock_model.make_future_dataframe.return_value = expected_df
        result = wrapper.make_future_dataframe({'periods': 5})
        mock_model.make_future_dataframe.assert_called_once_with(periods=5)
        assert result.equals(expected_df)

    def test_make_future_dataframe_raises_if_no_method(self, wrapper, mock_model):
        del mock_model.make_future_dataframe
        with pytest.raises(TypeError):
            wrapper.make_future_dataframe({})

    def test_plot_calls_model_method_and_returns(self, wrapper, mock_model, sample_df):
        plot_result = 'plot object'
        mock_model.plot.return_value = plot_result
        result = wrapper.plot(sample_df, {'xlabel': 'Date'})
        mock_model.plot.assert_called_once_with(sample_df, xlabel='Date')
        assert result == plot_result

    def test_plot_raises_if_no_method(self, wrapper, mock_model, sample_df):
        del mock_model.plot
        with pytest.raises(TypeError):
            wrapper.plot(sample_df, {})

    def test_plot_components_calls_model_method_and_returns(self, wrapper, mock_model, sample_df):
        comp_result = 'components plot'
        mock_model.plot_components.return_value = comp_result
        result = wrapper.plot_components(sample_df, {'ylabel': 'Value'})
        mock_model.plot_components.assert_called_once_with(sample_df, ylabel='Value')
        assert result == comp_result

    def test_plot_components_raises_if_no_method(self, wrapper, mock_model, sample_df):
        del mock_model.plot_components
        with pytest.raises(TypeError):
            wrapper.plot_components(sample_df, {})

    def test_fit_calls_model_fit_and_sets_fitted_model(self, wrapper, mock_model, sample_df):
        fitted_mock = MagicMock()
        mock_model.fit.return_value = fitted_mock
        wrapper.fit({'df': sample_df})
        mock_model.fit.assert_called_once_with(df=sample_df)
        assert wrapper.fitted_model is fitted_mock

    def test_forecast_uses_fitted_model_if_present(self, wrapper):
        fitted_mock = MagicMock()
        forecast_series = pd.Series([1, 2, 3])
        fitted_mock.forecast.return_value = forecast_series
        wrapper.fitted_model = fitted_mock
        result = wrapper.forecast({'horizon': 3})
        fitted_mock.forecast.assert_called_once_with(horizon=3)
        assert isinstance(result, pd.DataFrame)
        assert (result[0] == forecast_series).all()

    def test_forecast_raises_if_no_forecast_method(self, wrapper, mock_model):
        del mock_model.forecast
        wrapper.fitted_model = None
        with pytest.raises(TypeError):
            wrapper.forecast({})

    def test_get_forecast_returns_dataframe(self, wrapper, mock_model):
        forecast_series = pd.Series([1, 2, 3])
        mock_model.get_forecast.return_value = (forecast_series,)
        result = wrapper.get_forecast({'periods': 3})
        mock_model.get_forecast.assert_called_once_with(periods=3)
        assert isinstance(result, pd.DataFrame)

    def test_get_forecast_raises_if_no_method(self, wrapper, mock_model):
        del mock_model.get_forecast
        with pytest.raises(TypeError):
            wrapper.get_forecast({})

    def test_predict_with_fitted_model_returns_dataframe(self, wrapper):
        pred_series = pd.Series([1, 2, 3])
        fitted_mock = MagicMock()
        fitted_mock.predict.return_value = pred_series
        wrapper.fitted_model = fitted_mock
        result = wrapper.predict({'data': 'test'})
        fitted_mock.predict.assert_called_once_with(data='test')
        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns

    def test_predict_with_model_returns_dataframe_with_numpy_array(self, wrapper, mock_model):
        np_pred = np.array([1, 2, 3])
        mock_model.predict.return_value = np_pred
        wrapper.fitted_model = None
        result = wrapper.predict({})
        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns

    def test_predict_with_model_returns_dataframe_with_tuple(self, wrapper, mock_model):
        series = pd.Series([1, 2, 3])
        mock_model.predict.return_value = (series,)
        wrapper.fitted_model = None
        result = wrapper.predict({})
        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns

    def test_predict_with_model_returns_dataframe_with_yhat_column(self, wrapper, mock_model):
        df = pd.DataFrame({'yhat': [1, 2, 3]})
        mock_model.predict.return_value = df
        wrapper.fitted_model = None
        result = wrapper.predict({})
        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns
        assert 'yhat' not in result.columns