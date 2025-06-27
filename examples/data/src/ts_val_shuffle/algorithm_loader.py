from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    RANSACRegressor,
    Ridge,
    TheilSenRegressor,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from prophet import Prophet

from .custom_algorithms import (
    _ConstPrediction, 
    _CrostonTSB, 
    _RollingMean,
    _CatBoostRegressor_,
    _FourierModel,
    _SimpleExpSmoothing_,
    _Holt_,
    _ExponentialSmoothing_,
    _SARIMAX_,
)

import pandas as pd
import numpy as np

class _ModelAdapter:
    """
    Класс-адаптер для шаблонизации различных алгоритмов
    """
    def __init__(self) -> None:
        self.__models = {
            "Elastic Net": {"model": ElasticNet, "shuffle": True, "fit_type": 'sklearn'},
            "Huber": {"model": HuberRegressor, "shuffle": True, "fit_type": 'sklearn'},
            "Lasso": {"model": Lasso, "shuffle": True, "fit_type": 'sklearn'},
            #"Polynomial": 
            "RANSAC": {"model": RANSACRegressor, "shuffle": True, "fit_type": 'sklearn'},
            "Ridge": {"model": Ridge, "shuffle": True, "fit_type": 'sklearn'},
            "TheilSen": {"model": TheilSenRegressor, "shuffle": True, "fit_type": 'sklearn'},
            "Random Forest": {"model": RandomForestRegressor, "shuffle": True, "fit_type": 'sklearn'},
            "Exponential Smoothing": {"model": _SimpleExpSmoothing_, "shuffle": False, "fit_type": 'statsmodels'},
            "Holt": {"model": _Holt_, "shuffle": False, "fit_type": 'statsmodels'},
            "Holt Winters": {"model": _ExponentialSmoothing_, "shuffle": False, "fit_type": 'statsmodels'},
            "SARIMA": {"model": _SARIMAX_, "shuffle": False, "fit_type": 'sarimax'},
            "catboost": {"model": _CatBoostRegressor_, "shuffle": True, "fit_type": 'sklearn'},
            "Prophet": {"model": Prophet, "shuffle": True, "fit_type": 'prophet'},
            "ConstPrediction": {"model": _ConstPrediction, "shuffle": True, "fit_type": 'sklearn_like'},
            "CrostonTSB": {"model": _CrostonTSB, "shuffle": False, "fit_type": 'croston'},
            "RollingMean": {"model": _RollingMean, "shuffle": True, "fit_type": 'sklearn_like'},
            "Fourie": {"model": _FourierModel, "shuffle": False, "fit_type": 'sklearn'},
        }

        self.model = None
        self.adapter_config = None


    def __load_model(self, model_name: str, kwargs: dict) -> None:
        """
        Метод установки модели в шаблон

        Args:
            model_name (str): Название модели
            kwargs (dict): Словарь с гиперпараметрами модели

        Raises:
            KeyError: Ошибка отсутствия указанной модели в словаре с моделями
        """
        if not model_name in self.__models.keys():
            raise KeyError(f"Incorrect algorithm name ({model_name}). Required:\n{self.__models}")
        model_class = self.__models[model_name]["model"]
        self.adapter_config = self.__models[model_name]
        self.model = model_class(**kwargs)

    def set_model(self, model_name: str, kwargs: dict) -> None:
        """
        Публичный метод установки модели в шаблон

        Args:
            model_name (str): Название модели
            kwargs (dict): Словарь с гиперпараметрами модели
        """
        self.__load_model(model_name, kwargs)


    def add_regressor(self, feature_name: str, kwargs: dict):
        """
        Метод для упрощения использования метода add_regressor() у модели

        Args:
            feature_name (str): Название параметра
            kwargs (dict): Словарь с параметрами

        Raises:
            TypeError: Ошибка отстутствия у класса модели данного метода
        """
        if hasattr(self.model, "add_regressor"):
            self.model.add_regressor(feature_name, **kwargs)
        else:
            raise TypeError(f"Model class {self.model.__class__} does not have method [add_regressor()]")


    def add_seasonality(self, kwargs: dict):
        """
        Метод для упрощения использования метода add_seasonality() у модели

        Args:
            kwargs (dict): Словарь с параметрами

        Raises:
            TypeError: Ошибка отстутствия у класса модели данного метода
        """
        if hasattr(self.model, "add_seasonality"):
            self.model.add_seasonality(**kwargs)
        else:
            raise TypeError(f"Model class {self.model.__class__} does not have method [add_seasonality()]")


    def make_future_dataframe(self, kwargs: dict):
        """
        Метод для упрощения использования метода make_future_dataframe() у модели

        Args:
            kwargs (dict): Словарь с параметрами

        Raises:
            TypeError: Ошибка отстутствия у класса модели данного метода
        """
        if hasattr(self.model, "make_future_dataframe"):
            return self.model.make_future_dataframe(**kwargs)
        else:
            raise TypeError(f"Model class {self.model.__class__} does not have method [make_future_dataframe()]")


    def plot(self, forecast: pd.DataFrame, kwargs: dict):
        """
        Метод для упрощения использования метода plot() у модели

        Args:
            kwargs (dict): Словарь с параметрами

        Raises:
            TypeError: Ошибка отстутствия у класса модели данного метода
        """
        if hasattr(self.model, "plot"):
            return self.model.plot(forecast, **kwargs)
        else:
            raise TypeError(f"Model class {self.model.__class__} does not have method [plot()]")
        
    
    def plot_components(self, forecast: pd.DataFrame, kwargs: dict):
        """
        Метод для упрощения использования метода plot_components() у модели

        Args:
            kwargs (dict): Словарь с параметрами

        Raises:
            TypeError: Ошибка отстутствия у класса модели данного метода
        """
        if hasattr(self.model, "plot_components"):
            return self.model.plot_components(forecast, **kwargs)
        else:
            raise TypeError(f"Model class {self.model.__class__} does not have method [plot_components()]")


    def fit(self, kwargs: dict):
        """
        Метод для обобщения обучения моделей

        Args:
            kwargs (dict): Словарь с данными для обучения
        """
        self.fitted_model = self.model.fit(**kwargs)

    def forecast(self, kwargs: dict) -> pd.DataFrame:
        """
        Метод для обобщения предсказания на несколько классов моделей

        Args:
            kwargs (dict): Словарь с данными для предсказаяния

        Returns:
            pd.DataFrame: Датафрейм с результатами предсказания
        """
        if hasattr(self.model, "forecast") or hasattr(self.fitted_model, "forecast"):
            if self.fitted_model != None:
                prediction = self.fitted_model.forecast(**kwargs).to_frame()
            return prediction
        else:
            raise TypeError(f"Model class {self.model.__class__} does not have method [forecast()]")

    def get_forecast(self, kwargs: dict):
        """
        Метод для обобщения предсказания на несколько классов моделей

        Args:
            kwargs (dict): Словарь с данными для предсказаяния

        Returns:
            pd.DataFrame: Датафрейм с результатами предсказания
        """
        if hasattr(self.model, "get_forecast"):
            return self.model.get_forecast(**kwargs)[0].to_frame()
        else:
            raise TypeError(f"Model class {self.model.__class__} does not have method [get_forecast()]")

    def predict(self, kwargs: dict) -> pd.DataFrame:
        """
        Метод для обобщения предсказания

        Args:
            kwargs (dict): Словарь с данными для предсказаяния

        Returns:
            pd.DataFrame: Датафрейм с результатами предсказания
        """
        if self.fitted_model != None:
            prediction = self.fitted_model.predict(**kwargs)
        else:
            prediction = self.model.predict(**kwargs)

        # pd.Series - возвращается классами из statsmodels
        if isinstance(prediction, pd.Series):
            prediction = prediction.to_frame(name='prediction')
        # np.ndarray - возвращается классами из sklearn и catboost
        elif isinstance(prediction, np.ndarray):
            cols = [f'prediction_{i}' if i >= 1 else 'prediction' for i in range(1 if prediction.ndim == 1 else prediction.shape[1])]
            prediction = pd.DataFrame(prediction, columns=cols, index=None)
        # pd.DataFrame - возвращается Prophet
        elif isinstance(prediction, tuple):
            prediction = prediction[0].to_frame(name='prediction')
        else:
            prediction = prediction.rename(columns={'yhat':'prediction'})
        return prediction

         
