import pandas as pd
import numpy as np
import logging
from catboost.core import CatBoostRegressor

from symfit import (
    parameters, 
    variables, 
    sin, 
    cos, 
    Fit,
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, 
    Holt, 
    ExponentialSmoothing
)

from sympy import Expr

class CrostonTSB:
    """
    Метод Кростона
    """
    def __init__(self, alpha: float, beta: float, time_step: str, n_predict: int=None, sample: pd.DataFrame=None) -> None:
        """
        :param alpha: Параметр сглаживания для уровня;
        :param beta: Параметр сглаживания для вероятности;
        :param n_predict: Количество предсказаний;
        :param time_step: Шаг прогноза;
        :param sample: Датафрейм с признаками (не используется);
        """
        self.alpha = alpha
        self.beta = beta
        self.n_predict = n_predict
        self.time_step = time_step
        self.sample = sample


    def fit(self, X, y=None):
        """
        X: pd.Series или массив с временными метками или индексами
        y: pd.Series или массив с временным рядом (целевая переменная)
        Если y не задан, предполагается, что X — это временной ряд.
        """
        if y is None:
            y = X
            self.timestamps_ = None
        else:
            self.timestamps_ = pd.Series(X).copy()

        self.ts_ = pd.Series(y).copy()
        self.is_fitted_ = True
        return self

    def predict(self, X=None):
        """
        X: необязательный параметр — временные метки для прогноза.
        Если не задан, прогноз строится на self.n_predict шагов.
        """
        if not self.is_fitted_:
            raise RuntimeError("You must fit the model before prediction")

        ts = self.ts_
        if X is not None:
            timestamps = pd.Series(X)
            n_predict = len(timestamps)
        else:
            timestamps = pd.date_range(start=pd.Timestamp.now(), freq=self.time_step, periods=self.n_predict)
            n_predict = self.n_predict

        return self._croston_predict(ts, timestamps, n_predict)


    def __seasonal_coefficients(self, ts: pd.Series, timestamps: pd.Series, timestamps_res: pd.DatetimeIndex) -> tuple[dict, pd.Series]:
        """
        Вычисление сезонных коэффициентов на основе ряда.
        :param ts: Временной ряд;
        :param timestamps: Метка времени;
        :param time_step: Шаг времени;
        :param timestamps_res: Временные метки, для которых нужно получить коэффициенты
        :return: Словарь с сезонными коэффициентами, временные метки
        """
        mean_sales = np.mean(ts.values)
        marks_res = None
        marks = None
        match self.time_step:
            case 'YS':
                marks = timestamps.dt.year
                marks_res = timestamps_res.year
            case 'QS':
                marks = timestamps.dt.quarter
                marks_res = timestamps_res.quarter
            case 'MS':
                marks = timestamps.dt.month
                marks_res = timestamps_res.month
            case 'W' | 'W-MON':
                marks = (timestamps.dt.day - 1 + (
                        timestamps - pd.to_timedelta(timestamps.dt.day - 1, unit='d')).dt.dayofweek) // 7 + 1
                marks_res = (timestamps_res.day - 1 + (
                        timestamps_res - pd.to_timedelta(timestamps_res.day - 1, unit='d')).dayofweek) // 7 + 1
            case 'D':
                marks = timestamps.dt.dayofweek
                marks_res = timestamps_res.dayofweek
        dict_seasonality = {}
        df = pd.DataFrame({'ds': marks, 'y': ts})
        for timestamp in marks.unique():
            values = df[df.ds == timestamp].y
            dict_seasonality[timestamp] = np.mean(values / mean_sales)
        for timestamp in set(marks_res.unique()) - set(marks.unique()):
            dict_seasonality[timestamp] = dict_seasonality.get(timestamp, 1)
        return dict_seasonality, marks_res


    def _croston_predict(self, ts: pd.Series, timestamps: pd.Series, n_predict: int) -> tuple[pd.Series, dict]:
        d = np.array(ts)
        cols = len(d)
        d = np.append(d, [np.nan] * n_predict)

        a, p, f = np.full((3, cols + n_predict + 1), np.nan)

        first_occurrence = np.argmax(d[:cols] > 0)
        a[0] = d[first_occurrence]
        p[0] = 1 / (1 + first_occurrence)
        f[0] = p[0] * a[0]

        for t in range(cols):
            if d[t] > 0:
                a[t + 1] = self.alpha * d[t] + (1 - self.alpha) * a[t]
                p[t + 1] = self.beta * 1 + (1 - self.beta) * p[t]
            else:
                a[t + 1] = a[t]
                p[t + 1] = (1 - self.beta) * p[t]
            f[t + 1] = p[t + 1] * a[t + 1]

        a[cols + 1:cols + n_predict + 1] = a[cols]
        p[cols + 1:cols + n_predict + 1] = p[cols]
        f[cols + 1:cols + n_predict + 1] = f[cols]

        ts_res = pd.Series(f[1:cols + n_predict + 1], index=pd.RangeIndex(start=0, stop=cols + n_predict))
        ts_res.index = pd.date_range(start=timestamps.iloc[0], freq=self.time_step, periods=len(ts_res))

        return ts_res.iloc[-n_predict:], {}


class RollingMean:
    def __init__(self, time_step: str, window_size: int,
                 weights_coeffs: list, weights_type: str, sample: pd.DataFrame=None):
        """
        :param window_size: Размер окна
        :param weights_coeffs: Весовые коэффциенты
        :param weights_type: Тип весов
        :param sample: Датафрейм с признаками (не используется);
        """
        self.time_step = time_step
        self.window_size = window_size
        self.weights_coeffs = weights_coeffs
        self.weights_type = weights_type
        self.sample = sample

    def __calculate_weights_coeffs(self, n: int, weights_type: str, weights_coeffs: list) -> list:
        """
        Вычисление коэффициента весов на основе заданного типа весов и количества элементов
        :param n: Количество эламентов;
        :param weights_type: Тип весов;
        :param weights_coeffs: Изначальные коэффициенты весов;
        :return: Коэффициенты весов
        """
        if weights_type == 'custom':
            return weights_coeffs
        elif weights_type == 'new':
            reverse = False
        elif weights_type == 'old':
            reverse = True
        else:
            logging.exception(f'Передан несуществующий режим "{weights_type}". '
                            f'Необходимо выбирать режимы: "new", "old", "custom". '
                            f'Для данного пересечения будет по умолчанию будет выбран метод "new".')
            reverse = False
        a1 = 0.001
        d = (2 - 2 * a1 * n) / (n * (n - 1))
        res = [a1 + d * i for i in range(0, n)]
        res.sort(reverse=reverse)
        return res


    def __weighted_mean(self, x: pd.Series, weights_coeffs: list) -> pd.Series:
        return (x * pd.Series(weights_coeffs, index=x.index)).sum() / len(x.index)


    def predict(self, X: pd.Series=None, n_predict: int=None) -> pd.Series:
        """
        Создает скользящее среднее, формирует список компонентов

        :param n_predict: Количество прогнозируемых шагов
        :param X: Временной ряд
        :return: Прогноз
        """
        weights_coeffs = self.__calculate_weights_coeffs(self.window_size, self.weights_type, self.weights_coeffs)
        if (X is None or X.empty) and n_predict == None:
            raise ValueError("X or n_predict must be not NaN")
        n_predicts = len(X) if n_predict == None else n_predict
        ts_base = None
        forecast_values = []
        for i in range(n_predicts):
            self.ts_res[len(self.ts_res.index)] = np.nan
            rol = self.ts_res.fillna(0).rolling(self.window_size)
            if i == 0:
                ts_base = rol.apply(lambda x: self.__weighted_mean(x, weights_coeffs)).shift(1)[:-1]
                ts_base[:self.window_size] = self.X[:self.window_size].values
            self.ts_res = self.ts_res.where(pd.notna, other=rol.apply(lambda x: self.__weighted_mean(x, weights_coeffs)).shift(1))
            new_val = rol.apply(lambda x: self.__weighted_mean(x, weights_coeffs)).shift(1).iloc[-1]
            forecast_values.append(new_val)
        self.ts_res.loc[ts_base.index] = ts_base.values
        return pd.Series(np.array(forecast_values))
    
    def fit(self, X: pd.Series):
        self.ts_res = X.copy()
        self.X = X.copy()


class ConstPrediction:
    def __init__(self, type: str):
        self.type = type

    def predict(self, X: pd.Series) -> tuple[pd.Series, float]:
        """
        Прогноз - константа

        :param ts: Временной ряд;
        :param n_predict: Количество предсказаний;
        :param type: Тип константы;
        :return: Фрейм с предсказанием
        """
        

        ts_res = pd.Series(self.value, index=range(len(X)), dtype='float64')
        return ts_res, self.value

    def fit(self, X: pd.Series) -> None:
        self.value = None
        match (self.type):
            case 'Moda':
                self.value = X.mode()[0]
            case 'Mean':
                self.value = X.mean()
            case 'Min':
                self.value = X.min()
            case 'Max':
                self.value = X.max()
            case 'Median':
                self.value = X.median()
        

class CatBoostRegressor_(CatBoostRegressor):
    def predict(self, **kwargs):
        if 'X' not in kwargs:
            raise ValueError("Missing required argument 'X'")
        X = kwargs['X']
        return super().predict(X)
    

class FourierModel:
    def __init__(self, **kwargs) -> None:
        self.order = kwargs.get("order", None)
        if self.order == None:
            raise ValueError("Missed argument [order]")
        if not isinstance(self.order, int):
            raise ValueError("Incorrect [order] type. [order] must be an integer")
        kwargs.pop("order")
        self.params = None
        self.fitted_model = None
        self.params = kwargs

    def __fourier_series(self, *xs, f) -> Expr:
        a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, self.order + 1)]))
        sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, self.order + 1)]))
        series = a0
        for x in xs:
            series += sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                        for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        return series
    
    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        X_np = X.to_numpy(dtype=float)
        feature_names = X.columns.tolist()

        y_np = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
        y_np = y_np.ravel()

        # Объявляем переменные для всех признаков и целевой переменной
        var_names = feature_names + ['y']
        vars_ = variables(', '.join(var_names))

        # Объявляем параметры модели (пример с одним параметром частоты)
        w, = parameters('w')

        # Формируем модель, передавая все признаки в функцию __fourier_series
        model_dict = {vars_[-1]: self.__fourier_series(*vars_[:-1], f=w)}

        # Формируем словарь входных данных для Fit
        data_for_fit = {name: X_np[:, i] for i, name in enumerate(feature_names)}
        data_for_fit['y'] = y_np

        # Создаём и запускаем подгонку модели, передавая только входные данные
        fit = Fit(model_dict, **data_for_fit)
        self.fitted_model = fit.execute()

        # Сохраняем найденные параметры модели
        self.params = self.fitted_model.params
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted yet")

        f = self.params['w']
        a0 = self.params.get('a0', 0)
        y_pred = np.full(shape=len(X), fill_value=a0, dtype=float)

        for i in range(1, self.order + 1):
            ai = self.params.get(f'a{i}', 0)
            bi = self.params.get(f'b{i}', 0)
            # Суммируем вклад по всем признакам
            for col in X.columns:
                x_col = X[col].to_numpy(dtype=float)
                y_pred += ai * np.cos(i * f * x_col) + bi * np.sin(i * f * x_col)

        return pd.Series(y_pred, index=X.index)

    
class SARIMAX_:
    def __init__(self, order=(1,0,0), seasonal_order=(0,0,0,0), trend=None,
                 enforce_stationarity=True, enforce_invertibility=True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._fitted = False

    def fit(self, endog, exog=None, **fit_kwargs):
        self.model = SARIMAX(
            endog=endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        self.results = self.model.fit(**fit_kwargs)
        self._fitted = True
        return self

    def predict(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.predict(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.forecast(*args, **kwargs)

    def summary(self):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.summary()

class SimpleExpSmoothing_:
    def __init__(self, *args, **kwargs):
        # Здесь можно сохранить параметры, если нужны
        self.args = args
        self.kwargs = kwargs
        self._fitted = False
        self.model = None
        self.results = None

    def fit(self, endog, **fit_kwargs):
        self.model = SimpleExpSmoothing(endog, *self.args, **self.kwargs)
        self.results = self.model.fit(**fit_kwargs)
        self._fitted = True
        return self

    def predict(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.predict(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.forecast(*args, **kwargs)

    def summary(self):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.summary()


class Holt_:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._fitted = False
        self.model = None
        self.results = None

    def fit(self, endog, **fit_kwargs):
        self.model = Holt(endog, *self.args, **self.kwargs)
        self.results = self.model.fit(**fit_kwargs)
        self._fitted = True
        return self

    def predict(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.predict(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.forecast(*args, **kwargs)

    def summary(self):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.summary()


class ExponentialSmoothing_:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._fitted = False
        self.model = None
        self.results = None

    def fit(self, endog, **fit_kwargs):
        self.model = ExponentialSmoothing(endog, *self.args, **self.kwargs)
        self.results = self.model.fit(**fit_kwargs)
        self._fitted = True
        return self

    def predict(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.predict(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.forecast(*args, **kwargs)

    def summary(self):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.summary()
