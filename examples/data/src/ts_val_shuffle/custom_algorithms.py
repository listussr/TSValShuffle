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

class _CrostonTSB:
    """
    Класс для прогноза методом Кростона TSB с интерфейсом, похожим на sklearn.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1, n_predict: int = 1, time_step: str = 'D'):
        """
        Инициализация параметров модели.
        :param alpha: Параметр сглаживания для уровня (среднего спроса).
        :param beta: Параметр сглаживания для вероятности ненулевого спроса.
        :param n_predict: Количество точек прогноза.
        :param time_step: Шаг временного ряда для прогноза (например, 'D', 'W', 'M').
        """
        self.alpha = alpha
        self.beta = beta
        self.n_predict = n_predict
        self.time_step = time_step
        self.fitted_ = False
        self.seasonality_ = None
        self.timestamps_res_ = None
        self.last_level_ = None
        self.last_prob_ = None
        self.last_forecast_ = None

    def fit(self, timestamps: pd.Series, ts: pd.Series):
        """
        Обучение модели методом Кростона TSB.
        :param timestamps: Временные метки (pd.Series с типом datetime).
        :param ts: Временной ряд (pd.Series с числовыми значениями).
        :param sample: Дополнительные признаки (не используются, для совместимости).
        :return: self
        """
        d = np.array(ts)
        cols = len(d)
        d = np.append(d, [np.nan] * self.n_predict)

        a, p, f = np.full((3, cols + self.n_predict + 1), np.nan)

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

        a[cols + 1:cols + self.n_predict + 1] = a[cols]
        p[cols + 1:cols + self.n_predict + 1] = p[cols]
        f[cols + 1:cols + self.n_predict + 1] = f[cols]

        ts_res = pd.Series(index=range(cols + self.n_predict), dtype='float64')
        ts_res.loc[ts_res.index] = f[1:]

        timestamps_res = pd.date_range(start=timestamps.iloc[0], freq=self.time_step, periods=len(ts_res))
        dict_seasonality, marks_res = self._seasonal_coefficients(ts, timestamps, self.time_step, timestamps_res)

        df = pd.DataFrame({'y_pred': ts_res, 'indexes': marks_res})
        df.loc[cols + 1:cols + self.n_predict + 1, 'y_pred'] = df.loc[cols + 1:cols + self.n_predict + 1].apply(
            lambda x: x.y_pred * dict_seasonality.get(x.indexes, 1), axis=1)

        self.fitted_ = True
        self.seasonality_ = dict_seasonality
        self.timestamps_res_ = timestamps_res
        self.last_level_ = a[cols]
        self.last_prob_ = p[cols]
        self.last_forecast_ = f[cols]

        self.prediction_ = df.y_pred.reset_index(drop=True)
        return self

    def predict(self, n_predict: int = None):
        """
        Прогнозирование на n_predict точек вперед.
        :param n_predict: Количество точек прогноза. Если None, используется n_predict из fit.
        :return: pd.Series с прогнозом
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        if n_predict is None:
            n_predict = self.n_predict

        forecast = np.full(n_predict, np.nan)
        for i in range(n_predict):
            base_forecast = self.last_prob_ * self.last_level_
            idx = self.timestamps_res_[len(self.timestamps_res_) - n_predict + i]
            season_idx = self._get_season_index(idx)
            season_coeff = self.seasonality_.get(season_idx, 1)
            forecast[i] = base_forecast * season_coeff

        return pd.Series(forecast)

    def _seasonal_coefficients(self, ts: pd.Series, timestamps: pd.Series, time_step: str, timestamps_res: pd.DatetimeIndex):
        """
        Вычисление сезонных коэффициентов на основе ряда.
        """
        mean_sales = np.mean(ts.values)
        marks_res = None
        marks = None
        match time_step:
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
            case _:
                marks = pd.Series([0]*len(timestamps))
                marks_res = pd.Series([0]*len(timestamps_res))

        dict_seasonality = {}
        df = pd.DataFrame({'ds': marks, 'y': ts})
        for timestamp in marks.unique():
            values = df[df.ds == timestamp].y
            dict_seasonality[timestamp] = np.mean(values / mean_sales) if mean_sales != 0 else 1
        for timestamp in set(marks_res.unique()) - set(marks.unique()):
            dict_seasonality[timestamp] = dict_seasonality.get(timestamp, 1)
        return dict_seasonality, marks_res

    def _get_season_index(self, timestamp: pd.Timestamp):
        """
        Получить индекс сезонности для заданной временной метки.
        """
        match self.time_step:
            case 'YS':
                return timestamp.year
            case 'QS':
                return timestamp.quarter
            case 'MS':
                return timestamp.month
            case 'W' | 'W-MON':
                return (timestamp.day - 1 + (timestamp - pd.to_timedelta(timestamp.day - 1, unit='d')).dayofweek) // 7 + 1
            case 'D':
                return timestamp.dayofweek
            case _:
                return 0


class _RollingMean:
    def __init__(self, window_size: int,
                 weights_coeffs: list, weights_type: str, time_step: str = None, sample: pd.DataFrame=None):
        """
        
        :param window_size: Размер окна
        :param weights_coeffs: Весовые коэффциенты
        :param weights_type: Тип весов
        :param time_step: Шаг временного ряда (Не используется);
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


class _ConstPrediction:
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
        

class _CatBoostRegressor_(CatBoostRegressor):
    def predict(self, **kwargs):
        if 'X' not in kwargs:
            raise ValueError("Missing required argument 'X'")
        X = kwargs['X']
        return super().predict(X)
    

class _FourierModel:
    def __init__(self, **kwargs) -> None:
        self.order = kwargs.get("order", None)
        if self.order == None:
            raise ValueError("Missed argument [order]")
        if not isinstance(self.order, int):
            raise ValueError("Incorrect [order] type. [order] must be an integer")
        kwargs.pop("order")
        self.params = None
        self.fitted_model = None
        self.feature_names = None
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
        
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise ValueError("y must be numpy array or pandas Series")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data cannot be empty")
        if len(X) != len(y):
            raise ValueError("Length mismatch between X and y")
        self.feature_names = X.columns.tolist()

        X_np = X.to_numpy(dtype=float)
        feature_names = X.columns.tolist()

        y_np = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
        y_np = y_np.ravel()

        var_names = feature_names + ['y']
        vars_ = variables(', '.join(var_names))
        w, = parameters('w')
        model_dict = {vars_[-1]: self.__fourier_series(*vars_[:-1], f=w)}

        data_for_fit = {name: X_np[:, i] for i, name in enumerate(feature_names)}
        data_for_fit['y'] = y_np
        fit = Fit(model_dict, **data_for_fit)
        self.fitted_model = fit.execute()
        self.params = self.fitted_model.params
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted yet")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        if X.columns.to_list() != self.feature_names:
            raise ValueError("Features mismatch with training data")

        f = self.params['w']
        a0 = self.params.get('a0', 0)
        y_pred = np.full(shape=len(X), fill_value=a0, dtype=float)

        for i in range(1, self.order + 1):
            ai = self.params.get(f'a{i}', 0)
            bi = self.params.get(f'b{i}', 0)
            for col in X.columns:
                x_col = X[col].to_numpy(dtype=float)
                y_pred += ai * np.cos(i * f * x_col) + bi * np.sin(i * f * x_col)

        return pd.Series(y_pred, index=X.index)

    
class _SARIMAX_:
    def __init__(self, order=(1,0,0), seasonal_order=(0,0,0,0), trend=None,
                 enforce_stationarity=True, enforce_invertibility=True):
        if not (isinstance(order, tuple) and len(order) == 3 and 
                all(isinstance(x, int) for x in order)):
            raise ValueError("order must be a tuple of 3 integers")
        if any(x < 0 for x in order):
            raise ValueError("order values must be non-negative")
        if any(x < 0 for x in order):
            raise ValueError("order values must be non-negative")
        
        if not (isinstance(seasonal_order, tuple) and len(seasonal_order) == 4 and 
                all(isinstance(x, int) for x in seasonal_order)):
            raise ValueError("seasonal_order must be a tuple of 4 integers")
        if any(x < 0 for x in seasonal_order):
            raise ValueError("seasonal_order values must be non-negative")

        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._fitted = False

    def fit(self, endog, exog=None, **fit_kwargs):
        if not isinstance(endog, (np.ndarray, list, pd.Series)):
            raise ValueError("endog must be array-like (numpy array, list or pandas Series)")
        if len(endog) == 0:
            raise ValueError("endog cannot be empty")
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
        try:
            return self.results.predict(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"Invalid predict parameters: {str(e)}")
        
    def forecast(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        try:
            return self.results.forecast(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"Invalid forecast parameters: {str(e)}")

    def summary(self):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")
        return self.results.summary()


class _SimpleExpSmoothing_:
    def __init__(self, *args, **kwargs):
        # Здесь можно сохранить параметры, если нужны
        self.args = args
        self.kwargs = kwargs
        self._fitted = False
        self.model = None
        self.results = None

    def fit(self, endog, **fit_kwargs):
        if not isinstance(endog, (pd.Series, np.ndarray, list)):
            raise ValueError("endog must be array-like (pandas Series, numpy array or list)")
        if len(endog) == 0:
            raise ValueError("endog cannot be empty")
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


class _Holt_:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._fitted = False
        self.model = None
        self.results = None

    def fit(self, endog, **fit_kwargs):
        if not isinstance(endog, (pd.Series, np.ndarray, list)):
            raise ValueError("endog must be array-like (pandas Series, numpy array or list)")
        if len(endog) == 0:
            raise ValueError("endog cannot be empty")
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


class _ExponentialSmoothing_:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._fitted = False
        self.model = None
        self.results = None

    def fit(self, endog, **fit_kwargs):
        if not isinstance(endog, (pd.Series, np.ndarray, list)):
            raise ValueError("endog must be array-like (pandas Series, numpy array or list)")
        if len(endog) == 0:
            raise ValueError("endog cannot be empty")
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
