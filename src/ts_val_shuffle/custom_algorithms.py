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

from sympy import Expr

class CrostonTSB:
    """
    Метод Кростона
    """
    def __init__(self, alpha: float, beta: float, n_predict: int, time_step: str, sample: pd.DataFrame=None) -> None:
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


    def fit(self):
        pass


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

        
    def predict(self, timestamps: pd.Series, ts: pd.Series) -> tuple[pd.Series, dict]:
        """
        Прогноз - метод Кростона.

        :param timestamps: Временные метки;
        :param ts: Временной ряд;
        :return: Фрейм с предсказанием
        """
        d = np.array(ts)
        cols = len(d)
        d = np.append(d, [np.nan] * self.n_predict)

        # Уровень(a), Вероятность(p), Прогноз(f)
        a, p, f = np.full((3, cols + self.n_predict + 1), np.nan)

        # Инициализация
        first_occurrence = np.argmax(d[:cols] > 0)
        a[0] = d[first_occurrence]
        p[0] = 1 / (1 + first_occurrence)
        f[0] = p[0] * a[0]

        # Заполнение матриц уровня и вероятности, прогноз
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
        dict_seasonality, marks_res = self.__seasonal_coefficients(ts, timestamps, timestamps_res)
        df = pd.DataFrame({'y_pred': ts_res, 'indexes': marks_res})
        df.loc[cols + 1:cols + self.n_predict + 1, 'y_pred'] = df.iloc[cols + 1:cols + self.n_predict + 1].apply(
            lambda x: x.y_pred * dict_seasonality[x.indexes], axis=1)
        return df.y_pred.reset_index(drop=True), dict()  # , {'alpha': alpha, 'beta': beta}


class RollingMean:
    def __init__(self, time_step: str, n_predict: int, window_size: int,
                 weights_coeffs: list, weights_type: str, sample: pd.DataFrame=None):
        """
        :param n_predict: Количество периодов для прогнозирования
        :param window_size: Размер окна
        :param weights_coeffs: Весовые коэффциенты
        :param weights_type: Тип весов
        :param sample: Датафрейм с признаками (не используется);
        """
        self.time_step = time_step
        self.n_predict = n_predict
        self.window_size = window_size
        self.weights_coeffs = weights_coeffs
        self.weights_type = weights_type
        self.sample = sample

    def fit(self):
        pass

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


    def predict(self, timestamps: pd.Series, ts: pd.Series) -> pd.Series:
        """
        Создает скользящее среднее, формирует список компонентов

        :param timestamps: Временные метки
        :param ts: Временной ряд
        :return: Прогноз
        """
        weights_coeffs = self.__calculate_weights_coeffs(self.window_size, self.weights_type, self.weights_coeffs)
        ts_res = ts.copy()
        ts_base = None
        for i in range(self.n_predict):
            ts_res[len(ts_res.index)] = np.nan
            rol = ts_res.fillna(0).rolling(self.window_size)
            if i == 0:
                ts_base = rol.apply(lambda x: self.__weighted_mean(x, weights_coeffs)).shift(1)[:-1]
                ts_base[:self.window_size] = ts[:self.window_size].values
            ts_res = ts_res.where(pd.notna, other=rol.apply(lambda x: self.__weighted_mean(x, weights_coeffs)).shift(1))
        ts_res.loc[ts_base.index] = ts_base.values
        return ts_res



class ConstPrediction:
    def __init__(self, n_predict: int, type: str):
        self.n_predict = n_predict
        self.type = type

    def predict(self, ts: pd.Series) -> tuple[pd.Series, float]:
        """
        Прогноз - константа

        :param ts: Временной ряд;
        :param n_predict: Количество предсказаний;
        :param type: Тип константы;
        :return: Фрейм с предсказанием
        """
        n = len(ts.index)
        value = None
        match (self.type):
            case 'Moda':
                value = ts.mode()[0]
            case 'Mean':
                value = ts.mean()
            case 'Min':
                value = ts.min()
            case 'Max':
                value = ts.max()
            case 'Median':
                value = ts.median()

        ts_res = pd.Series(value, index=range(n + self.n_predict), dtype='float64')
        return ts_res, value

    def fit(self):
        pass


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

    def __fourier_series(self, x, f) -> Expr:
        a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, self.order + 1)]))
        sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, self.order + 1)]))
        series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                          for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
        return series
    
    def fit(self, X_data: np.ndarray | pd.Series, y_data: np.ndarray | pd.Series) -> None:
        x, y = variables('x, y')
        w, = parameters('w')

        X_data = np.asarray(X_data.values if isinstance(X_data, pd.Series) else X_data, dtype=float)
        y_data = np.asarray(y_data.values if isinstance(y_data, pd.Series) else y_data, dtype=float)

        model_dict = {y: self.__fourier_series(x, f=w)}
        fit = Fit(model_dict, x=X_data, y=y_data, **self.params)
        self.fitted_model = fit.execute()
        self.params = self.fitted_model.params
    
    def predict(self, X: np.ndarray | pd.Series) -> pd.Series:
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted yet")
        f = self.params['w']
        a0 = self.params['a0']
        y_pred = a0
        for i in range(1, self.order + 1):
            ai = self.params.get(f'a{i}', 0)
            bi = self.params.get(f'b{i}', 0)
            y_pred += ai * np.cos(i * f * X) + bi * np.sin(i * f * X)
        return pd.Series(y_pred, index=X.index if isinstance(X, pd.Series) else None)