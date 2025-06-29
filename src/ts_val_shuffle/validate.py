import json
import numpy as np
import pandas as pd

from .algorithm_loader import _ModelAdapter
from .ts_split import (
    _ExpandingTimeSeriesSplitLoader,
    _ExpandingTimeSeriesSplitLoader_,
    _ScrollTimeSeriesSplitLoader,
    _ScrollTimeSeriesSplitLoader_
)
from .features_generation import FeaturesGenerator

from .utils import (
    MAPE,
    SMAPE,
    WAPE,
    _validate_json_keys,
    _validate_json_values,
)

class Validator:
    def __init__(self):
        self.adapter = _ModelAdapter()
        self.ts = None
        self.generated_ts = None
        self.__metric_dict = {
            "MAPE": MAPE,
            "SMAPE": SMAPE,
            "WAPE": WAPE,
        }
        self.__split_dict = {
            "rolling": _ScrollTimeSeriesSplitLoader_,
            "expanding": _ExpandingTimeSeriesSplitLoader_,
        }
        self.metric_values = []
        self.best_model = {"metric": None, "model": None, "fold_num": None}

    def __set_model(self, model_name: str, params: dict) -> 'Validator':
        """
        Метод инициализации используемой модели

        Args:
            model_name (str): Название модели
            params (dict): Параметры инициализации

        Returns:
            Validator:
        """
        self.__init_params = params
        self.adapter.set_model(model_name, params)
        return self

    def set_data(self, ts: pd.DataFrame) -> 'Validator':
        """
        Метод инициализации временного ряда

        Args:
            ts (pd.DataFrame): Временный ряд

        Returns:
            Validator:
        """
        self.ts = ts
        return self

    def set_generator(self, json_path: str) -> 'Validator':
        """
        Метод инициализации генератора признаков и последующей генерации

        Args:
            json_path (str): Путь к JSON файлу с конфигурацией

        Raises:
            RuntimeError: Ошибка отсутствия исходного временного ряда

        Returns:
            Validator:
        """
        if self.ts is None or self.ts.empty:
            raise RuntimeError(f"Field [ts] is None. Call [set_data()] beafore calling [set_generator()]")
        self.__generator = FeaturesGenerator(json_path)
        self.generated_ts = self.__generator.generate_features(self.ts)
        return self
    
    def __load_config(self, json_path: str) -> dict:
        """
        Загрузка конфигурации признаков из JSON
        Args:
            json_path (str): Путь к JSON файлу с конфигурацией
        """
        with open(json_path, 'r') as file:
            config = json.load(file)

        return config

    def __validate_init_params(self) -> None:
        """
        Метод валидации параметров инициализации модели

        Args:
            init_params (dict): Параметры инициализации модели

        Raises:
            KeyError: Ошибка отсутствия обязательных параметров
            TypeError: Ошибка несоответствия типов параметров
        """
        _validate_json_keys({"model_name": str, "params": dict}, self.__init_params)
        _validate_json_values("model_name", self.__init_params, str)
        _validate_json_values("params", self.__init_params, dict)

    def __validate_split_params(self) -> None:
        """
        Метод валидации параметров схемы кросс-валидации

        Raises:
            KeyError: Ошибка отсутствия обязательных параметров
            TypeError: Ошибка несоответствия типов параметров
        """
        _validate_json_keys({"method": str, "n_splits": int, "test_size": int}, self.__split_params)
        _validate_json_values("method", self.__split_params, str)
        _validate_json_values("n_splits", self.__split_params, int)
        if self.__split_params.get("test_size", None) != None:
            _validate_json_values("test_size", self.__split_params, int)

    def __validate_validation_params(self) -> None:
        """
        Метод валидации параметров валидации модели

        Raises:
            KeyError: Ошибка отсутствия обязательных параметров
            TypeError: Ошибка несоответствия типов параметров
        """
        valid_keys = {
            "metric": str, 
            "target_feature": str, 
            "time_feature": str, 
            "shuffling": bool, 
            "extra_fit_params": dict,
            "add_feature_params": dict
        }
        _validate_json_keys(valid_keys, self.__validation_params)
        _validate_json_values("metric", self.__validation_params, str)
        _validate_json_values("target_feature", self.__validation_params, str)
        _validate_json_values("time_feature", self.__validation_params, str)
        if self.__validation_params.get("shuffling", None) != None:
            _validate_json_values("shuffling", self.__validation_params, bool)
        else:
            # если shuffling не указан, то по умолчанию False
            self.__validation_params["shuffling"] = False
        # по умолчанию пустой словарь
        if self.__validation_params.get("extra_fit_params", None) != None:
            _validate_json_values("extra_fit_params", self.__validation_params, dict)
        else:
            # если extra_fit_params не указан, то по умолчанию пустой словарь
            self.__validation_params["extra_fit_params"] = {}
        # персонально для Prophet
        if self.__validation_params.get("add_feature_params", None) != None:
            _validate_json_values("add_feature_params", self.__validation_params, dict)
        else:
            # если add_feature_params не указан, то по умолчанию пустой словарь
            self.__validation_params["add_feature_params"] = {}

    def load_params(self, json_path: str) -> 'Validator':
        """
        Метод загрузки параметров модели из JSON файла

        Args:
            json_path (str): Путь к JSON файлу с параметрами

        Returns:
            Validator:
        """
        self.params = self.__load_config(json_path)
        self.__init_params = self.params.get("init_params", {})
        self.__validate_init_params()
        self.__split_params = self.params.get("split_params", {})
        self.__validate_split_params()
        self.__validation_params = self.params.get("validate_params", {})
        self.__validate_validation_params()
        return self

    def __set_split_method(self, method: str, n_splits: int, test_size: int=None) -> 'Validator':
        """
        Метод установки схемы кросс-валидации

        Args:
            method (str): Название метода
            n_splits (int): Количество фолдов
            test_size (int, optional): Размер тестовой выборки. Defaults to None.

        Returns:
            Validator:
        """
        self.split_handler = self.__split_dict[method](self.generated_ts, n_splits, test_size)
        return self
        
    def __form_fit_data(self, train: pd.DataFrame, target_feature: str, time_feature: str, shuffling: bool=False) -> dict:
        """
        Метод формирования словаря с данными для обучения модели-адаптера

        Args:
            train (pd.DataFrame): Не специфицированная под модель обучающая выборка
            target_feature (str): Название целевого признака
            time_feature (str): Название исходного временного признака
            add_feature_params (dict, optional): Доплнительные параметры для добавления признака для Prophet. Defaults to {}.

        Returns:
            dict: Словарь с данными для обучения модели-адаптера
        """
        fit_config = self.adapter.adapter_config["fit_type"]
        if shuffling:
            print("shuffling")
            train = train.sample(frac=1).reset_index(drop=True)
        if fit_config == 'sklearn':
            # вариант для sklearn и catboost
            if time_feature in train.columns.to_list():
                train = train.drop(columns=[time_feature])
            X = train.drop(columns=[target_feature])
            y = train[target_feature]
            return {"X": X, 'y': y}
        elif fit_config == 'statsmodels':
            # базовый временный ряд для моделей statsmodels: Holt, SimpleExpSmoothing, ExpSmoothing
            # ставим индесовым столбцом исходный временной признак
            train = train.set_index(time_feature)
            endog = train[target_feature]
            return {'endog': endog}
        elif fit_config == 'sarimax':
            # dataframe для дополнительных признаков кроме исходного ряда
            train = train.set_index(time_feature)
            #train = train.drop(columns=[time_feature])
            endog = train[target_feature]
            exog = train.drop(columns=[target_feature])
            return {'endog': endog, 'exog': exog}            
        elif fit_config == 'prophet':
            # для Prophet
            train = train.rename(columns={time_feature: 'ds', target_feature: 'y'})
            return {"df": train}
        elif fit_config == 'croston':
            X = train[time_feature]
            y = train[target_feature]
            return {"timestamps": X, 'ts': y}
        else:
            return {"X": train[target_feature]}

    def __test_model(self, test: pd.DataFrame, target_feature: str, time_feature: str) -> pd.Series:
        """
        Метод для тестирования модели-адаптера на тестовой выборке

        Args:
            train (pd.DataFrame): Не специфицированная под модель тестовая выборка
            target_feature (str): Название целевого признака
            time_feature (str): Название исходного временного признака

        Returns:
            pd.Series: Предсказания модели на тестовой выборке
        """
        fit_config = self.adapter.adapter_config["fit_type"]
        if fit_config == 'sklearn' or fit_config == 'sklearn_like':
            # вариант для sklearn и catboost
            test = test.drop(columns=[target_feature, time_feature])
            predict_params = {'X': test}
            result = self.adapter.predict(predict_params)['prediction']
        elif fit_config == 'statsmodels':
            # базовый временный ряд для моделей statsmodels: Holt, SimpleExpSmoothing, ExpSmoothing
            test = test.set_index(time_feature)
            predict_params = {
                'start': test.index[0],
                'end': test.index[-1]
            }
            result = self.adapter.predict(predict_params)['prediction']
        elif fit_config == 'sarimax':
            # dataframe для дополнительных признаков кроме исходного ряда
            test = test.set_index(time_feature)
            exog = test.drop(columns=[target_feature])
            predict_params = {
                'start': test.index[0],
                'end': test.index[-1],
                'exog': exog,
            }
            result = self.adapter.predict(predict_params)
            print("exog.shape: ", exog.shape)
            print(exog.head())
            print(f"start: {predict_params['start']}, end: {predict_params['end']}")
            print("result.shape: ", result.shape)
            print(result.head())
            result = result['prediction']
        elif fit_config == 'prophet':
            # для Prophet
            test = test.rename(columns={time_feature: 'ds', target_feature: 'y'})
            features = test.drop(columns=['y', 'ds']).columns.to_list()
            freq = pd.infer_freq(test['ds'])
            mfd_params = {
                "periods": len(test),
                "freq": freq,
            }
            future = self.adapter.make_future_dataframe(mfd_params)
            future_only = future[future['ds'] >= test['ds'].min()]
            for feat in features:
                future_only[feat] = test[feat].values
            predict_params = {
                'df': future_only
            }
            result = self.adapter.predict(predict_params)
            result = result['prediction']
        elif fit_config == 'croston':
            predict_params = {
                'n_predict': len(test[time_feature]),
            }
            result = self.adapter.predict(predict_params)['prediction']
        return result

    def __add_regressor(self, train: pd.DataFrame, target_feature: str, time_feature: str, add_feature_params: dict={}) -> None:
        """
        Добавление регрессоров в модель Prophet

        Args:
            train (pd.DataFrame): обучающая выборка
            target_feature (str): название целевого признака
            time_feature (str): название временного признака
            add_feature_params (dict, optional): дополнительные параметры для добавления регрессоров. Defaults to {}.
        """
        train = train.rename(columns={time_feature: 'ds', target_feature: 'y'})
        features = train.drop(columns=['ds', 'y']).columns.to_list()
        for feature in features:
            extra_params = add_feature_params.get(feature, {})
            self.adapter.add_regressor(feature, extra_params)

    def __validate(self, metric: str, target_feature: str, time_feature: str, extra_fit_params: dict={}, 
                 shuffling: bool=False, add_feature_params: dict={}):
        """
        Метод валидации выбранной модели на временном ряду

        Args:
            metric (str): Название метрики
            target_feature (str): Название целевого признака
            time_feature (str): Название исходного временного признака
            extra_fit_params (dict, optional): Дополнительные параметры для метода fit(). Defaults to {}.
            shuffling (bool, optional): Флаг перемешивания данных в ходе валидации. Defaults to False.
            add_feature_params (dict, optional): Доплнительные параметры для добавления признака для Prophet. Defaults to {}.

        Raises:
            RuntimeError: Ошибка отсутствия сгенерированного временного ряда
            RuntimeError: Ошибка отсутствия заданной модели
            RuntimeError: Ошибка отсутствия метода для валидации
            KeyError: Ошибка неправильного указания метрики
        """
        if self.generated_ts.empty or self.generated_ts is None:
            raise RuntimeError(f"Field [generated_ts] is None. Call [set_generator()] beafore calling [validate()]")
        if self.adapter.adapter_config == None:
            raise RuntimeError(f"Field [adapter] is None. Call [set_model()] beafore calling [validate()]")
        if self.split_handler == None:
            raise RuntimeError(f"Field [split_handler] is None. Call [set_split_method()] beafore calling [validate()]")

        # если индексовый столбец с датами, то делаем его простым столбцом       
        if time_feature not in self.generated_ts.columns.to_list():
            self.generated_ts = self.generated_ts.reset_index(drop=False)
            self.generated_ts = self.generated_ts.rename(columns={self.generated_ts.columns[0], time_feature})

        self.metric = self.__metric_dict[metric]
        if self.metric == None:
            raise KeyError(f"Incorrect metric. Required: {self.__metric_dict}")
        learning_flag = True
        # костыль для prophet
        if self.adapter.adapter_config['fit_type'] == 'prophet':
            train, _ = self.split_handler.get_current_fold()
            self.__add_regressor(train, target_feature, time_feature, add_feature_params)
        # кросс-валидация
        while learning_flag:
            # костыль для prophet
            if self.adapter.adapter_config['fit_type'] == 'prophet':
                self.set_model("Prophet", self.__init_params)
            train, test = self.split_handler.get_current_fold()
            fit_data = self.__form_fit_data(train, target_feature, time_feature, shuffling and self.adapter.adapter_config['shuffle'])
            fit_params = {**fit_data, **extra_fit_params}
            self.adapter.fit(fit_params)

            prediction = self.__test_model(test, target_feature, time_feature)

            self.metric_values.append(self.metric(prediction, test[target_feature]))
            if self.best_model['metric'] is None or self.best_model['metric'] > self.metric_values[-1]:
                self.best_model['metric'] = self.metric_values[-1]
                self.best_model['model'] = self.adapter.model if self.adapter.fitted_model != None else self.adapter.fitted_model
                self.best_model['fold_num'] = self.split_handler.fold_num
            learning_flag = self.split_handler.next_fold()

    def validate(self):
        self.__set_model(**self.__init_params)
        self.__set_split_method(**self.__split_params)
        self.__validate(**self.__validation_params)

    def predict(self, kwargs):
        """Метод предсказания

        Args:
            kwargs (_type_): см. документацию для разных моделей

        Returns:
            pd.Series: series с предсказанием
        """
        return self.adapter.predict(kwargs)
    
    def forecast(self, kwargs):
        """Метод предсказания на какое-то количество шагов для statsmodels

        Args:
            kwargs (_type_): см. документацию statsmodels

        Returns:
            pd.Series: series с предсказанием
        """
        return self.adapter.forecast(kwargs)
    
    def plot(self, kwargs):
        """Метод отрисовки для Prophet

        Args:
            kwargs (_type_): Парметры для отрисовки (см. документацию Prophet)

        Returns:
            _type_: см. документацию Prophet
        """
        return self.adapter.plot(kwargs)
    
    def plot_components(self, kwargs):
        """Метод отрисовки компонент для Prophet

        Args:
            kwargs (_type_): Парметры для отрисовки (см. документацию Prophet)

        Returns:
            _type_: см. документацию Prophet
        """
        return self.adapter.plot_components(kwargs)
    
    def get_best_model(self) -> dict:
        """
        Метод получения лучшей модели по метрике

        Returns:
            dict: Словарь с лучшей моделью, метрикой и номером фолда
        """
        return self.best_model