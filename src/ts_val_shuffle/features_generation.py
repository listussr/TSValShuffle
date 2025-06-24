import numpy as np
import pandas as pd
import json
from utils import validate_json_keys, validate_json_values

class FeaturesGenerator:
    """
    Класс для генерации признаков для временных рядов
    """
    def __init__(self, config_path: str):
        """
        Инициализация класса генератора
        Args:
            config_path (str): путь к файлу конфигурации признаков в формате JSON
        """
        self.path = config_path

        # статистики
        self.agg_functions = {
            'mean': np.mean,
            'sum': np.sum,
            'max': np.max,
            'min': np.min,
            'std': np.std,
            'var': np.var
        }

        # арифметические операции для сдвиговых признаков
        self.operations = {
            'diff': np.subtract,
            'sum': np.add,
            'mul': np.multiply,
            'div': np.divide,
        }

        # единицы времени
        self.time_units = {
            'second': lambda x: x.dt.second,
            'minute': lambda x: x.dt.minute,
            'hour': lambda x: x.dt.hour,
            'day': lambda x: x.dt.day,
            'month': lambda x: x.dt.month,
            'year': lambda x: x.dt.year,
            'dayofweek': lambda x: x.dt.dayofweek,
            'quarter': lambda x: x.dt.quarter,
        }

        # единицы времени для подсчёта относительного времени
        self.relative_time_units = {
            'total_second': lambda x1, x2: (x1 - x2).dt.total_seconds(),
            'hour': lambda x1, x2: (x1 - x2).dt.total_seconds() / 3600,
            'minute': lambda x1, x2: (x1 - x2).dt.total_seconds() / 60,
            'day': lambda x1, x2: (x1 - x2).dt.total_seconds() / 3600 / 24,
            'month': lambda x1, x2: (x1.dt.year - x2.year) * 12 + (x1.dt.month - x2.month),
            'year': lambda x1, x2: x1.dt.year - x2.year - ((x1.dt.month < x2.month) | ((x1.dt.month == x2.month) & (x1.dt.day < x2.day)))
        }

        self.cycle_function = {
            "sin": np.sin,
            "cos": np.cos,
        }

        self.config = self.__load_config()
        self.__check_config()
        self.__validate_option()


    def __load_config(self) -> dict:
        """
        Загрузка конфигурации признаков из JSON
        """
        with open(self.path, 'r') as file:
            config = json.load(file)

        return config


    def __check_config(self) -> None:
        """
        Проверка на соответствие JSON файла с конфигурацией шаблону

        При несоответствии ключей 2 верхних уровней выбрасывается KeyError
        """
        required_basic_keys = {
            "features",
            "options"
        }
        required_features_keys = {
            "lags",
            "shift",
            "rolling",
            "absolute_time",
            "relative_time"
        }
        validate_json_keys(required_basic_keys, self.config.keys())
        validate_json_keys(required_features_keys, self.config["features"].keys())


    def __generate_lag_features(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Метод генерации лаговых признаков

        Args:
            time_series (pd.DataFrame): Исходный временный ряд

        Returns:
            pd.DataFrame: Датафрейм со столбцами: [index_col, lag_feature_1, ... lag_feature_m]
        """
        df = time_series.copy()

        lag_features_config = self.config["features"].get("lags", None)

        if lag_features_config == None: # Если не указаны лаговые признаки возвращаем индексовый столбец
            return pd.DataFrame()
        else:
            features_to_keep = [] # Столбцы только лаговых признаков + столбец индексов

            for feature in lag_features_config:
                validate_json_values("name", feature, str)
                validate_json_values("source", feature, str)
                validate_json_values("window", feature, int)
                df[feature["name"]] = df[feature["source"]].shift(feature["window"]) # Создаём новые признаки
                features_to_keep.append(feature["name"])
            
            return df[features_to_keep]
        

    def __generate_shift_features(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Метод генерации различных сдвиговых признаков

        Args:
            time_series (pd.DataFrame): Исходный временный ряд

        Returns:
            pd.DataFrame: Датафрейм со столбцами: [index_col, shift_feature_1, ... shift_feature_m]
        """
        df = time_series.copy()

        shift_features_config = self.config["features"].get("shift", None)

        if shift_features_config == None: # Если не указаны сдвиговые признаки возвращаем индексовый столбец
            return pd.DataFrame()
        else:
            features_to_keep = [] # Столбцы только сдвиговых признаков + столбец индексов

            for feature in shift_features_config:
                validate_json_values("name", feature, str)
                validate_json_values("source", feature, str)
                validate_json_values("window", feature, int)
                source_col = df[feature["source"]]
                shifted_col = df[feature["source"]].shift(feature["window"])

                operation = self.operations[feature["operation"]] # устанавливаем выбранную арифметическую операцию

                df[feature["name"]] = operation(source_col, shifted_col)  # Создаём новые признаки

                features_to_keep.append(feature["name"])

            return df[features_to_keep]


    def __generate_rolling(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Метод генерации скользящих статистик

        Args:
            time_series (pd.DataFrame): Исходный временный ряд

        Returns:
            pd.DataFrame: Датафрейм со столбцами: [index_col, mean_3_1, ... std_2_m]
        """
        df = time_series.copy()

        rolling_config = self.config["features"].get("rolling", None)

        if rolling_config == None: # Если не указаны скользящие статичтики возвращаем индексовый столбец
            return pd.DataFrame()
        else:
            features_to_keep = [] # Столбцы только скользящих статистик + столбец индексов

            for feature in rolling_config:
                validate_json_values("name", feature, str)
                validate_json_values("source", feature, str)
                validate_json_values("window", feature, int)
                validate_json_values("agg", feature, str)
                agg_function = self.agg_functions[feature["agg"]] # устанавливаем выбранную статистику

                df[feature["name"]] = df[feature["source"]].rolling(window=feature["window"]).apply(agg_function, raw=True)  # Создаём новые признаки
                
                lag_window = feature.get("lag_window", None)

                if lag_window != None:
                    validate_json_values("lag_window", feature, int)
                    df[feature["name"]] = df[feature["name"]].shift(feature["lag_window"])

                features_to_keep.append(feature["name"])

            return df[features_to_keep]


    def __generate_calendar(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Метод генерации абсолютных временных признаков

        Args:
            time_series (pd.DataFrame): Исходный временный ряд

        Returns:
            pd.DataFrame: Датафрейм со столбцами: [index_col, absolute_time_1, ... absolute_time_m]
        """
        df = time_series.copy()

        abs_time_config = self.config["features"].get("absolute_time", None)

        if abs_time_config == None:
            return pd.DataFrame()
        else:
            features_to_keep = []

            # чтобы использовать все столбцы
            index = df.index.name
            if index != None:
                df = df.reset_index()

            for feature in abs_time_config:
                validate_json_values("name", feature, str)
                validate_json_values("source", feature, str)
                validate_json_values("time_unit", feature, str)
                # значение периода времени
                fetch_time_unit = self.time_units[feature["time_unit"]]

                # столбец, из которого получаем время
                source = feature.get("source", None)
                # новый временный признак
                df[feature["name"]] = fetch_time_unit(df[source])
                
                # смотрим на то, циклическое время или нет
                cycle = feature.get("cycle", None)
                cycle_function = feature.get("function", None)

                if cycle != None:
                    validate_json_values("function", feature, str)
                    validate_json_values("cycle", feature, int)
                    cycle_function = self.cycle_function[cycle_function]

                    df[feature["name"]] = cycle_function(2 * np.pi * df[feature["name"]] / cycle)

                
                features_to_keep.append(feature["name"])
            # возвращаем индекс для датафрейма
            if index != None:
                df.set_index([index], inplace=True)

            return df[features_to_keep]
        

    def __generate_relative_time(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Метод генерации 'relative time' признаков

        Args:
            time_series (pd.DataFrame): Исходный временный ряд

        Returns:
            pd.DataFrame: Датафрейм со столбцами: [index_col, relative_time_1, ... relative_time_m]
        """
        df = time_series.copy()

        relative_time_config = self.config["features"].get("relative_time", None)

        if relative_time_config == None:
            return pd.DataFrame()
        else:
            features_to_keep = []
            # чтобы использовать все столбцы
            index = df.index.name
            if index != None:
                df = df.reset_index()
            for feature in relative_time_config:
                validate_json_values("name", feature, str)
                validate_json_values("source", feature, str)
                validate_json_values("time_unit", feature, str)
                # значение периода времени
                unit = feature.get("time_unit", None)
                fetch_time_unit = self.relative_time_units[unit]
                
                # столбец, из которого получаем время
                source = feature.get("source", None)

                begin_time = df[source].min()

                # новый временный признак
                df[feature["name"]] = fetch_time_unit(df[source], begin_time)
                
                # смотрим на то, циклическое время или нет
                time_range = feature.get("range", None)

                if time_range != None:
                    min_diff = df[feature["name"]].min()
                    max_diff = df[feature["name"]].max()
                    min_val, max_val = time_range
                    df[feature["name"]] = min_val + (max_val - min_val) * (df[feature["name"]] - min_diff) / (max_diff - min_diff)

                features_to_keep.append(feature["name"])

            # возвращаем индекс для датафрейма
            if index != None:
                df.set_index([index], inplace=True)
            return df[features_to_keep]


    def __validate_option(self) -> None:
        """
        Метод для проверки флага отброса NaN элементов после генерации признаков
        """
        validate_json_values("drop_na", self.config["options"], bool)
        self.dropna = self.config["options"]["drop_na"]


    def generate_features(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Метод для генерации признаков для временного ряда по заданному шаблону

        Args:
            time_series (pd.DataFrame): Исходный временный ряд, для которого генерируются признаки
        """
        lags_df = self.__generate_lag_features(time_series)
        shift_df = self.__generate_shift_features(time_series)
        rolling_df = self.__generate_rolling(time_series)
        abs_time_df = self.__generate_calendar(time_series)
        relative_time_df = self.__generate_relative_time(time_series)
        generated_features = time_series.join(lags_df, how='outer')
        generated_features = generated_features.join(shift_df, how='outer')
        generated_features = generated_features.join(rolling_df, how='outer')
        generated_features = generated_features.join(abs_time_df, how='outer')
        generated_features = generated_features.join(relative_time_df, how='outer')
        if self.dropna:
            generated_features = generated_features.dropna()
        return generated_features
