import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class ScrollTimeSeriesSplitLoader:
    def __init__(self, data: pd.DataFrame, train_size: int, sliding_size_norm: float=0.4, test_size_norm: float=0.2):
        """
        Конструктор класса

        Args:
            data (pd.DataFrame): Исходный временный ряд
            train_size (int): Размер обучающей выборки
            sliding_size_norm (float, optional): Размер сдвига окна относительно размера обучающей выборки. Defaults to 0.4.
            test_size_norm (float, optional): Размер тестовой выборки относительно размера обучающей. Defaults to 0.2.
        """
        self.data = data
        self.data_size = data.shape[0]
        self.train_size = train_size
        self.__check_values(sliding_size_norm, test_size_norm)
        self.sliding_size = int(sliding_size_norm * train_size)
        self.test_size = int(test_size_norm * train_size)
        self.fold_num = 0

    def __check_values(self, sliding_size_norm: float, test_size_norm: float):
        """
        Проверка на валидность значений из конструктора

        Args:
            sliding_size_norm (float): Отношение размера сдвига к размеру обучающей выборки
            test_size_norm (float): Отношение размера тестовой выборки к размеру обучающей

        Raises:
            ValueError: Ошибка выхода за размер временного ряда
            ValueError: Ошибка выбора коэффицента смещения
            ValueError: Ошибка выбора коэффицента тестовой выборки
        """
        if self.train_size >= self.data_size:
            raise ValueError(f"[train_size] ({self.train_size}) must be less then time series length ({self.data_size})")
        if sliding_size_norm <= 0 or sliding_size_norm > 1:
            raise ValueError(f"[sliding_size_norm] must be in interval (0, 1]")
        if test_size_norm <= 0 or test_size_norm > 1:
            raise ValueError(f"[test_size_norm] must be in interval (0, 1]")

    def get_current_fold(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Функция циклического разделения выборки на фолды

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Обучающая и тестовая выборки
        """
        begin = int(self.fold_num * self.sliding_size)
        end = int(self.fold_num * self.sliding_size + self.train_size)
        traind_df = self.data.iloc[begin:end, :]
        test_df = self.data.iloc[end:min(end + self.test_size, self.data_size), :]
        return [traind_df, test_df]
    
    def next_fold(self) -> bool:
        """
        Функция для перехода на следующий фолд

        Returns:
            bool: Флаг достижения конца выборки. Если все фолды выборки использованы, возвращается False
        """
        if self.sliding_size * (self.fold_num + 1) + self.train_size < self.data_size:
            self.fold_num += 1
            return True
        else:
            return False
        
    def reset(self) -> None:
        """
        Метод для сбрасывания индекса текущего фолда
        """
        self.fold_num = 0
        

class TimeSeriesSplitLoader:

    def __init__(self, data: pd.DataFrame, train_size: int, test_size_norm: float=0.8):
        """
        Конструктор класса

        Args:
            data (pd.DataFrame): Исходный временный ряд
            train_size (int): Размер обучающей выборки
            test_size_norm (float, optional): _description_. Defaults to 0.8.
        """
        self.data = data
        self.data_size = data.shape[0]
        self.train_size = train_size
        self.__check_values(test_size_norm)
        self.test_size = int(test_size_norm * train_size)
        self.fold_num = 0

    
    def __check_values(self, test_size_norm: float):
        """
        Проверка на валидность значений из конструктора

        Args:
            test_size_norm (float): Отношение размера тестовой выборки к размеру обучающей

        Raises:
            ValueError: Ошибка выхода за размер временного ряда
            ValueError: Ошибка выбора коэффицента тестовой выборки
        """
        if self.train_size >= self.data_size:
            raise ValueError(f"[train_size] ({self.train_size}) must be less then time series length ({self.data_size})")
        if test_size_norm <= 0 or test_size_norm > 1:
            raise ValueError(f"[test_size_norm] must be in interval (0, 1]")


    def get_current_fold(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Функция накопительного разделения выборки на фолды

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Обучающая и тестовая выборки
        """
        end = int(self.test_size * self.fold_num + self.train_size)
        traind_df = self.data.iloc[0:end, :]
        test_df = self.data.iloc[end:min(end + self.test_size, self.data_size), :]
        return [traind_df, test_df]
    
    def next_fold(self) -> bool:
        """
        Функция для перехода на следующий фолд

        Returns:
            bool: Флаг достижения конца выборки. Если все фолды выборки использованы, возвращается False
        """
        if self.test_size * (self.fold_num + 1) + self.train_size < self.data_size:
            self.fold_num += 1
            return True
        else:
            return False
        
class TimeSeriesSplitLoader_:

    def __init__(self, data: pd.DataFrame, n_splits: int, test_size: int = None):
        """
        Конструктор класса

        Args:
            data (pd.DataFrame): Исходный временный ряд
            n_splits (int): Количество фолдов
        """
        self.data = data
        self.data_size = data.shape[0]
        self.n_splits = n_splits
        self.__count_train_test_sizes(test_size)
        self.fold_num = 0

    
    def __count_train_test_sizes(self, test_size: int):
        """
        Подсчёт размеров обучающей и тестовой выборок

        Args:
            test_size_norm (float): Отношение размера тестовой выборки к размеру обучающей

        Raises:
            ValueError: Ошибка выхода за размер временного ряда
            ValueError: Ошибка выбора коэффицента тестовой выборки
        """
        if test_size == None:
            self.test_size = self.data_size // (self.n_splits + 1)
            self.train_size = self.data_size - self.test_size * self.n_splits
        else:
            if test_size >= self.data_size or test_size <= 0:
                raise ValueError(f"[test_size] must be in interval (0, {self.data_size})")
            self.test_size = test_size
            max_splits = (self.data_size - 1) // self.test_size
            if max_splits < self.n_splits:
                raise ValueError(f"[test_size]=({test_size}) is too high for [n_splits]=({self.n_splits}).\nIt turns {max_splits} split(s) as possible.\nDecrease [test_size] or [n_splits]")
            self.train_size = self.data_size - self.test_size * self.n_splits

       

    def get_current_fold(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Функция накопительного разделения выборки на фолды

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Обучающая и тестовая выборки
        """
        end = int(self.test_size * self.fold_num + self.train_size)
        traind_df = self.data.iloc[0:end, :]
        test_df = self.data.iloc[end:min(end + self.test_size, self.data_size), :]
        return [traind_df, test_df]
    

    def next_fold(self) -> bool:
        """
        Функция для перехода на следующий фолд

        Returns:
            bool: Флаг достижения конца выборки. Если все фолды выборки использованы, возвращается False
        """
        if self.fold_num < self.n_splits:
            self.fold_num += 1
            return True
        else:
            return False
        
    def reset(self) -> None:
        """
        Метод для сбрасывания индекса текущего фолда
        """
        self.fold_num = 0