import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import mock_open, patch
from src.ts_val_shuffle.ts_split import _ScrollTimeSeriesSplitLoader_, _ExpandingTimeSeriesSplitLoader_

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5, 6, 7],
        'col2': [3, 4, 5, 6, 7, 8, 9],
    })

@pytest.fixture
def sample_scroll(sample_df):
    return _ScrollTimeSeriesSplitLoader_(sample_df, 3, 2)

@pytest.fixture
def sample_expanding(sample_df):
    return _ExpandingTimeSeriesSplitLoader_(sample_df, 3, 2)

@pytest.fixture
def empty_df():
    return pd.DataFrame()

class TestClassInitializationScroll:
    def test_full_init_success(self, sample_df):
        stss = _ScrollTimeSeriesSplitLoader_(sample_df, 3, 2)
        assert isinstance(stss.data, pd.DataFrame)
        assert isinstance(stss.data_size, int)
        assert isinstance(stss.test_size, int)
        assert isinstance(stss.train_size, int)
        assert isinstance(stss.n_splits, int)
        assert isinstance(stss.fold_num, int)
        assert stss.data_size == 7
        assert stss.n_splits == 3
        assert stss.test_size == 2
        assert stss.fold_num == 0
        assert stss.train_size == 1

    def test_part_init_success(self, sample_df):
        stss = _ScrollTimeSeriesSplitLoader_(sample_df, 3)
        assert isinstance(stss.data, pd.DataFrame)
        assert isinstance(stss.data_size, int)
        assert isinstance(stss.test_size, int)
        assert isinstance(stss.train_size, int)
        assert isinstance(stss.n_splits, int)
        assert stss.data_size == 7
        assert stss.n_splits == 3
        assert stss.test_size == 1
        assert stss.fold_num == 0
        assert stss.train_size == 4

    def test_empty_df_init(self, empty_df):
        with pytest.raises(ValueError, match="DataFrame is empty. Cannot initialize ScrollTimeSeriesSplitLoader_"):
            _ScrollTimeSeriesSplitLoader_(empty_df, 3)
        

class TestClassCountTrainTestSizeScroll:
    def test_count_train_test_sizes_success(self, sample_df):
        stss = _ScrollTimeSeriesSplitLoader_(sample_df, 3, 2)
        assert stss.train_size == 1
        assert stss.test_size == 2

    def test_count_train_test_sizes_invalid_n_splits(self, empty_df):
        with pytest.raises(ValueError, match="DataFrame is empty. Cannot initialize ScrollTimeSeriesSplitLoader_"):
            _ScrollTimeSeriesSplitLoader_(empty_df, 3, 2)

    def test_count_train_test_sizes_invalid_test_size(self, sample_df):
        with pytest.raises(ValueError, match=r'\[test_size\] must be in interval \(0, 7\)'):
            _ScrollTimeSeriesSplitLoader_(sample_df, 3, 10)

    def test_count_train_test_sizes_invalid_n_splits_or_test_size(self, sample_df):
        error_pattern = (
            r"\[test_size\]=\(\d+\) is too high for \[n_splits\]=\(\d+\)\."
            r"\nIt turns \d+ split\(s\) as possible\."
            r"\nDecrease \[test_size\] or \[n_splits\]"
        )
        with pytest.raises(ValueError, match=error_pattern):
            _ScrollTimeSeriesSplitLoader_(sample_df, 5, 3)

class TestClassGetNextFoldScroll:
    def test_get_next_fold_success(self, sample_df):
        stss = _ScrollTimeSeriesSplitLoader_(sample_df, 3, 2)
        fold, _ = stss.get_current_fold()
        assert fold.shape == (1, 2)
        assert _.shape == (2, 2)
        flag = stss.next_fold()
        assert flag is True
        fold, _ = stss.get_current_fold()
        assert fold.shape == (1, 2)
        assert _.shape == (2, 2)
        flag = stss.next_fold()
        assert flag is True
        fold, _ = stss.get_current_fold()
        assert fold.shape == (1, 2)
        assert _.shape == (2, 2)
        flag = stss.next_fold()
        assert flag is False

class TestClassInitializationExpanding:
    def test_full_init_success(self, sample_df):
        stss = _ExpandingTimeSeriesSplitLoader_(sample_df, 3, 2)
        assert isinstance(stss.data, pd.DataFrame)
        assert isinstance(stss.data_size, int)
        assert isinstance(stss.test_size, int)
        assert isinstance(stss.train_size, int)
        assert isinstance(stss.n_splits, int)
        assert isinstance(stss.fold_num, int)
        assert stss.data_size == 7
        assert stss.n_splits == 3
        assert stss.test_size == 2
        assert stss.fold_num == 0
        assert stss.train_size == 1

    def test_part_init_success(self, sample_df):
        stss = _ExpandingTimeSeriesSplitLoader_(sample_df, 3)
        assert isinstance(stss.data, pd.DataFrame)
        assert isinstance(stss.data_size, int)
        assert isinstance(stss.test_size, int)
        assert isinstance(stss.train_size, int)
        assert isinstance(stss.n_splits, int)
        assert stss.data_size == 7
        assert stss.n_splits == 3
        assert stss.test_size == 1
        assert stss.fold_num == 0
        assert stss.train_size == 4

    def test_empty_df_init(self, empty_df):
        with pytest.raises(ValueError, match="DataFrame is empty. Cannot initialize ExpandingTimeSeriesSplitLoader_"):
            _ExpandingTimeSeriesSplitLoader_(empty_df, 3)

class TestClassCountTrainTestSizeExpanding:
    def test_count_train_test_sizes_success(self, sample_df):
        stss = _ExpandingTimeSeriesSplitLoader_(sample_df, 3, 2)
        assert stss.train_size == 1
        assert stss.test_size == 2

    def test_count_train_test_sizes_invalid_n_splits(self, empty_df):
        with pytest.raises(ValueError, match="DataFrame is empty. Cannot initialize ExpandingTimeSeriesSplitLoader_"):
            _ExpandingTimeSeriesSplitLoader_(empty_df, 3, 2)

    def test_count_train_test_sizes_invalid_test_size(self, sample_df):
        with pytest.raises(ValueError, match=r'\[test_size\] must be in interval \(0, 7\)'):
            _ExpandingTimeSeriesSplitLoader_(sample_df, 3, 10)

    def test_count_train_test_sizes_invalid_n_splits_or_test_size(self, sample_df):
        error_pattern = (
            r"\[test_size\]=\(\d+\) is too high for \[n_splits\]=\(\d+\)\."
            r"\nIt turns \d+ split\(s\) as possible\."
            r"\nDecrease \[test_size\] or \[n_splits\]"
        )
        with pytest.raises(ValueError, match=error_pattern):
            _ExpandingTimeSeriesSplitLoader_(sample_df, 5, 3)

class TestClassGetNextFoldScroll:
    def test_get_next_fold_success(self, sample_df):
        stss = _ExpandingTimeSeriesSplitLoader_(sample_df, 3, 2)
        fold, _ = stss.get_current_fold()
        assert fold.shape == (1, 2)
        assert _.shape == (2, 2)
        flag = stss.next_fold()
        assert flag is True
        fold, _ = stss.get_current_fold()
        assert fold.shape == (3, 2)
        assert _.shape == (2, 2)
        flag = stss.next_fold()
        assert flag is True
        fold, _ = stss.get_current_fold()
        assert fold.shape == (5, 2)
        assert _.shape == (2, 2)
        flag = stss.next_fold()
        assert flag is False