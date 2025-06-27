from src.ts_val_shuffle.utils import (
    MAPE,
    SMAPE,
    WAPE,
    _validate_json_keys,
    _validate_json_values,
)
import pandas as pd
import pytest

def test_MAPE():
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([110, 190, 330])
    assert round(MAPE(y_pred, y_true), 4) == 0.0833

def test_SMAPE():
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([110, 190, 330])
    assert round(SMAPE(y_pred, y_true), 4) == 0.0806

def test_WAPE():
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([110, 190, 330])
    assert round(WAPE(y_pred, y_true), 4) == 0.0833

def test_matching_keys_no_error():
    required = {"lag": int, "agg": str}
    tested = {"lag": 1, "agg": "mean"}
    _validate_json_keys(required, tested)
    
def test_extra_key_raises_error():
    required = {"lag": int}
    tested = {"lag": 1, "agg": "mean"}
    with pytest.raises(KeyError) as exc_info:
        _validate_json_keys(required, tested)
    assert "Unsupported tag [agg]" in str(exc_info.value)
    
def test_empty_inputs_no_error():
    _validate_json_keys({}, {})

def test_required_empty_but_input_not():
    with pytest.raises(KeyError):
        _validate_json_keys({}, {"key": "value"})


    
def test_validate_json_values_key_exists_no_type_check():
    test_dict = {"lag": 1, "agg": "mean"}
    _validate_json_values("lag", test_dict)
    _validate_json_values("agg", test_dict)

def test_validate_json_values_missing_key():
    test_dict = {"lag": 1}
    with pytest.raises(KeyError) as exc_info:
        _validate_json_values("agg", test_dict)
    assert str(exc_info.value) == "'Missing key [agg]'"

def test_validate_json_values_correct_type():
    test_dict = {"lag": 1, "agg": "mean", "active": True}
    _validate_json_values("lag", test_dict, int)
    _validate_json_values("agg", test_dict, str)
    _validate_json_values("active", test_dict, bool)

def test_validate_json_values_incorrect_type():
    test_dict = {"lag": "1", "agg": "mean"}
    with pytest.raises(TypeError) as exc_info:
        _validate_json_values("lag", test_dict, int)
    error_msg = str(exc_info.value)
    assert "Incorrect [lag] type. Required type" in error_msg
    assert "int" in error_msg

def test_validate_json_values_none_value():
    test_dict = {"lag": None}
    with pytest.raises(KeyError) as exc_info:
        _validate_json_values("lag", test_dict)
    assert str(exc_info.value) == "'Missing key [lag]'"

def test_validate_json_values_false_value():
    test_dict = {"active": False}
    _validate_json_values("active", test_dict)
    _validate_json_values("active", test_dict, bool)


def test_validate_json_values_none_dict():
    with pytest.raises(AttributeError):
        _validate_json_values("key", None)

def test_validate_json_values_zero_value():
    test_dict = {"count": 0}
    _validate_json_values("count", test_dict)
    _validate_json_values("count", test_dict, int)

def test_validate_json_values_empty_string():
    test_dict = {"name": ""}
    _validate_json_values("name", test_dict)
    _validate_json_values("name", test_dict, str)

def test_validate_json_values_subclass_type():
    class MyInt(int): pass
    
    test_dict = {"num": MyInt(5)}
    with pytest.raises(TypeError):
        _validate_json_values("num", test_dict, int)