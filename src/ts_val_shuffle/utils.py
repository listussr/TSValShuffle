def validate_json_keys(required_keys: dict, keys: dict) -> None:
    """
    Функция проверки соответствия ключей JSON файла шаблону

    Args:
        required_keys (dict): Шаблонные ключи
        keys (dict): Ключи, проверяемые на соответствие

    Raises:
        KeyError: Ошибка несоответствия шаблону
    """
    for key in keys:
        if key not in required_keys:
            raise KeyError(f"Unsupported tag [{key}]. Required: {required_keys}")
        

def validate_json_values(key: str, dictionary, checking_type=None) -> None:
    """
    Функция проверки существования элемента словаря и на соответствие шаблонному типу

    Args:
        key (str): Ключ в словаре
        dictionary (_type_): Словарь, в котором проверяется ключ и значение
        checking_type (_type_, optional): Опционально. Проверка типа переменной из словаря

    Raises:
        KeyError: Ошибка отстутствия обязательного ключа
        TypeError: Ошибка несоответствия типа
    """
    val = dictionary.get(key, None)
    if val == None:
        raise KeyError(f"Missing key [{key}]")
    if checking_type != None:
        if type(val) != checking_type:
            raise TypeError(f"Incorrect [{key}] type. Reuqired type ({checking_type}).")