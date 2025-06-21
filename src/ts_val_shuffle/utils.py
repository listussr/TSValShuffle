def validate_json_keys(required_keys: dict, keys: dict):
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