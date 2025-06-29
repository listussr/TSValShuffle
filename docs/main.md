### Главный модуль предоставляет упрощённый интерфейс взаимодействия с валидацей временных рядов

___

### Примеры использования интерфейса для 6 разных шаблонов приведены в следующих файлах:

### [1. Elastic Net](examples/Example_ElasticNet.ipynb),
### [2. Holt Winters](examples/Example_ExpSmoothing.ipynb),
### [3. Catboost](examples/Example_Catboost.ipynb),
### [4. Prophet](examples/Example_Prophet.ipynb),
### [5. Fourie](examples/Example_Fourie.ipynb),
### [6. CrostonTSB](examples/Example_Croston.ipynb).

___

### Импортирование модели выглядит следующим образом:

```py
from ts_val_shuffle import Validator
```

### Для использования модуля необходимо сначала передать в модель валидатора временный ряд в формате ```pd.DataFrame```, а затем сформировать 2 файла в JSON формате:
#### [1. Файл с конфигурацией генерируемых признаков](../examples/data/config.json),
#### [2. Файл с конфигурацией гиперпарметров модели, дополнительными параметрами для обучения модели и конфигурацией осуществления кросс-валидации](../examples/data/validation_params_randomforest.json).

### Шаблонный код использования модели выглядит следующим образом

```py
val = Validator()

val.set_data(train)
val.set_generator(r"features_generations_params.json")
val.load_params(r"model_params.json")
val.validate()
```

### Вызов методов требует соблюдения строгой очерёдности

___

### Документация по классам ```Validator``` и ```FeaturesGenerator```

### [_> Класс Validator](Validator.md).

### [_> Класс FeaturesGenerator](FeaturesGenerator.md).