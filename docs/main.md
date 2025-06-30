### Главный модуль предоставляет упрощённый интерфейс взаимодействия с валидацей временных рядов

___

### Примеры использования интерфейса для 6 разных шаблонов приведены в следующих файлах:

#### [1. Elastic Net](../examples/Example_ElasticNet.ipynb),
#### [2. Holt Winters](../examples/Example_ExpSmoothing.ipynb),
#### [3. Polynomial Regression](../examples/Example_PolinomialRegression.ipynb)
#### [4. Catboost](../examples/Example_Catboost.ipynb),
#### [5. Random Forest](../examples/Example_RandomForest.ipynb)
#### [6. Prophet](../examples/Example_Prophet.ipynb),
#### [7. Fourie](../examples/Example_Fourie.ipynb),
#### [8. CrostonTSB](../examples/Example_Croston.ipynb),
#### [9. SARIMA](../examples/Example_SARIMA.ipynb).


___

### Импортирование модели выглядит следующим образом:

```py
from ts_val_shuffle import Validator
```

### Для использования модуля необходимо сначала передать в модель валидатора временный ряд в формате ```pd.DataFrame```, а затем сформировать 2 файла в JSON формате:
#### [1. Файл с конфигурацией генерируемых признаков](../examples/data/configs/features/demo_config.json),
#### [2. Файл с конфигурацией гиперпарметров модели, дополнительными параметрами для обучения модели и конфигурацией осуществления кросс-валидации](../examples/data/configs/params/validation_params_randomforest.json).

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

### [_> Описание алгоритмов](CustomAlgorithms.md).