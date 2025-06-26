from features_generation import FeaturesGenerator
import pandas as pd
import numpy as np
from ts_split import (
    ScrollTimeSeriesSplitLoader, 
    ExpandingTimeSeriesSplitLoader_,
    ExpandingTimeSeriesSplitLoader,
    ScrollTimeSeriesSplitLoader_
)

from sklearn.linear_model import LinearRegression

from utils import MAPE, SMAPE, WAPE

from algorithm_loader import ModelAdapter

import matplotlib.pyplot as plt

from validate import Validator

"""stss = ScrollTimeSeriesSplitLoader(df, 40, 0.5, 0.5)
fold, _ = stss.get_current_fold()

stss.next_fold()
fold, _ = stss.get_current_fold()
print(f"================\nstss: {stss.fold_num}\n")
print(fold.tail())
print(_.head())
print(fold.shape)

tss = TimeSeriesSplitLoader_(df, 12, 9)


tss_fold, tss_ = tss.get_current_fold()
print(f"================\ntss: {tss.fold_num}\n")
print(tss_fold.tail())
print(tss_.head())
print(tss_fold.shape)
print(tss_.shape)

tss.next_fold()
tss_fold, tss_ = tss.get_current_fold()
print(f"================\ntss: {tss.fold_num}\n")
print(tss_fold.tail())
print(tss_.head())
print(tss_fold.shape)
print(tss_.shape)

tss.next_fold()
tss_fold, tss_ = tss.get_current_fold()
print(f"================\ntss: {tss.fold_num}\n")
print(tss_fold.tail())
print(tss_.head())
print(tss_fold.shape)
print(tss_.shape)"""

# const
"""params = {
    "n_predict": 10,
    "type": "Median"
}
adapter.set_model("ConstPrediction", params)
adapter.fit({})
pred = adapter.predict({"ts":train})
print(pred.tail(10))"""

# rolling mean
"""params = {
    "time_step": 1,
    "n_predict": 20,
    "window_size": 5,
    "weights_coeffs": [],
    "weights_type": "new",
}
adapter.set_model("RollingMean", params)
fit_params = {}
adapter.fit(fit_params)
predict_params = {
    "timestamps": pd.Series(),
    "ts": train
}
pred = adapter.predict(predict_params)
print(data[80:100])
print(pred.tail(20))
print(WAPE(pd.Series(pred.iloc[80:100, -1]), pd.Series(data[80:100])))"""

# croston
"""params = {
    "alpha": 0.2,
    "beta": 0.8,
    "n_predict": 10,
    "time_step": "W",
}
adapter.set_model("CrostonTSB", params)
fit_params = {}
adapter.fit(fit_params)
predict_params = {
    "timestamps": df.index.to_series().iloc[[0, 80]],
    "ts": train
}
pred = adapter.predict(predict_params)
print(data[80:90])
print(pred.tail(10))"""


df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "value", "DATE": "date"})
print(df.head())

df['date'] = pd.to_datetime(df['date'])

gen = FeaturesGenerator(r"src/ts_val_shuffle/config.json")
new = gen.generate_features(df)

new = new.drop(columns=['date'])

X = new.drop(columns=['value'])
y = new['value']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

adapter = ModelAdapter()

# Elastic Net
"""init_params = {
    "alpha": 0.1,
    "l1_ratio": 0.5,
    "max_iter": 1000,
    "random_state": 42,
}
adapter.set_model("Elastic Net", init_params)
fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)
predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(y_test)))"""

# Huber
"""init_params = {
    "epsilon": 1.35, 
    "max_iter": 1000, 
    "alpha": 0.0001, 
    "tol": 1e-05
}
adapter.set_model("Huber", init_params)
fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)
predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(y_test)))"""

# Lasso
"""init_params = {
    "alpha": 0.01, 
    "max_iter": 10000, 
    "random_state": 42, 
}
adapter.set_model("Lasso", init_params)
fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)
predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(y_test)))"""

# RANSAC
"""base_estimator = LinearRegression()

init_params = {
    "estimator": base_estimator, 
    "min_samples": 0.5, 
    "max_trials": 100,
    "residual_threshold": None,
    "random_state": 42
}
adapter.set_model("RANSAC", init_params)
fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)
predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(y_test)))"""

# Ridge + features
"""init_params = {
    "alpha": 0.01, 
    "max_iter": 10000, 
    "random_state": 42, 
}
adapter.set_model("Ridge", init_params)
fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)
predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(y_test)))"""

# TheilSen + features
"""init_params = {
    "max_iter": 300, 
    "random_state": 42, 
}
adapter.set_model("TheilSen", init_params)
fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)
predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(y_test)))"""

# Random forest + features
"""init_params = {
    'n_estimators': 50,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}
adapter.set_model("Random Forest", init_params)
fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)
predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(y_test)))"""

# Exponential smoothing
"""df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "value", "DATE": "date"})
print(df.head())

ts = df['value'].iloc[:-20]

test = df['value'].iloc[-20:]

init_params = {
    'endog': ts,
    "initialization_method": "estimated",
}
adapter.set_model("Exponential Smoothing", init_params)
fit_params = {
    "optimized": True,
}
adapter.fit(fit_params)
predict_params = {
    "steps": 10
}
pred = adapter.forecast(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(test)))"""

# Holt
"""df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "value", "DATE": "date"})
print(df.head())

ts = df['value'].iloc[:-20]

test = df['value'].iloc[-20:]

init_params = {
    'endog': ts,
    "initialization_method": "estimated",
}
adapter.set_model("Holt", init_params)
fit_params = {
    "optimized": True,
}
adapter.fit(fit_params)
predict_params = {
    "steps": 10
}
pred = adapter.forecast(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(test)))"""

# Holt Winters
"""df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "value", "DATE": "date"})
print(df.head())

ts = df['value'].iloc[:-20]

test = df['value'].iloc[-20:]

init_params = {
    'endog': ts,
    "initialization_method": "estimated",
    "trend": "add",
    "seasonal": "add",
    "seasonal_periods": 12,
}
adapter.set_model("Holt Winters", init_params)
fit_params = {
    "optimized": True,
}
adapter.fit(fit_params)
predict_params = {
    "steps": 10
}
pred = adapter.forecast(predict_params)
print(y_test[-10:])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(test)))"""

# Prophet
"""df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "y", "DATE": "ds"})
print(df.head())

#gen = FeaturesGenerator(r"src/ts_val_shuffle/config copy.json")
#new = gen.generate_features(df)

#features = new.drop(columns=['y', 'ds']).columns.to_list()

train_df = df.iloc[:-20]

test_df = df.iloc[-20:]

init_params = {
    "growth": "linear",
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
}
adapter.set_model("Prophet", init_params)
fit_params = {
    "df": train_df,
}
adapter.fit(fit_params)

mfd_params = {
    "periods": 20,
    "freq": 'MS'
}

future = adapter.make_future_dataframe(mfd_params)

predict_params = {
    "df": future
}
pred = adapter.predict(predict_params)
plot_params = {
    'figsize': (8, 6)
}
plot = adapter.plot(pred, plot_params)
plot_comp = adapter.plot_components(pred, plot_params)
print(test_df['y'])
print(pred['yhat'].iloc[-20:])
print(WAPE(pred['yhat'].iloc[-20:], pd.Series(test_df['y'][-20:])))
plt.show()"""

# Prophet + features
"""df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "y", "DATE": "ds"})
print(df.head())

df['ds'] = pd.to_datetime(df['ds'], freq='MS')

gen = FeaturesGenerator(r"src/ts_val_shuffle/config copy.json")
new = gen.generate_features(df)


features = new.drop(columns=['y', 'ds']).columns.to_list()

train_df = new.iloc[:-20]

test_df = new.iloc[-20:]

init_params = {
    "growth": "linear",
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
}
adapter.set_model("Prophet", init_params)

for feat in features:
    adapter.model.add_regressor(feat)

fit_params = {
    "df": train_df,
}
adapter.fit(fit_params)

mfd_params = {
    "periods": 20,
    "freq": 'MS'
}

future = adapter.make_future_dataframe(mfd_params)

for feat in features:
    future[feat] = train_df[feat].iloc[-1]

predict_params = {
    "df": future
}
pred = adapter.predict(predict_params)
plot_params = {
    'figsize': (8, 6)
}
plot = adapter.plot(pred, plot_params)
plot_comp = adapter.plot_components(pred, plot_params)
print(test_df['y'])
print(pred['yhat'].iloc[-20:])
print(WAPE(pred['yhat'].iloc[-20:], pd.Series(test_df['y'][-20:])))
plt.show()"""

# SARIMAX
"""df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "value", "DATE": "date"})
df['date'] = pd.to_datetime(df['date'])

df.set_index('date', inplace=True)
df.index = pd.DatetimeIndex(df.index.values,  freq='MS')
print(df.head())

ts = df['value'].iloc[:-20]

test = df['value'].iloc[-20:]

init_params = {
    'endog': ts,
    "order": (1, 1, 1),
    "seasonal_order": (1, 1, 1, 12),
    "trend": "t",
    "enforce_stationarity": False,
    "enforce_invertibility": False,
}
adapter.set_model("SARIMA", init_params)
fit_params = {
    "disp": False,
}
adapter.fit(fit_params)
predict_params = {
    "steps": 10
}
pred = adapter.forecast(predict_params)
print(pred)
print(test[0:10])
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], pd.Series(test[0:10])))"""

# catboost
"""init_params = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "random_seed": 42,
    "verbose": False,
}

adapter.set_model("catboost", init_params)

fit_params = {
    "X": X_train,
    "y": y_train
}
adapter.fit(fit_params)

predict_params = {
    "X": X_test
}
pred = adapter.predict(predict_params)

# Вывод результатов
print(y_test.tail(10))
print(pred.tail(10))
print(WAPE(pred.iloc[:, 0], y_test))"""

# Fourie
"""df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "value", "DATE": "date"})
df['date'] = pd.to_datetime(df['date'])

gen = FeaturesGenerator(r"src/ts_val_shuffle/fourie.json")
new = gen.generate_features(df)

init_params = {
    "order": 5, # обязательный параметр
    "absolute_sigma": True, # опционально
}

adapter.set_model("Fourie", init_params)

fit_params = {
    "X_data": new['time_month'].iloc[:-20],
    "y_data": new['value'].iloc[:-20]
}

adapter.fit(fit_params)

pred_params = {
    "X": new['time_month'].iloc[-20:],
}

pred = adapter.predict(pred_params)

plt.figure(figsize=(12,6))
plt.plot(df['date'].iloc[-20:], new['value'].iloc[-20:], label='Исходные данные')
plt.plot(df['date'].iloc[-20:], pred, label='Аппроксимация Fourier', linestyle='--')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.title('Аппроксимация временного ряда с помощью FourierModel')
plt.legend()
plt.show()"""

df = pd.read_csv(r"src\ts_val_shuffle\Electric_Production.csv")

df = df.rename(columns={"IPG2211A2N": "value", "DATE": "date"})

df['date'] = pd.to_datetime(df['date'])
print(df.head())

init_params = {
    "alpha": 0.1,
    "l1_ratio": 0.5,
    "max_iter": 1000,
    "random_state": 42,
}

val = Validator()
train = df.iloc[:-30]
test = df.iloc[-30:]
val.set_data(train)

init_params = {
    #'endog': df,
    "order": (1, 1, 1),
    "seasonal_order": (1, 1, 1, 12),
    "trend": "t",
    "enforce_stationarity": False,
    "enforce_invertibility": False,
}

init_params = {
    "growth": "linear",
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
}

init_params = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "random_seed": 42,
    "verbose": False,
}

init_params = {
    #'endog': ts,
    "initialization_method": "estimated",
    "trend": "add",
    "seasonal": "add",
    "seasonal_periods": 12,
}

init_params = {
    "order": 5, # обязательный параметр
    #"absolute_sigma": True, # опционально
}

init_params = {
    "n_predict": 10,
    "type": "Median"
}

init_params = {
    "time_step": 1,
    "window_size": 3,
    "weights_coeffs": [],
    "weights_type": "new",
}

init_params = {
    "alpha": 0.15,
    "beta": 0.95,
    "time_step": "MS",
}

val.set_generator(r"src/ts_val_shuffle/config.json")
#val.set_model("Elastic Net", init_params)
#val.set_model("SARIMA", init_params)
#val.set_model("Prophet", init_params)
#val.set_model("catboost", init_params)
#val.set_model("Holt Winters", init_params)
#val.set_model("Fourie", init_params)
#val.set_model("ConstPrediction", init_params)
#val.set_model("RollingMean", init_params)

val.set_model("CrostonTSB", init_params)


val.set_split_method("expanding", 10)
val.validate("WAPE", 'value', 'date', shuffling=False)


print(val.metric_values)

print(test['value'])
print(test['date'])
pred = val.predict({'X': test['date']})
print(pred)
plt.plot(val.metric_values)
plt.show()