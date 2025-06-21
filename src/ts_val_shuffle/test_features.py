from features_generation import FeaturesGenerator
import pandas as pd
import numpy as np

dates = pd.date_range(start='2023-12-01', periods=100, freq='D')

data = np.random.randn(100).cumsum() 

df = pd.DataFrame({'date': dates, 'value': data})

df.set_index('date', inplace=True)
print(df.head())

gen = FeaturesGenerator(r"src/ts_val_shuffle/demo_config.json")
new = gen.generate_features(df)
print(new.sample(15))