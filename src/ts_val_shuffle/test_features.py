from features_generation import FeaturesGenerator
import pandas as pd
import numpy as np
from ts_split import ScrollTimeSeriesSplitLoader, TimeSeriesSplitLoader_

dates = pd.date_range(start='2023-12-01', periods=100, freq='D')

data = np.random.randn(100).cumsum() 

df = pd.DataFrame({'date': dates, 'value': data})

df.set_index('date', inplace=True)
#print(df.head())

#gen = FeaturesGenerator(r"src/ts_val_shuffle/demo_config.json")
#new = gen.generate_features(df)
#print(new.sample(15))

stss = ScrollTimeSeriesSplitLoader(df, 40, 0.5, 0.5)
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
print(tss_.shape)
