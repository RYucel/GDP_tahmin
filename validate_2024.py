# -*- coding: utf-8 -*-
"""2024 validasyon testi: 1982-2023 ile egit, 2024'u tahmin et."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

df = pd.read_csv('GDP_veriler.csv')
df.columns = df.columns.str.strip()
if 'deflator' not in df.columns:
    for c in df.columns:
        if 'deflat' in c.lower():
            df.rename(columns={c: 'deflator'}, inplace=True)

# DummyCorona (2020+2021)
df['DummyCorona'] = 0
df.loc[df['Year'].isin([2020, 2021]), 'DummyCorona'] = 1

features = ['PubSPEN77', 'DEPOSIT77', 'IMP77', 'KREDI77', 'POP',
            'ElectricKwH', 'DummyCorona', 'USDTRYchg', 'CPIchg']

# Egitim: 1982-2023, Test: 2024
mask_all = df['GDP77'].notna()
for f in features:
    mask_all = mask_all & df[f].notna()
full = df[mask_all].sort_values('Year')

train = full[full['Year'] <= 2023]
test = full[full['Year'] == 2024]

X_train = train[features].values
y_train = train['GDP77'].values
X_test = test[features].values
y_true = test['GDP77'].values[0]  # 21310.57

scaler = StandardScaler()
scaler.fit(X_train)
X_tr_sc = scaler.transform(X_train)
X_te_sc = scaler.transform(X_test)

models = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.5, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=42),
    'SVR': SVR(kernel='rbf', C=1000, epsilon=100),
    'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=3, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=3, num_leaves=8, random_state=42, verbose=-1),
}

# ARDL (lag dahil)
train_ardl = train.copy()
train_ardl['GDP77_lag1'] = train_ardl['GDP77'].shift(1)
ardl_features = features + ['GDP77_lag1']
train_ardl = train_ardl.dropna(subset=ardl_features)
X_ardl = train_ardl[ardl_features].values
y_ardl = train_ardl['GDP77'].values
sc_ardl = StandardScaler()
sc_ardl.fit(X_ardl)
ardl_model = LinearRegression()
ardl_model.fit(sc_ardl.transform(X_ardl), y_ardl)
# 2024 tahmin icin 2023 GDP lag kullan
gdp_2023 = train[train['Year']==2023]['GDP77'].values[0]
test_ardl = test[features].copy()
test_ardl['GDP77_lag1'] = gdp_2023
ardl_pred = ardl_model.predict(sc_ardl.transform(test_ardl[ardl_features].values))[0]

# LOO RMSE hesapla (egitim setinde) + 2024 tahmin
results = {}
preds_2024 = {}
loo = LeaveOneOut()

for name, model in models.items():
    # LOO
    lp = np.zeros(len(y_train))
    for ti, si in loo.split(X_tr_sc):
        model.fit(X_tr_sc[ti], y_train[ti])
        lp[si] = model.predict(X_tr_sc[si])
    loo_rmse = np.sqrt(mean_squared_error(y_train, lp))
    
    # 2024 tahmin
    model.fit(X_tr_sc, y_train)
    pred = model.predict(X_te_sc)[0]
    error = pred - y_true
    pct_error = error / y_true * 100
    
    results[name] = {'loo_rmse': loo_rmse, 'pred': pred, 'error': error, 'pct_error': pct_error}
    preds_2024[name] = pred

# ARDL sonuc
ardl_loo_rmse = 0
lp_ardl = np.zeros(len(y_ardl))
loo2 = LeaveOneOut()
X_ardl_sc = sc_ardl.transform(X_ardl)
for ti, si in loo2.split(X_ardl_sc):
    ardl_model.fit(X_ardl_sc[ti], y_ardl[ti])
    lp_ardl[si] = ardl_model.predict(X_ardl_sc[si])
ardl_loo_rmse = np.sqrt(mean_squared_error(y_ardl, lp_ardl))
ardl_model.fit(X_ardl_sc, y_ardl)

results['ARDL'] = {
    'loo_rmse': ardl_loo_rmse,
    'pred': ardl_pred,
    'error': ardl_pred - y_true,
    'pct_error': (ardl_pred - y_true) / y_true * 100
}
preds_2024['ARDL'] = ardl_pred

# Ensemble (top 5)
sorted_models = sorted(results.items(), key=lambda x: x[1]['loo_rmse'])
top5 = sorted_models[:5]
inv = [1/r['loo_rmse'] for _, r in top5]
tot = sum(inv)
wts = [w/tot for w in inv]
ensemble_pred = sum(w * results[n]['pred'] for (n, _), w in zip(top5, wts))
ensemble_error = ensemble_pred - y_true
ensemble_pct = ensemble_error / y_true * 100

print("="*70)
print(f"  2024 VALIDASYON TESTI")
print(f"  Egitim: 1982-2023 ({len(y_train)} gozlem)")
print(f"  Gercek 2024 GDP77: {y_true:,.2f}")
print("="*70)

print(f"\n{'Model':<20} {'LOO RMSE':>10} {'Tahmin':>10} {'Hata':>10} {'Hata%':>8}")
print("-"*60)
for name, r in sorted(results.items(), key=lambda x: abs(x[1]['pct_error'])):
    marker = " <--" if abs(r['pct_error']) == min(abs(r2['pct_error']) for r2 in results.values()) else ""
    print(f"{name:<20} {r['loo_rmse']:>10.2f} {r['pred']:>10.1f} {r['error']:>+10.1f} {r['pct_error']:>+7.2f}%{marker}")

print(f"\n{'ENSEMBLE (top5)':<20} {'':>10} {ensemble_pred:>10.1f} {ensemble_error:>+10.1f} {ensemble_pct:>+7.2f}%")

print(f"\n  Ensemble agirliklar:")
for (n, _), w in zip(top5, wts):
    print(f"    {n:<20} {w:.4f}")

# En iyi tek model vs ensemble
best_single = min(results.items(), key=lambda x: abs(x[1]['pct_error']))
print(f"\n{'='*60}")
print(f"  En iyi tek model: {best_single[0]} (hata: {best_single[1]['pct_error']:+.2f}%)")
print(f"  Ensemble hata: {ensemble_pct:+.2f}%")
print(f"  Ridge hata: {results['Ridge']['pct_error']:+.2f}%")
if abs(ensemble_pct) < abs(results['Ridge']['pct_error']):
    print(f"  >>> ENSEMBLE daha iyi ({abs(ensemble_pct):.2f}% vs {abs(results['Ridge']['pct_error']):.2f}%)")
else:
    print(f"  >>> RIDGE daha iyi ({abs(results['Ridge']['pct_error']):.2f}% vs {abs(ensemble_pct):.2f}%)")
print(f"{'='*60}")
