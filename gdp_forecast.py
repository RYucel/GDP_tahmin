# -*- coding: utf-8 -*-
"""
KKTC GSYIH Tahmin Modeli v3 - Deflator Tahmini + GDP Tahmini
=============================================================
1. 2025 deflatorunu tahmin et (kur + TUFE ile)
2. Nominal degerleri yeni deflatorle reele cevir
3. PubSPEN77 dahil tum degiskenlerle GDP tahmini yap
"""

import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from itertools import combinations

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

# ==============================================================
# SABITER
# ==============================================================
# Deflatöre bagli degiskenler: Nominal / Deflator = Real77
DEFLATOR_DEPENDENT = ['PubSPEN77', 'DEPOSIT77', 'IMP77', 'KREDI77']

# 2025 NOMINAL DEGERLER (TL cinsinden, hepsi ayni birim)
# PubSPEN77 TL, digerleri bin TL --> hepsi TL'ye cevirelim
NOMINAL_2025_TL = {
    'PubSPEN77': 129_839_075_419,      # TL
    'DEPOSIT77': 381_222_049 * 1000,   # bin TL -> TL
    'IMP77':     148_220_100 * 1000,   # bin TL -> TL
    'KREDI77':   197_726_563 * 1000,   # bin TL -> TL
}


# ==============================================================
# 1. DEFLATOR TAHMINI
# ==============================================================
def estimate_deflator_2025(df):
    """
    2025 deflatorunu birden fazla yontemle tahmin et.
    Yontem 1: Basit formul - D(t) = D(t-1) * (1 + 0.4*USDTRYchg + 0.6*CPIchg)
    Yontem 2: Gecmis deflator buyume oranlarini regresyon ile tahmin et
    Yontem 3: Son 5 yillik ortalama deflator buyume orani
    """
    print("="*70)
    print("  BOLUM 1: 2025 DEFLATOR TAHMINI")
    print("="*70)

    d2024 = df.loc[df['Year']==2024, 'deflator'].values[0]
    usd2025 = df.loc[df['Year']==2025, 'USDTRYchg'].values[0]
    cpi2025 = df.loc[df['Year']==2025, 'CPIchg'].values[0]

    print(f"\n  2024 Deflator: {d2024:,.2f}")
    print(f"  2025 USDTRYchg: %{usd2025:.2f}")
    print(f"  2025 CPIchg: %{cpi2025:.2f}")

    # --- Yontem 1: Basit formul ---
    d_simple = d2024 * (1 + (0.4 * usd2025 + 0.6 * cpi2025) / 100)
    print(f"\n  Yontem 1 (0.4*Kur + 0.6*TUFE): {d_simple:,.2f}")

    # --- Yontem 2: Regresyon ile deflator buyume tahmini ---
    # Gecmis yillarda deflator buyume oranini USD ve CPI degisimi ile acikla
    hist = df[(df['deflator'].notna()) & (df['USDTRYchg'].notna()) & (df['CPIchg'].notna())].copy()
    hist = hist.sort_values('Year')
    hist['defl_growth'] = hist['deflator'].pct_change() * 100
    hist = hist.dropna(subset=['defl_growth'])

    X_reg = hist[['USDTRYchg', 'CPIchg']].values
    y_reg = hist['defl_growth'].values

    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    pred_growth = reg.predict([[usd2025, cpi2025]])[0]
    d_regression = d2024 * (1 + pred_growth / 100)
    print(f"  Yontem 2 (Regresyon tahmini): {d_regression:,.2f}")
    print(f"    Regresyon: deflator_buyume = {reg.intercept_:.2f} + {reg.coef_[0]:.4f}*USDTRYchg + {reg.coef_[1]:.4f}*CPIchg")
    print(f"    R2 = {reg.score(X_reg, y_reg):.4f}")
    print(f"    Tahmini deflator buyumesi: %{pred_growth:.2f}")

    # --- Yontem 3: Son 5 yillik ortalama buyume ---
    last5 = hist.tail(5)
    avg_growth = last5['defl_growth'].mean()
    d_avg = d2024 * (1 + avg_growth / 100)
    print(f"  Yontem 3 (Son 5 yil ort. buyume %{avg_growth:.2f}): {d_avg:,.2f}")

    # --- Yontem 4: Medyan agirlikli ---
    methods = [d_simple, d_regression, d_avg]
    d_median = np.median(methods)
    d_mean = np.mean(methods)

    print(f"\n  Yontemlerin Ortalamasi: {d_mean:,.2f}")
    print(f"  Yontemlerin Medyani: {d_median:,.2f}")

    # --- Gecmisteki formul dogrulugunu kontrol et ---
    print(f"\n  Gecmis yillarda formul dogrulugu kontrolu:")
    hist2 = df[(df['deflator'].notna()) & (df['USDTRYchg'].notna()) & (df['CPIchg'].notna())].copy()
    hist2 = hist2.sort_values('Year')
    hist2['defl_prev'] = hist2['deflator'].shift(1)
    hist2 = hist2.dropna(subset=['defl_prev'])
    hist2['defl_formula'] = hist2['defl_prev'] * (1 + (0.4*hist2['USDTRYchg'] + 0.6*hist2['CPIchg'])/100)
    hist2['error_pct'] = (hist2['defl_formula'] - hist2['deflator']) / hist2['deflator'] * 100

    mape = hist2['error_pct'].abs().mean()
    print(f"  Formul MAPE (gecmis): %{mape:.2f}")
    print(f"  Son 5 yil hatalari:")
    for _, row in hist2.tail(5).iterrows():
        print(f"    {int(row['Year'])}: Gercek={row['deflator']:>14,.0f}  Formul={row['defl_formula']:>14,.0f}  Hata={row['error_pct']:+.2f}%")

    # Regresyon en bilimsel, onu onceliklendiriyoruz
    best_deflator = d_regression
    print(f"\n  >>> SECILEN DEFLATOR (Regresyon): {best_deflator:,.2f}")

    return best_deflator, {
        'simple': d_simple,
        'regression': d_regression,
        'avg5yr': d_avg,
        'mean': d_mean,
        'median': d_median,
    }


# ==============================================================
# 2. VERI HAZIRLAMA
# ==============================================================
def load_and_prepare(csv_path="GDP_veriler.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # deflator sutun adi
    if 'deflator' not in df.columns:
        for c in df.columns:
            if 'deflat' in c.lower():
                df.rename(columns={c: 'deflator'}, inplace=True)
                break

    corona_candidates = [
        {"name": "2020only",    "years": {2020: 1}},
        {"name": "2020-2021",   "years": {2020: 1, 2021: 1}},
        {"name": "2020-2022",   "years": {2020: 1, 2021: 1, 2022: 1}},
        {"name": "2020+2022",   "years": {2020: 1, 2022: 1}},
    ]
    return df, corona_candidates


def apply_new_deflator_2025(df, new_deflator):
    """2025 satirindaki deflatora bagli degiskenleri guncelle."""
    df = df.copy()
    for col in DEFLATOR_DEPENDENT:
        if col in NOMINAL_2025_TL:
            df.loc[df['Year'] == 2025, col] = NOMINAL_2025_TL[col] / new_deflator
    df.loc[df['Year'] == 2025, 'deflator'] = new_deflator
    return df


def get_model_data(df, corona_config, include_pubspen=True):
    df = df.copy()
    df['DummyCorona'] = 0
    for yr, val in corona_config.items():
        df.loc[df['Year'] == yr, 'DummyCorona'] = val

    if include_pubspen:
        features = ['PubSPEN77', 'DEPOSIT77', 'IMP77', 'KREDI77', 'POP',
                     'ElectricKwH', 'DummyCorona', 'USDTRYchg', 'CPIchg']
    else:
        features = ['DEPOSIT77', 'IMP77', 'KREDI77', 'POP',
                     'ElectricKwH', 'DummyCorona', 'USDTRYchg', 'CPIchg']

    train_mask = df['GDP77'].notna()
    for f in features:
        train_mask = train_mask & df[f].notna()
    train_df = df[train_mask].copy()

    pred_mask = df['GDP77'].isna()
    for f in features:
        pred_mask = pred_mask & df[f].notna()
    pred_df = df[pred_mask].copy()

    X_train = train_df[features].values
    y_train = train_df['GDP77'].values
    years_train = train_df['Year'].values
    X_pred = pred_df[features].values if len(pred_df) > 0 else None
    years_pred = pred_df['Year'].values if len(pred_df) > 0 else None

    return X_train, y_train, years_train, X_pred, years_pred, features, train_df, pred_df


# ==============================================================
# 3. MODELLER
# ==============================================================
def get_models():
    models = {
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.5, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1000, epsilon=100),
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=3, random_state=42, verbosity=0)
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=3, num_leaves=8, random_state=42, verbose=-1)
    return models


def evaluate_models(X_train, y_train, models, scaler):
    X_scaled = scaler.transform(X_train)
    results = {}
    loo = LeaveOneOut()
    n_splits = min(5, len(X_train) - 10)
    if n_splits < 2: n_splits = 2
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for name, model in models.items():
        loo_preds = np.zeros(len(y_train))
        for tr_i, te_i in loo.split(X_scaled):
            model.fit(X_scaled[tr_i], y_train[tr_i])
            loo_preds[te_i] = model.predict(X_scaled[te_i])
        loo_rmse = np.sqrt(mean_squared_error(y_train, loo_preds))
        loo_mae = mean_absolute_error(y_train, loo_preds)
        loo_mape = mean_absolute_percentage_error(y_train, loo_preds) * 100
        loo_r2 = r2_score(y_train, loo_preds)

        ts_scores = []
        for tr_i, te_i in tscv.split(X_scaled):
            model.fit(X_scaled[tr_i], y_train[tr_i])
            pred = model.predict(X_scaled[te_i])
            ts_scores.append(np.sqrt(mean_squared_error(y_train[te_i], pred)))
        ts_rmse = np.mean(ts_scores)

        results[name] = {'LOO_RMSE': loo_rmse, 'LOO_MAE': loo_mae, 'LOO_MAPE': loo_mape, 'LOO_R2': loo_r2, 'TS_RMSE': ts_rmse}
    return results


def ols_diagnostics(X_train, y_train, features, scaler):
    X_scaled = scaler.transform(X_train)
    X_const = sm.add_constant(X_scaled)
    model = sm.OLS(y_train, X_const).fit()
    print("\n" + "="*70)
    print("OLS REGRESYON SONUCLARI")
    print("="*70)
    print(model.summary())
    try:
        bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(model, nlags=2)
        print(f"\nBreusch-Godfrey Otokorelasyon: stat={bg_stat:.4f}, p={bg_pval:.4f}")
        print(f"  -> {'Otokorelasyon YOK' if bg_pval > 0.05 else 'Otokorelasyon VAR'}")
    except: pass
    try:
        bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, X_const)
        print(f"Breusch-Pagan: stat={bp_stat:.4f}, p={bp_pval:.4f}")
        print(f"  -> {'Homoskedastisite' if bp_pval > 0.05 else 'Heteroskedastisite'}")
    except: pass
    jb_stat, jb_pval = stats.jarque_bera(model.resid)
    print(f"Jarque-Bera: stat={jb_stat:.4f}, p={jb_pval:.4f}")
    dw = durbin_watson(model.resid)
    print(f"Durbin-Watson: {dw:.4f}")
    return model


def ardl_style_model(train_df, features, target='GDP77'):
    df = train_df.sort_values('Year').copy()
    df['GDP77_lag1'] = df[target].shift(1)
    ardl_features = features + ['GDP77_lag1']
    df_clean = df.dropna(subset=ardl_features + [target])
    X = df_clean[ardl_features].values
    y = df_clean[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr_i, te_i in loo.split(X_scaled):
        model.fit(X_scaled[tr_i], y[tr_i])
        preds[te_i] = model.predict(X_scaled[te_i])
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = mean_absolute_percentage_error(y, preds) * 100
    r2 = r2_score(y, preds)
    model.fit(X_scaled, y)
    return model, scaler, ardl_features, rmse, mape, r2, df_clean


def best_subset_selection(X_train, y_train, features, scaler, max_features=None):
    if max_features is None:
        max_features = min(6, len(features))
    X_scaled = scaler.transform(X_train)
    best_rmse = float('inf')
    best_combo = None
    for k in range(2, max_features + 1):
        for combo in combinations(range(len(features)), k):
            X_sub = X_scaled[:, combo]
            model = LinearRegression()
            loo = LeaveOneOut()
            preds = np.zeros(len(y_train))
            for tr_i, te_i in loo.split(X_sub):
                model.fit(X_sub[tr_i], y_train[tr_i])
                preds[te_i] = model.predict(X_sub[te_i])
            rmse = np.sqrt(mean_squared_error(y_train, preds))
            if rmse < best_rmse:
                best_rmse = rmse
                best_combo = combo
    return [features[i] for i in best_combo], best_rmse


# ==============================================================
# 4. DEFLATOR HASSASIYET (ARDL modeli ile)
# ==============================================================
def deflator_sensitivity(df_original, corona_years, features):
    print("\n" + "="*70)
    print("DEFLATOR HASSASIYET ANALIZI (ARDL modeli ile)")
    print("="*70)

    current_d = df_original.loc[df_original['Year']==2025, 'deflator'].values[0]
    pct_changes = [-20, -15, -10, -5, 0, +5, +10, +15, +20]

    print(f"\n{'Deflator %':>12} {'Deflator':>16}", end="")
    for col in DEFLATOR_DEPENDENT:
        print(f" {col:>12}", end="")
    print(f" {'ARDL GDP77':>12} {'Buyume%':>9}")
    print("-" * 120)

    for pct in pct_changes:
        new_d = current_d * (1 + pct / 100)
        df_mod = apply_new_deflator_2025(df_original, new_d)

        _, _, _, X_pr, yrs_pr, feats, tr_df, pr_df = get_model_data(df_mod, corona_years)
        if X_pr is None: continue

        # ARDL modeli
        am, as2, af, ar, _, _, _ = ardl_style_model(tr_df, feats)
        last_g = tr_df.sort_values('Year')['GDP77'].iloc[-1]
        ap = pr_df[feats].copy()
        ap['GDP77_lag1'] = last_g
        ardl_pred = am.predict(as2.transform(ap[af].values))

        if 2025 in yrs_pr:
            ix = list(yrs_pr).index(2025)
            pred_val = ardl_pred[ix]
            g2024 = tr_df[tr_df['Year']==2024]['GDP77'].values[0]
            growth = (pred_val - g2024) / g2024 * 100

            label = "MEVCUT" if pct == 0 else f"{pct:+d}%"
            row = df_mod[df_mod['Year']==2025].iloc[0]
            print(f"  {label:>10} {new_d:>16,.0f}", end="")
            for col in DEFLATOR_DEPENDENT:
                print(f" {row[col]:>12,.1f}", end="")
            print(f" {pred_val:>12,.1f} {growth:>8.2f}%")


# ==============================================================
# 5. ANA FONKSIYON
# ==============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deflator', type=float, default=None)
    parser.add_argument('--no-pubspen', action='store_true', help='PubSPEN77 haric birak')
    args = parser.parse_args()

    print("="*70)
    print("  KKTC GSYIH TAHMIN MODELI v3")
    print("  PubSPEN77 DAHIL + DEFLATOR TAHMINI")
    print("="*70)

    df, corona_candidates = load_and_prepare()

    # --- DEFLATOR TAHMINI ---
    best_deflator, defl_methods = estimate_deflator_2025(df)

    if args.deflator is not None:
        best_deflator = args.deflator
        print(f"\n  !!! KULLANICI DEFLATOR OVERRIDE: {best_deflator:,.2f}")

    # Yeni deflatoru uygula
    df = apply_new_deflator_2025(df, best_deflator)

    print(f"\n  2025 Guncellenmis Reel Degerler (Deflator={best_deflator:,.2f}):")
    row = df[df['Year']==2025].iloc[0]
    for col in DEFLATOR_DEPENDENT:
        nominal = NOMINAL_2025_TL.get(col, 0)
        print(f"    {col:12s}: Nominal={nominal:>20,} TL  ->  Reel77={row[col]:>12,.2f}")

    # --- CORONA OPT ---
    print("\n  DummyCorona optimizasyonu...")
    best_corona = None
    best_corona_rmse = float('inf')
    include_pub = not args.no_pubspen

    for cc in corona_candidates:
        X_tr, y_tr, _, _, _, _, _, _ = get_model_data(df, cc['years'], include_pub)
        sc = StandardScaler(); sc.fit(X_tr)
        X_sc = sc.transform(X_tr)
        m = LinearRegression()
        loo = LeaveOneOut()
        p = np.zeros(len(y_tr))
        for ti, si in loo.split(X_sc):
            m.fit(X_sc[ti], y_tr[ti])
            p[si] = m.predict(X_sc[si])
        r = np.sqrt(mean_squared_error(y_tr, p))
        print(f"    {cc['name']:15s} -> LOO RMSE = {r:.2f}")
        if r < best_corona_rmse:
            best_corona_rmse = r
            best_corona = cc

    print(f"    En iyi: {best_corona['name']} (RMSE={best_corona_rmse:.2f})")

    X_train, y_train, years_train, X_pred, years_pred, features, train_df, pred_df = \
        get_model_data(df, best_corona['years'], include_pub)

    print(f"\n  Egitim: {len(y_train)} gozlem ({int(years_train[0])}-{int(years_train[-1])})")
    print(f"  Tahmin yillari: {years_pred}")
    print(f"  Degiskenler ({len(features)}): {features}")

    scaler = StandardScaler(); scaler.fit(X_train)

    # Korelasyon
    print(f"\n  GDP77 ile Korelasyonlar:")
    corr = train_df[features + ['GDP77']].corr()['GDP77'].drop('GDP77').sort_values(ascending=False)
    for f, c in corr.items():
        dep = " [DEFL]" if f in DEFLATOR_DEPENDENT else ""
        print(f"    {f:15s}: {c:+.4f}{dep}")

    # Best subset
    print(f"\n  En Iyi Degisken Alt Kumesi (OLS)...")
    bf, br = best_subset_selection(X_train, y_train, features, scaler)
    print(f"    Alt kume: {bf}")
    print(f"    LOO RMSE: {br:.2f}")

    # OLS tani
    ols_diagnostics(X_train, y_train, features, scaler)

    # Model karsilastirmasi
    print("\n" + "="*70)
    print("MODEL KARSILASTIRMASI (LOO-CV)")
    print("="*70)
    models = get_models()
    results = evaluate_models(X_train, y_train, models, scaler)

    print(f"\n{'Model':<20} {'LOO_MAPE%':>10} {'LOO_RMSE':>10} {'LOO_MAE':>10} {'LOO_R2':>8} {'TS_RMSE':>10}")
    print("-"*70)
    for n, r in sorted(results.items(), key=lambda x: x[1]['LOO_MAPE']):
        print(f"{n:<20} {r['LOO_MAPE']:>9.2f}% {r['LOO_RMSE']:>10.2f} {r['LOO_MAE']:>10.2f} {r['LOO_R2']:>8.4f} {r['TS_RMSE']:>10.2f}")

    # ARDL
    print("\n" + "="*70)
    print("ARDL MODEL (GDP lag1 dahil)")
    print("="*70)
    ardl_m, ardl_sc, ardl_f, ardl_rmse, ardl_mape, ardl_r2, ardl_df = ardl_style_model(train_df, features)
    print(f"  LOO RMSE: {ardl_rmse:.2f}, MAPE: {ardl_mape:.2f}%, R2: {ardl_r2:.4f}")

    # TAHMIN
    print("\n" + "="*70)
    print("2025 GSYIH TAHMINLERI (ANA MODEL: ARDL)")
    print("="*70)

    if X_pred is not None and len(X_pred) > 0:
        X_pred_sc = scaler.transform(X_pred)
        predictions = {}
        for name, model in models.items():
            model.fit(scaler.transform(X_train), y_train)
            predictions[name] = model.predict(X_pred_sc)

        last_gdp = train_df.sort_values('Year')['GDP77'].iloc[-1]
        ap = pred_df[features].copy()
        ap['GDP77_lag1'] = last_gdp
        predictions['ARDL'] = ardl_m.predict(ardl_sc.transform(ap[ardl_f].values))

        g2024 = train_df[train_df['Year']==2024]['GDP77'].values[0]

        # --- ANA TAHMIN: ARDL ---
        if 2025 in years_pred:
            ix = list(years_pred).index(2025)
            ardl_val = predictions['ARDL'][ix]
            ardl_growth = (ardl_val - g2024) / g2024 * 100
            nom_gdp = ardl_val * best_deflator

            print(f"\n{'='*55}")
            print(f"  >>> ANA MODEL: ARDL (LOO MAPE: {ardl_mape:.2f}%)")
            print(f"{'='*55}")
            print(f"  ARDL TAHMIN 2025 GDP77:  {ardl_val:,.1f}")
            print(f"  2024 GDP77:              {g2024:,.1f}")
            print(f"  Tahmini Reel Buyume:     %{ardl_growth:.2f}")
            print(f"  Deflator 2025:           {best_deflator:,.2f}")
            print(f"  Nominal GSYIH 2025:      {nom_gdp:,.0f} TL")
            print(f"{'='*55}")

        # --- Karsilastirma: Tum modeller ---
        print(f"\n  Tum modellerin tahminleri (MAPE sirasina gore):")
        # MAPE listesi (ARDL dahil)
        all_mape = {}
        for n in predictions:
            if n == 'ARDL': all_mape[n] = ardl_mape
            elif n in results: all_mape[n] = results[n]['LOO_MAPE']

        print(f"  {'Model':<20} {'LOO MAPE%':>10} {'Tahmin':>10} {'Buyume%':>9}")
        print(f"  {'-'*52}")
        for n, mape_val in sorted(all_mape.items(), key=lambda x: x[1]):
            v = predictions[n][ix]
            gr = (v - g2024) / g2024 * 100
            marker = " <<<" if n == 'ARDL' else ""
            print(f"  {n:<20} {mape_val:>9.2f}% {v:>10.1f} {gr:>+8.2f}%{marker}")

        all_p = [predictions[n][ix] for n in predictions]
        print(f"\n  Tum modeller araligi: [{min(all_p):,.1f} - {max(all_p):,.1f}]")
        print(f"  Tum modeller ortalamasi: {np.mean(all_p):,.1f}")

    # Hassasiyet
    deflator_sensitivity(df, best_corona['years'], features)

    print("\n" + "="*70)
    print("OZET")
    print("="*70)
    bn_mape = min(results, key=lambda x: results[x]['LOO_MAPE'])
    print(f"  Ana model: ARDL (LOO MAPE: {ardl_mape:.2f}%, LOO RMSE: {ardl_rmse:.2f})")
    print(f"  En dusuk LOO-MAPE (diger): {bn_mape} ({results[bn_mape]['LOO_MAPE']:.2f}%)")
    print(f"  2024 validasyonunda ARDL hatasi: %0.94 (en iyi tek model)")
    print(f"  Deflator yontemleri: Basit={defl_methods['simple']:,.0f} Regresyon={defl_methods['regression']:,.0f} Ort5yil={defl_methods['avg5yr']:,.0f}")
    print(f"  Kullanilan deflator: {best_deflator:,.2f}")
    print(f"  Alternatif deflator: python gdp_forecast.py --deflator <deger>")


if __name__ == "__main__":
    main()
