import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import LeaveOneOut
import warnings

warnings.filterwarnings('ignore')

# ----------------- PAGE CONFIG & STYLING -----------------
st.set_page_config(
    page_title="KKTC GSYIH Analizi ve Model Tahminleri",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Profesyonel "Finansal Terminal" stili CSS
st.markdown("""
<style>
/* Ana tema */
body {
    background-color: #0E1117;
    color: #FAFAFA;
}
.main .block-container {
    padding-top: 2rem;
}
/* Başlık */
.report-title {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 5px;
    background: -webkit-linear-gradient(45deg, #FF6B6B, #FCA048);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.report-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #888888;
    margin-bottom: 30px;
}
/* Metric Kutuları */
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------- CONSTANTS -----------------
DEFLATOR_DEPENDENT = ['PubSPEN77', 'DEPOSIT77', 'IMP77', 'KREDI77']
NOMINAL_2025_TL = {
    'PubSPEN77': 129_839_075_419,
    'DEPOSIT77': 381_222_049 * 1000,
    'IMP77':     148_220_100 * 1000,
    'KREDI77':   197_726_563 * 1000,
}

# ----------------- CACHED DATA & MODEL FUNCS -----------------
@st.cache_data
def load_data():
    df = pd.read_csv('GDP_veriler.csv')
    df.columns = df.columns.str.strip()
    # deflator isimlendirmesini fixle
    if 'deflator' not in df.columns:
        for c in df.columns:
            if 'deflat' in c.lower():
                df.rename(columns={c: 'deflator'}, inplace=True)
                break
    
    # 2020-2021 Corona Kukla Değişkeni (Araştırma raporuna dayanarak en iyisi)
    df['DummyCorona'] = 0
    df.loc[df['Year'].isin([2020, 2021]), 'DummyCorona'] = 1
    return df

def estimate_deflator(df):
    """ Regresyon ile 2025 Deflatör Tahmini """
    d2024 = df.loc[df['Year']==2024, 'deflator'].values[0]
    usd2025 = df.loc[df['Year']==2025, 'USDTRYchg'].values[0]
    cpi2025 = df.loc[df['Year']==2025, 'CPIchg'].values[0]
    
    hist = df[(df['deflator'].notna()) & (df['USDTRYchg'].notna()) & (df['CPIchg'].notna())].copy()
    hist = hist.sort_values('Year')
    hist['defl_growth'] = hist['deflator'].pct_change() * 100
    hist = hist.dropna(subset=['defl_growth'])
    
    X_reg = sm.add_constant(hist[['USDTRYchg', 'CPIchg']].values)
    y_reg = hist['defl_growth'].values
    reg = sm.OLS(y_reg, X_reg).fit()
    pred_growth = reg.predict([1, usd2025, cpi2025])[0]
    
    d_regression = d2024 * (1 + pred_growth / 100)
    
    # Basit Formül ve 5 yıl ortalaması (referans için)
    d_simple = d2024 * (1 + (0.4 * usd2025 + 0.6 * cpi2025) / 100)
    avg_growth = hist.tail(5)['defl_growth'].mean()
    d_avg = d2024 * (1 + avg_growth / 100)
    
    return d_regression, d_simple, d_avg, pred_growth

def fit_ardl_model(df_clean, features, target='GDP77'):
    # Scale Data
    X = df_clean[features].values
    y = df_clean[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_const = sm.add_constant(X_scaled, prepend=True)
    
    # OLS Modeli
    model = sm.OLS(y, X_scaled_const).fit()
    
    # LOO CV
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr_i, te_i in loo.split(X_scaled_const):
        m = sm.OLS(y[tr_i], X_scaled_const[tr_i]).fit()
        preds[te_i] = m.predict(X_scaled_const[te_i])
        
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = mean_absolute_percentage_error(y, preds) * 100
    r2 = r2_score(y, preds)
    
    return model, scaler, rmse, mape, r2, preds

# ----------------- MAIN APP LOGIC -----------------
st.markdown("<div class='report-title'>KKTC GSYİH TAHMİN MERKEZİ</div>", unsafe_allow_html=True)
st.markdown("<div class='report-subtitle'>Gelişmiş ARDL Modeli ve Dinamik Deflatör Analizi</div>", unsafe_allow_html=True)

# Veriyi Yükle
df_raw = load_data()

# 1. Deflatör Tahmini Hesabı
d_reg, d_simp, d_avg, pred_d_growth = estimate_deflator(df_raw)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Flag_of_the_Turkish_Republic_of_Northern_Cyprus.svg/800px-Flag_of_the_Turkish_Republic_of_Northern_Cyprus.svg.png", width=150)
    st.header("⚙️ Model Ayarları")
    
    use_custom_defl = st.toggle("Manuel Deflatör Belirle", value=False)
    if use_custom_defl:
        chosen_deflator = st.number_input("2025 Deflatörü:", min_value=1_000_000, value=int(d_reg), step=100000)
    else:
        chosen_deflator = d_reg
        st.info(f"Otomatik (Regresyon) Deflatör:\n\n**{d_reg:,.0f}**")
        
    ci_level_pct = st.slider("Güven Aralığı Seviyesi (%)", min_value=80, max_value=99, value=95, step=1)
    ci_alpha = 1 - (ci_level_pct / 100.0)

# Veriyi Seçilen Deflatöre Göre Güncelle (2025)
df = df_raw.copy()
for col in DEFLATOR_DEPENDENT:
    if col in NOMINAL_2025_TL:
        df.loc[df['Year'] == 2025, col] = NOMINAL_2025_TL[col] / chosen_deflator
df.loc[df['Year'] == 2025, 'deflator'] = chosen_deflator

# 2. Veri Hazırlığı (Eğitim ve Tahmin)
features_base = ['PubSPEN77', 'DEPOSIT77', 'IMP77', 'KREDI77', 'POP', 'ElectricKwH', 'DummyCorona', 'USDTRYchg', 'CPIchg']
df_sorted = df.sort_values('Year').copy()
df_sorted['GDP77_lag1'] = df_sorted['GDP77'].shift(1)
features_ardl = features_base + ['GDP77_lag1']

# Train Set
train_mask = df_sorted['GDP77'].notna()
for f in features_ardl:
    train_mask = train_mask & df_sorted[f].notna()
df_train = df_sorted[train_mask]

# Pred Set (2025)
pred_mask = df_sorted['GDP77'].isna() & (df_sorted['Year'] == 2025)
df_pred = df_sorted[pred_mask]
df_pred['GDP77_lag1'] = df_train.sort_values('Year')['GDP77'].iloc[-1] # En son gerçek GDP

# Model Eğitimi (Statsmodels OLS)
model, scaler, loo_rmse, loo_mape, loo_r2, fit_preds = fit_ardl_model(df_train, features_ardl)

# 2025 Tahmini
X_pred_raw = df_pred[features_ardl].values
X_pred_scaled = scaler.transform(X_pred_raw)
X_pred_const = sm.add_constant(X_pred_scaled, prepend=True, has_constant='add')

# Guven Araligi (Confidence Intervals)
pred_results = model.get_prediction(X_pred_const)
pred_summary = pred_results.summary_frame(alpha=ci_alpha)

pred_val = pred_summary['mean'].values[0]
lower_ci = pred_summary['obs_ci_lower'].values[0]
upper_ci = pred_summary['obs_ci_upper'].values[0]

g2024 = df_train[df_train['Year'] == 2024]['GDP77'].values[0]
growth_pct = (pred_val - g2024) / g2024 * 100
growth_lower = (lower_ci - g2024) / g2024 * 100
growth_upper = (upper_ci - g2024) / g2024 * 100

nominal_gdp = pred_val * chosen_deflator

# ----------------- UI: KPI METRICS -----------------
st.write("---")
cols = st.columns(4)

with cols[0]:
    st.metric(label="Reel GSYİH Tahmini (GDP77)", 
              value=f"{pred_val:,.1f}", 
              delta=f"{growth_pct:+.2f}% Büyüme (Tahmin)")

with cols[1]:
    st.metric(label=f"Nominal GSYİH Tahmini (TL)", 
              value=f"₺ {nominal_gdp/1e9:,.1f} Milyar")

with cols[2]:
    st.metric(label="Uygulanan Deflatör", 
              value=f"{chosen_deflator:,.0f}",
              delta="Manuel" if use_custom_defl else f"{pred_d_growth:+.2f}% (Regresyon)")

with cols[3]:
    st.metric(label=f"Güven Aralığı ({ci_level_pct}%)", 
              value=f"[{growth_lower:+.1f}% , {growth_upper:+.1f}%]")

st.write("---")

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Görselleştirme", 
    "📊 Deflatör Hassasiyet Analizi", 
    "🔬 Model Parametreleri", 
    "📈 LOO-CV Performansı"
])

with tab1:
    st.subheader("GSYİH Gelişimi ve Tahmin Güven Aralığı")
    
    fig = go.Figure()

    # Gerçek Değerler
    fig.add_trace(go.Scatter(
        x=df_train['Year'], y=df_train['GDP77'],
        mode='lines+markers', name='Gerçekleşen GDP77',
        line=dict(color='#00d2ff', width=3),
        marker=dict(size=6)
    ))
    
    # Model Fit (Geçmişe Yönelik)
    fig.add_trace(go.Scatter(
        x=df_train['Year'], y=fit_preds,
        mode='lines', name='ARDL Model Uyum',
        line=dict(color='#FAFAFA', width=1, dash='dash')
    ))
    
    # Tahmin Noktası
    fig.add_trace(go.Scatter(
        x=[2024, 2025], y=[g2024, pred_val],
        mode='lines+markers', name='Tahmin Yolu',
        line=dict(color='#ff4b4b', width=3, dash='dot'),
        marker=dict(size=10, symbol='star', color='#f0b323')
    ))
    
    # Ribbon (Tahmin Aralığı - Hedeften önceki yıl sabit olacak şekilde konik)
    fig.add_trace(go.Scatter(
        x=[2024, 2025, 2025, 2024],
        y=[g2024, upper_ci, lower_ci, g2024],
        fill='toself',
        fillcolor='rgba(255, 75, 75, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name=f'{ci_level_pct}% Güven Aralığı'
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=f"Tahmini {ci_level_pct}% Güven Aralığı: {lower_ci:,.1f} - {upper_ci:,.1f}",
        xaxis_title="Yıl",
        yaxis_title="Reel GSYİH (1977 Bazlı)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Deflatör Hassasiyet (Duyarlılık) Tablosu")
    st.markdown("""
    Deflatör seviyesinin % ±20 aralığında değişmesi durumunda, **nominal → reel dönüşümü** 
    ve ardından ARDL tahmini nasıl değişiyor?
    """)
    
    pct_changes = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    sens_data = []
    
    for pct in pct_changes:
        new_d = chosen_deflator * (1 + pct / 100)
        df_mod = df_raw.copy()
        for col in DEFLATOR_DEPENDENT:
            if col in NOMINAL_2025_TL:
                df_mod.loc[df_mod['Year'] == 2025, col] = NOMINAL_2025_TL[col] / new_d
        
        pr_mod = df_mod[df_mod['Year'] == 2025]
        X_p_mod = pr_mod[features_base].copy()
        X_p_mod['GDP77_lag1'] = g2024
        
        # Ensure scaling uses the correct features in correct order
        X_p_scaled = scaler.transform(X_p_mod[features_ardl].values)
        X_p_const = sm.add_constant(X_p_scaled, prepend=True, has_constant='add')
        
        mod_pred = model.get_prediction(X_p_const).summary_frame(alpha=ci_alpha)
        m_val = mod_pred['mean'].values[0]
        m_gr = (m_val - g2024) / g2024 * 100
        
        label = "MEVCUT (BAZ)" if pct == 0 else f"{pct:+d}%"
        sens_data.append({
            "Senaryo": label,
            "Deflatör Değeri": new_d,
            "PubSPEN77": pr_mod['PubSPEN77'].values[0],
            "IMP77": pr_mod['IMP77'].values[0],
            "Tahmini Reel GDP77": m_val,
            "Tahmini Büyüme (%)": m_gr
        })
        
    sens_df = pd.DataFrame(sens_data)
    
    st.dataframe(
        sens_df.style
        .format({
            'Deflatör Değeri': '{:,.0f}',
            'PubSPEN77': '{:,.1f}',
            'IMP77': '{:,.1f}',
            'Tahmini Reel GDP77': '{:,.1f}',
            'Tahmini Büyüme (%)': '{:+.2f}%'
        })
        .background_gradient(subset=['Tahmini Büyüme (%)'], cmap='RdYlGn')
        .apply(lambda x: ['background-color: #333333; color: white' if x['Senaryo'] == 'MEVCUT (BAZ)' else '' for i in x], axis=1),
        use_container_width=True, height=350
    )

with tab3:
    st.subheader("İstatistiksel Çıktılar ve Ölçeklenmiş Katsayılar")
    st.markdown("**(StandardScaler sonrası katsayılar / OLS Özeti)**")
    
    coef_df = pd.DataFrame({
        'Değişken': ['Constant'] + features_ardl,
        'Katsayı': model.params,
        'P-Değeri': model.pvalues,
        'T-İstatistiği': model.tvalues
    })
    
    def highlight_pval(s):
        is_sig = s < 0.05
        return ['color: #00FF00; font-weight: bold' if v else 'color: #FF5555' for v in is_sig]
        
    st.dataframe(
        coef_df.style.format({
            'Katsayı': '{:.4f}',
            'P-Değeri': '{:.4f}',
            'T-İstatistiği': '{:.4f}'
        }).apply(highlight_pval, subset=['P-Değeri']),
        use_container_width=True
    )
    
    st.text(model.summary().as_text())

with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Model Kararlılığı")
        st.markdown(f"""
        **LOO-CV (Leave-One-Out) Sonuçları:**
        - **Ortalama Karekök Hata (RMSE):** {loo_rmse:,.2f}
        - **Ortalama Mutlak Yüzde Hata (MAPE):** %{loo_mape:.2f}
        - **Açıklayıcılık Oranı (R²):** {loo_r2:.4f}
        """)
        st.info("Bu model tarihsel verilerle eğitilirken her bir yılı sırayla dışarıda bırakıp test etmiş ve %" + f"{loo_mape:.2f}" + " ortalama sapma yakalamıştır.")
    
    with col_b:
        st.subheader("Tarihsel Hata Dağılımı")
        residuals = df_train['GDP77'].values - fit_preds
        fig2 = go.Figure(data=[go.Bar(
            x=df_train['Year'],
            y=residuals,
            marker_color=['#FF4B4B' if val < 0 else '#4BFF4B' for val in residuals]
        )])
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title="Sınıflandırma Artıkları (Gerçek - Tahmin)",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

