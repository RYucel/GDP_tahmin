# KKTC GSYIH Tahmin Sistemi — Walkthrough

> **Proje Klasörü:** `e:\DPO\Tahmin\data\GDP\GDP_Tahmin_Kod`
> **Son Güncelleme:** 8 Nisan 2026

---

## 1. Proje Özeti

Bu proje, **Kuzey Kıbrıs Türk Cumhuriyeti (KKTC) Gayri Safi Yurt İçi Hasılası'nı (GSYIH)** makroekonomik değişkenler kullanarak tahmin eden, Python tabanlı bir istatistiksel/makine öğrenmesi sistemidir.

**Temel Hedefler:**
- 2025 yılı **reel GSYIH (1977 bazlı)** tahminini üretmek
- Birden fazla modeli karşılaştırarak en güvenilir tahmini seçmek
- Deflator belirsizliğini hassasiyet analizi ile yönetmek
- 2024 gerçek verisiyle back-test (validasyon) yaparak model güvenilirliğini doğrulamak

---

## 2. Dosya Yapısı

```
GDP_Tahmin_Kod/
├── GDP_veriler.csv        ← Ana veri seti (1977–2027)
├── gdp_forecast.py        ← Ana tahmin betiği (v3)
├── validate_2024.py       ← 2024 back-test / validasyon betiği
├── check_nominals.py      ← Nominal-reel dönüşüm kontrol betiği
└── walkthrough.md         ← Bu döküman
```

---

## 3. Veri Seti: `GDP_veriler.csv`

### 3.1 Genel Bilgi

| Özellik       | Değer                          |
|---------------|--------------------------------|
| Dönem         | 1977 – 2027                    |
| Satır sayısı  | 53 (yıllık)                    |
| Hedef değişken| `GDP77` (Reel GSYIH, 1977=100) |
| Tahmin yılı   | 2025 (GDP77 boş)               |

### 3.2 Değişkenler

| Değişken       | Açıklama                                    | Birim / Not                        |
|----------------|---------------------------------------------|------------------------------------|
| `Year`         | Yıl                                        |                                    |
| `PubSPEN77`    | Kamu harcamaları (reel, 1977 bazlı)        | Deflatöre bağlı                    |
| `DEPOSIT77`    | Bankacılık mevduatları (reel)               | Deflatöre bağlı                    |
| `IMP77`        | İthalat (reel)                              | Deflatöre bağlı                    |
| `KREDI77`      | Banka kredileri (reel)                      | Deflatöre bağlı                    |
| `POP`          | Nüfus                                      |                                    |
| `GDP77`        | Reel GSYIH (hedef değişken)                 | 1977 bazlı                        |
| `deflatör`     | GSYIH deflatörü                             | Kod içinde `deflator` olarak kullanılır |
| `ElectricKwH`  | Elektrik tüketimi                           | kWh                                |
| `DummyCorona`  | COVID-19 kukla değişkeni                    | 0 veya 1                          |
| `USDTRYchg`    | USD/TRY kur değişimi (%)                    |                                    |
| `CPIchg`       | TÜFE değişimi (%)                           |                                    |

### 3.3 Deflatöre Bağlı Değişkenler

`PubSPEN77`, `DEPOSIT77`, `IMP77` ve `KREDI77` nominal değerlerden deflatörle reele çevrilir:

```
Reel Değer = Nominal Değer / Deflatör
```

2025 yılı için nominal değerler (TL cinsinden) kod içinde sabit olarak tanımlıdır:

| Değişken     | 2025 Nominal (TL)          |
|--------------|----------------------------|
| PubSPEN77    | 129.839.075.419 TL         |
| DEPOSIT77    | 381.222.049.000 TL (bin TL → TL) |
| IMP77        | 148.220.100.000 TL (bin TL → TL) |
| KREDI77      | 197.726.563.000 TL (bin TL → TL) |

---

## 4. Ana Tahmin Betiği: `gdp_forecast.py`

### 4.1 Çalıştırma

```bash
# Varsayılan (otomatik deflator tahmini)
python gdp_forecast.py

# Manuel deflator geçersiz kılma
python gdp_forecast.py --deflator 15000000

# PubSPEN77 hariç bırakma
python gdp_forecast.py --no-pubspen
```

### 4.2 İş Akışı (Pipeline)

```
     CSV Verisi Yükle
           │
     ┌─────▼──────┐
     │ BÖLÜM 1:   │
     │ Deflator    │──── 3 yöntemle deflator tahmin et
     │ Tahmini     │     (basit formül / regresyon / 5-yıl ort.)
     └─────┬──────┘
           │
     Nominal → Reel Dönüşüm (2025 satırı)
           │
     ┌─────▼──────┐
     │ Corona     │──── 4 farklı dummy konfigürasyonu dene
     │ Dummy Opt. │     En düşük LOO-RMSE'yi seç
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ BÖLÜM 2:   │──── OLS tanı testleri
     │ Model      │     LOO-CV ile 9+ model karşılaştır
     │ Eğitimi    │     ARDL model (lag1 dahil)
     │             │     Best subset selection
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ BÖLÜM 3:   │──── ARDL ana tahmin
     │ 2025       │     Tüm model tahminlerini karşılaştır
     │ Tahmini    │     Nominal GSYIH hesapla
     └─────┬──────┘
           │
     ┌─────▼──────┐
     │ BÖLÜM 4:   │──── Deflator ±%20 hassasiyet tablosu
     │ Hassasiyet │     ARDL ile GDP tahmininin değişimi
     └─────┬──────┘
           │
        Özet Rapor
```

### 4.3 Bölüm 1: Deflator Tahmini

2025 deflatörü bilinmediğinden, üç farklı yöntemle tahmin edilir:

| Yöntem | Formül / Açıklama |
|--------|-------------------|
| **Basit formül** | `D(2025) = D(2024) × (1 + (0.4×USDTRYchg + 0.6×CPIchg) / 100)` |
| **Regresyon** | Geçmiş `defl_growth ~ USDTRYchg + CPIchg` ilişkisini OLS ile öğren |
| **5 yıl ortalaması** | Son 5 yılın deflator büyüme ortalamasını kullan |

**Varsayılan seçim:** Regresyon yöntemi (en bilimsel yaklaşım olarak değerlendirilir).

Geçmiş yıllarda formül doğruluğu MAPE ile kontrol edilir.

### 4.4 Bölüm 2: Model Eğitimi ve Karşılaştırma

#### Kullanılan Modeller

| Model | Tür | Önemli Parametreler |
|-------|-----|---------------------|
| OLS | Doğrusal regresyon | — |
| Ridge | L2 düzenlileştirme | α = 1.0 |
| Lasso | L1 düzenlileştirme | α = 0.5 |
| ElasticNet | L1+L2 | α = 0.5, l1_ratio = 0.5 |
| Random Forest | Topluluk (ağaç) | 200 ağaç, derinlik=5 |
| Gradient Boosting | Topluluk (boost) | 150 ağaç, lr=0.05 |
| SVR | Destek vektör | RBF çekirdek, C=1000 |
| XGBoost | Gradient boost | 150 ağaç, lr=0.05 |
| LightGBM | Gradient boost | 150 ağaç, lr=0.05 |
| **ARDL** | Otoregresif (lag dahil) | GDP77_lag1 eklenir |

#### Değerlendirme Yöntemleri

1. **Leave-One-Out Cross-Validation (LOO-CV):** Her gözlem sırayla test edilir
2. **Time Series Split (TS-CV):** Zaman serisi yapısına uygun çapraz doğrulama
3. **Metrikler:** RMSE, MAE, MAPE, R²

#### DummyCorona Optimizasyonu

4 farklı corona dummy konfigürasyonu denenir:

| Konfigürasyon | Yıllar |
|---------------|--------|
| 2020only      | 2020 |
| 2020-2021     | 2020, 2021 |
| 2020-2022     | 2020, 2021, 2022 |
| 2020+2022     | 2020, 2022 |

En düşük LOO-RMSE'yi veren konfigürasyon seçilir.

#### OLS Tanı Testleri

- **Breusch-Godfrey:** Otokorelasyon testi
- **Breusch-Pagan:** Heteroskedastisite testi
- **Jarque-Bera:** Normallik testi
- **Durbin-Watson:** Otokorelasyon istatistiği

#### Best Subset Selection

Tüm olası değişken alt kümeleri (2 ila 6 değişken) LOO-CV ile denenerek en düşük RMSE veren kombinasyon rapor edilir.

### 4.5 Bölüm 3: 2025 GDP Tahmini

- **Ana model: ARDL** (geçmiş GDP'nin lag'ını kullanarak en düşük MAPE'yi sağlar)
- Tüm modellerin tahminleri karşılaştırılır
- Nominal GSYIH hesaplanır: `Nominal = Reel × Deflatör`

### 4.6 Bölüm 4: Deflator Hassasiyet Analizi

Seçilen deflatörün ±%20 aralığında değiştirilerek GDP tahmininin nasıl etkilendiği test edilir. Bu, nominal→reel dönüşüm belirsizliğinin boyutunu gösterir.

```
Deflator %     Deflator    PubSPEN77   DEPOSIT77    IMP77    KREDI77  ARDL GDP77  Büyüme%
------------ -----------  ---------- ---------- --------- ---------- ---------- --------
   -20%       11,478,283    ...        ...        ...       ...       ...        ...
   MEVCUT     14,347,854    ...        ...        ...       ...       ...        ...
   +20%       17,217,424    ...        ...        ...       ...       ...        ...
```

---

## 5. Validasyon Betiği: `validate_2024.py`

### 5.1 Amaç

2024 yılının gerçek GDP77 değeri bilindiğine göre, **1982–2023 ile eğitim yapıp 2024'ü tahmin ederek** modellerin gerçek performansını ölçmek.

### 5.2 Çalıştırma

```bash
python validate_2024.py
```

### 5.3 İş Akışı

1. Veri yükleme ve ön işleme
2. DummyCorona = 1 (2020, 2021)
3. 1982–2023 → eğitim, 2024 → test
4. Tüm modeller + ARDL eğitilir ve tahmin yapılır
5. LOO-RMSE (eğitim setinde) hesaplanır
6. 2024 tahmin hatası (% cinsinden) raporlanır
7. **Ensemble:** En iyi 5 modelin ters-RMSE ağırlıklı ortalaması

### 5.4 Beklenen Sonuçlar

- **ARDL:** ~%0.94 hata ile en iyi tek model
- Gerçek 2024 GDP77: **21.310,57**
- Modeller hata yüzdesine göre sıralanarak raporlanır

---

## 6. Kontrol Betiği: `check_nominals.py`

### 6.1 Amaç

CSV'deki reel değerlerle manuel olarak girilen nominal değerler arasındaki tutarlılığı kontrol etmek.

### 6.2 Çalıştırma

```bash
python check_nominals.py
```

### 6.3 Kontroller

1. **CSV reel × deflatör = nominal?** → Tutarlılık kontrolü
2. **Nominal / CSV reel = implied deflatör?** → Tüm değişkenler aynı deflatörü mü ima ediyor
3. **Binlik çarpan kontrolü** → Birim uyumsuzluğu tespiti (bin TL vs TL)

---

## 7. Bağımlılıklar

```
pandas
numpy
scikit-learn
statsmodels
scipy
xgboost         (opsiyonel — yoksa atlanır)
lightgbm        (opsiyonel — yoksa atlanır)
```

### Kurulum

```bash
pip install pandas numpy scikit-learn statsmodels scipy xgboost lightgbm
```

---

## 8. Metodoloji Detayları

### 8.1 Neden ARDL Ana Model?

| Kriter | ARDL Avantajı |
|--------|---------------|
| **Otoregresif yapı** | GDP'nin geçmiş değerini (lag1) içerir; ekonomik serilerde güçlü süreklilik yakalanır |
| **2024 validasyonu** | Back-test'te en düşük hata (~%0.94) |
| **LOO-CV MAPE** | Diğer modellere kıyasla tutarlı düşük hata |
| **Yorumlanabilirlik** | Doğrusal model, katsayılar yorumlanabilir |

### 8.2 Deflator Belirsizliği

2025 deflatörü henüz açıklanmadığından, nominal değerlerin reele çevrilmesi belirsizdir. Bu risk:

1. **Üç farklı tahmin yöntemi** ile azaltılır
2. **±%20 hassasiyet analizi** ile ölçülür
3. `--deflator` parametresiyle **kullanıcı müdahalesi** mümkündür

### 8.3 Ölçekleme (Scaling)

Tüm değişkenler `StandardScaler` ile standartlaştırılır (`mean=0`, `std=1`). Bu:
- Ridge/Lasso/SVR gibi modeller için zorunludur
- Olası birim çelişkilerini (nüfus vs kWh vs kur) ortadan kaldırır

---

## 9. Tipik Kullanım Senaryoları

### Senaryo 1: Standart Tahmin
```bash
cd e:\DPO\Tahmin\data\GDP\GDP_Tahmin_Kod
python gdp_forecast.py
```
Çıktı: Deflator tahmini → Model karşılaştırma → ARDL GDP tahmini → Hassasiyet analizi

### Senaryo 2: Farklı Deflator ile Tahmin
```bash
python gdp_forecast.py --deflator 16000000
```
Kullanıcı belirlediği deflator ile tüm analizlerin tekrarlanması.

### Senaryo 3: PubSPEN77 Hariç
```bash
python gdp_forecast.py --no-pubspen
```
Kamu harcamaları verisi belirsiz ise bu değişken çıkarılarak analiz.

### Senaryo 4: Model Güvenilirliği Kontrolü
```bash
python validate_2024.py
```
2024 gerçek değeriyle geriye dönük test.

### Senaryo 5: Veri Tutarlılık Kontrolü
```bash
python check_nominals.py
```
Nominal-reel dönüşüm tutarlılığının doğrulanması.

---

## 10. Veri Güncelleme Rehberi

Yeni yıl verileri geldiğinde yapılması gerekenler:

1. **`GDP_veriler.csv`** dosyasına yeni satır ekle veya mevcut satırı güncelle
2. Nominal değerler güncellendiyse `gdp_forecast.py` içindeki `NOMINAL_2025_TL` sözlüğünü güncelle
3. `validate_2024.py`'daki eğitim/test yıllarını güncelle (örn: 2025 gerçek verisi gelince 2025 validasyonu ekle)
4. `python gdp_forecast.py` çalıştırarak yeni tahminleri üret

---

## 11. Bilinen Sınırlamalar

- **Küçük örneklem:** 1982–2024 arası yalnızca ~43 gözlem (bazı değişkenler için daha az)
- **Yapısal kırılmalar:** KKTC ekonomisi 2003–2005 (TL reformu), 2020 (COVID) gibi dönemlerde yapısal değişiklikler geçirmiş
- **Deflator belirsizliği:** 2025 deflatörü tahmin edildiğinden, reel değerlere bağlı sonuçlar deflator varsayımına duyarlı
- **Tek ülke modeli:** Panel veri veya bölgesel karşılaştırma yoktur
