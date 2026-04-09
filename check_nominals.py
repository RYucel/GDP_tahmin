# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv('GDP_veriler.csv')
df.columns = df.columns.str.strip()
r = df[df['Year']==2025].iloc[0]
d2025 = 14347853.67

# Kullanicinin verdigi nominaller (bin TL)
noms = {
    'PubSPEN77': 129839075419,
    'DEPOSIT77': 381222049,
    'KREDI77': 197726563,
    'IMP77': 148220100,
}

print("=== CSV reel * deflator = nominal ===")
for col in noms:
    csv_nom = r[col] * d2025
    print(f"  {col}: reel={r[col]:.2f} * defl={d2025:.2f} = {csv_nom:,.0f}")

print("\n=== Kullanici nominal / CSV reel = implied deflator ===")
for col, nom in noms.items():
    impl = nom / r[col]
    print(f"  {col}: {nom:,} / {r[col]:.2f} = {impl:,.2f}")

# Binlik carpan kontrolu
print("\n=== Nominal * 1000 / reel = implied deflator ===")
for col, nom in noms.items():
    impl = nom * 1000 / r[col]
    print(f"  {col}: {nom*1000:,} / {r[col]:.2f} = {impl:,.2f}")
