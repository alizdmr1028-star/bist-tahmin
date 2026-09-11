import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="BIST 30 Tahmin", layout="wide")
st.title("📈 BIST 30 Yapay Zeka Tahmin Modeli")

bist30_hisseler = [
    "AKBNK",
    "ALARK",
    "ASELS",
    "BIMAS",
    "BRSAN",
    "DOAS",
    "EKGYO",
    "ENKAI",
    "EREGL",
    "FROTO",
    "GARAN",
    "GUBRF",
    "HEKTS",
    "ISCTR",
    "KCHOL",
    "KONTR",
    "KOZAL",
    "KRDMD",
    "MGROS",
    "ODAS",
    "OYAKC",
    "PETKM",
    "PGSUS",
    "SAHOL",
    "SASA",
    "SISE",
    "TCELL",
    "THYAO",
    "TOASO",
    "TUPRS",
]

if st.button("🚀 Taramayı Başlat"):
    st.info("BIST 30 verileri çekiliyor ve model eğitiliyor...")
    sonuclar = []

    progress_bar = st.progress(0)

    for idx, hisse in enumerate(bist30_hisseler):
        try:
            df = yf.download(
                f"{hisse}.IS", period="1y", interval="1d", progress=False
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) > 50:
                df["Returns"] = df["Close"].pct_change()
                df["SMA_10"] = df["Close"].rolling(10).mean()
                df["SMA_30"] = df["Close"].rolling(30).mean()
                df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
                df = df.dropna()

                X = df[["Returns", "SMA_10", "SMA_30"]]
                y = df["Target"]

                if len(X) > 20:
                    model = RandomForestClassifier(
                        n_estimators=100, random_state=42
                    )
                    model.fit(X[:-1], y[:-1])

                    son_veri = X.iloc[[-1]]
                    tahmin = model.predict(son_veri)[0]
                    olasilik = model.predict_proba(son_veri)[0][1]
                    son_fiyat = float(df["Close"].iloc[-1])

                    if tahmin == 1:
                        sonuclar.append({
                            "Hisse": hisse,
                            "Son Fiyat (TL)": round(son_fiyat, 2),
                            "Yükseliş Olasılığı (%)": round(olasilik * 100, 1),
                        })
        except Exception:
            pass

        progress_bar.progress((idx + 1) / len(bist30_hisseler))

    if sonuclar:
        st.success("Tarama Tamamlandı!")
        res_df = pd.DataFrame(sonuclar).sort_values(
            by="Yükseliş Olasılığı (%)", ascending=False
        )
        st.dataframe(res_df, use_container_width=True)
    else:
        st.warning(
            "Yükseliş sinyali veren hisse bulunamadı veya veri çekilemedi."
        )
