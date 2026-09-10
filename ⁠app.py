import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

st.title("🚀 BIST 30 Akşam Tahmin Paneli")
st.write("Yarın en yüksek yükseliş potansiyeli olan hisseler taranıyor...")

bist30 = [
    "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", 
    "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", 
    "GUBRF.IS", "HEKTS.IS", "KCHOL.IS", "KONTR.IS", "KOZAL.IS", 
    "KRDMD.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", 
    "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", 
    "TOASO.IS", "TUPRS.IS", "ULKER.IS", "YKBNK.IS", "YLTEK.IS"
]

if st.button("Taramayı Başlat"):
    sonuclar = []
    bar = st.progress(0)
    
    for i, hisse in enumerate(bist30):
        try:
            df = yf.download(hisse, period="2y", interval="1d", progress=False)
            if len(df) >= 100:
                df['SMA_10'] = df['Close'].rolling(10).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                df['RSI'] = 100 - (100 / (1 + (gain / loss)))
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
                df.dropna(inplace=True)
                
                features = ['Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI']
                X, y = df[features], df['Target']
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X[:-1], y[:-1])
                
                prob = model.predict_proba(X.iloc[[-1]])[0][1] * 100
                sonuclar.append({
                    'Hisse': hisse.replace('.IS', ''),
                    'Fiyat (TL)': round(float(df['Close'].iloc[-1]), 2),
                    'RSI': round(float(df['RSI'].iloc[-1]), 2),
                    'Yükseliş İhtimali (%)': round(prob, 1)
                })
        except:
            pass
        bar.progress((i + 1) / len(bist30))
        
    res_df = pd.DataFrame(sonuclar).sort_values(by='Yükseliş İhtimali (%)', ascending=False)
    st.dataframe(res_df.head(5), use_container_width=True)
