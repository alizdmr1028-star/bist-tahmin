import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# ==========================================
# 1. SAYFA VE BİLDİRİM YAPILANDIRMASI
# ==========================================
st.set_page_config(page_title="BIST Trading & Tavan Hunter Terminal", layout="wide")

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_signal(message):
    if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN":
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            st.warning(f"Telegram bildirimi gönderilemedi: {e}")

# ==========================================
# 2. VERİ ÇEKME VE İNDİKATÖR HESAPLAMA
# ==========================================
@st.cache_data(ttl=60)
def fetch_data(ticker, period="5d", interval="15m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty or len(df) < 14:
        return pd.DataFrame()

    # ATR Hesaplaması
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
    
    # VWAP Hesaplaması
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / (v.cumsum() + 1e-9)
    
    # RSI Hesaplaması
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Hacim Ortalama Karşılaştırması
    df['Vol_SMA'] = df['Volume'].rolling(10).mean()

    return df.dropna()

st.title("⚡ BIST Algo-Trading & Tavan Hunter Terminal")

# SEKMELİ YAPI (TABS)
tab1, tab2 = st.tabs(["🚀 Tavan Hunter & Hisse Taraması", "📊 Scalp & Detaylı Analiz"])

# ==========================================
# TAB 1: TAVAN POTANSİYELİ VE GENEL TARAMA
# ==========================================
with tab1:
    st.subheader("🔎 BIST Tavan Potansiyeli ve Momentum Taraması")
    watch_list = st.text_area("Taranacak Hisseler (Virgülle Ayırın)", "THYAO.IS, GARAN.IS, EREGL.IS, AKBNK.IS, SISE.IS, TUPRS.IS, SASA.IS, KCHOL.IS")
    tickers = [t.strip() for t in watch_list.split(",")]
    
    if st.button("Taramayı Başlat"):
        results = []
        progress_bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            try:
                data = fetch_data(t, period="5d", interval="15m")
                if not data.empty:
                    last_c = data['Close'].iloc[-1]
                    last_vwap = data['VWAP'].iloc[-1]
                    last_rsi = data['RSI'].iloc[-1]
                    last_vol = data['Volume'].iloc[-1]
                    avg_vol = data['Vol_SMA'].iloc[-1]
                    
                    # Tavan İhtimali Algoritması Kriterleri
                    vol_spike = last_vol > (avg_vol * 2.0)  # Hacim son 10 bar ortalamasının 2 katına çıkmış mı?
                    price_above_vwap = last_c > last_vwap   # Fiyat VWAP üstünde mi?
                    rsi_strong = last_rsi > 65              # RSI ivmeli boğa bölgesinde mi?

                    signal = "NÖTR"
                    if vol_spike and price_above_vwap and rsi_strong:
                        signal = "🔥 TAVAN ADAYI / YÜKSEK İVME"
                    elif price_above_vwap and last_rsi > 50:
                        signal = "🟢 GÜÇLÜ BOĞA"
                    elif last_c < last_vwap and last_rsi < 40:
                        signal = "🔴 SAT / ZAYIF"

                    results.append({
                        "Hisse": t,
                        "Son Fiyat": round(last_c, 2),
                        "VWAP": round(last_vwap, 2),
                        "RSI (14)": round(last_rsi, 1),
                        "Hacim Patlaması": "EVET" if vol_spike else "HAYIR",
                        "Sinyal": signal
                    })
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(tickers))
        
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

# ==========================================
# TAB 2: DETAYLI SCALP & EXECUTION
# ==========================================
with tab2:
    symbol = st.text_input("Detaylı Analiz Edilecek Hisse", value="THYAO.IS")
    interval = st.selectbox("Zaman Dilimi", ["1m", "5m", "15m"], index=2)
    atr_mult = st.slider("ATR Stop Katsayısı", 1.0, 3.0, 1.5)

    df_single = fetch_data(symbol, period="5d", interval=interval)

    if not df_single.empty:
        last_price = float(df_single['Close'].iloc[-1])
        last_atr = float(df_single['ATR'].iloc[-1])
        last_vwap = float(df_single['VWAP'].iloc[-1])

        atr_stop = last_price - (last_atr * atr_mult)
        tp1 = last_price + (last_atr * atr_mult * 1.5)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Anlık Fiyat", f"{last_price:.2f} TL")
        col2.metric("VWAP", f"{last_vwap:.2f} TL")
        col3.metric("ATR Stop", f"{atr_stop:.2f} TL")
        col4.metric("TP1 Hedef", f"{tp1:.2f} TL")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_single.index, open=df_single['Open'], high=df_single['High'], low=df_single['Low'], close=df_single['Close'], name="Fiyat"))
        fig.add_trace(go.Scatter(x=df_single.index, y=df_single['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
        fig.update_layout(title=f"{symbol} Grafik", xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("📱 Telegram'a Sinyal Gönder"):
            msg = f"🚀 *SCALP SİNYALİ: {symbol}*\n\n• Fiyat: {last_price:.2f} TL\n• VWAP: {last_vwap:.2f} TL\n• ATR Stop: {atr_stop:.2f} TL\n• TP1: {tp1:.2f} TL"
            send_telegram_signal(msg)
            st.success("Sinyal gönderildi!")
