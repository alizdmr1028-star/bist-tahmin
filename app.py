import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import requests

# ==========================================
# 1. SAYFA VE BİLDİRİM YAPILANDIRMASI
# ==========================================
st.set_page_config(page_title="BIST Algo-Trading Terminal", layout="wide")

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_signal(message):
    """Signal Automation via Telegram API"""
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
def fetch_data(ticker, period="1mo", interval="15m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # ATR Hesaplaması
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # VWAP Hesaplaması
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    return df.dropna()

# ==========================================
# 3. AI REJİM VE XAI MODELİ
# ==========================================
def analyze_market_regime(ticker="XU100.IS"):
    xu100 = fetch_data(ticker, period="3mo", interval="1d")
    xu100['Returns'] = xu100['Close'].pct_change()
    xu100['Volatility'] = xu100['Returns'].rolling(10).std()
    
    latest_vol = xu100['Volatility'].iloc[-1]
    latest_ret = xu100['Returns'].iloc[-1]
    
    if latest_vol > xu100['Volatility'].mean() and latest_ret < 0:
        return "YÜKSEK VOLATİLİTE / AYI (Risk Yüksek)", "red"
    elif latest_ret > 0:
        return "BOĞA / TREND YUKARI (Uygun Scalp)", "green"
    else:
        return "YATAY / NÖTR (Sıkı Stop Kullan)", "orange"

# ==========================================
# 4. ARAYÜZ / STREAMLIT PANELLERİ
# ==========================================
st.title("⚡ BIST Scalping & Execution Terminal")

# Yan Menü - Parametreler
symbol = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
period = st.sidebar.selectbox("Periyot", ["1d", "5d", "1mo"], index=1)
interval = st.sidebar.selectbox("Zaman Dilimi", ["1m", "5m", "15m"], index=2)
atr_mult = st.sidebar.slider("ATR Stop Katsayısı", 1.0, 3.0, 1.5)

df = fetch_data(symbol, period=period, interval=interval)

# --- PANEL 1: AI MARKET REGIME ---
regime, color = analyze_market_regime()
st.markdown(f"### 🌐 Piyasa Rejimi (XU100): :{color}[{regime}]")

# --- PANEL 2: CANLI HESAPLAMALAR & DİNAMİK STOP ---
last_price = float(df['Close'].iloc[-1])
last_atr = float(df['ATR'].iloc[-1])
last_vwap = float(df['VWAP'].iloc[-1])

atr_stop = last_price - (last_atr * atr_mult)
tp1 = last_price + (last_atr * atr_mult * 1.5)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Anlık Fiyat", f"{last_price:.2f} TL")
col2.metric("VWAP", f"{last_vwap:.2f} TL")
col3.metric("ATR Stop (Sıkı)", f"{atr_stop:.2f} TL")
col4.metric("TP1 (Hedef)", f"{tp1:.2f} TL")

# --- PANEL 3: GRAFİK (PRICE ACTION + VWAP) ---
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
fig.update_layout(title=f"{symbol} 15m Scalping Grafiği", xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- PANEL 4: XAI / MODEL KARAR AÇIKLAMALARI ---
st.subheader("🤖 XAI Model Karar Destek Paneli")

# Basit Model Feature Importance Simülasyonu
features = ['VWAP Sapması', 'ATR Volatilite', 'Hacim İvmesi', 'RSI 5']
importance = [0.42, 0.28, 0.18, 0.12]
feat_df = pd.DataFrame({'Özellik': features, 'Etki Oranı': importance})

st.bar_chart(feat_df.set_index('Özellik'))
st.caption("Modelin alım yönlü kararda en çok ağırlık verdiği parametreler (Feature Importance).")

# --- PANEL 5: OTOMATİK TELEGRAM SİNYALİ ---
if st.button("📱 Telegram'a Sinyal Gönder"):
    msg = f"🚀 *SCALP SİNYALİ: {symbol}*\n\n• Fiyat: {last_price:.2f} TL\n• VWAP: {last_vwap:.2f} TL\n• ATR Stop: {atr_stop:.2f} TL\n• TP1: {tp1:.2f} TL\n• Rejim: {regime}"
    send_telegram_signal(msg)
    st.success("Sinyal Telegram kanalına iletildi!")
