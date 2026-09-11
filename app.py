import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# ==========================================
# 1. GENEL SAYFA YAPILANDIRMASI & TELEGRAM
# ==========================================
st.set_page_config(page_title="BIST & Forex Universal Trading Terminal", layout="wide")

TELEGRAM_BOT_TOKEN = "8863809606:AAGkMlwvTRLeg985E2a53rn1p_mud-HPhqY"
TELEGRAM_CHAT_ID = "711670989"

def send_telegram_signal(message):
    """Telegram hesabınıza anlık sinyal mesajı gönderir."""
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            st.warning(f"Telegram bildirimi gönderilemedi: {e}")

# ==========================================
# 2. VARLIK LİSTELERİ
# ==========================================
BIST_TUM_LIST = [
    "THYAO.IS", "GARAN.IS", "EREGL.IS", "AKBNK.IS", "SISE.IS", "TUPRS.IS", "SASA.IS", "KCHOL.IS",
    "YKBNK.IS", "ISCTR.IS", "BIMAS.IS", "ASELS.IS", "SAHOL.IS", "HEKTS.IS", "PETKM.IS", "ENKAI.IS",
    "DOHOL.IS", "ARCLK.IS", "TOASO.IS", "FROTO.IS", "TCELL.IS", "TTKOM.IS", "PGSUS.IS", "KONTR.IS",
    "SMRTG.IS", "ASTOR.IS", "ALARK.IS", "GUBRF.IS", "ODAS.IS", "KOZAL.IS", "KOZAA.IS", "OYAKC.IS",
    "GESAN.IS", "EUPWR.IS", "MIATK.IS", "REEDR.IS", "DOFRB.IS", "TEHOL.IS"
]

FOREX_PAIRS = {
    "Ons Altın (XAUUSD)": "GC=F",
    "Brent Petrol": "BZ=F",
    "Ham Petrol (WTI)": "CL=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "S&P 500 Index": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Bitcoin (BTCUSD)": "BTC-USD"
}

# ==========================================
# 3. VERİ MOTORU VE İNDİKATÖRLER
# ==========================================
@st.cache_data(ttl=40)
def fetch_market_data(symbol, period="5d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 14:
            return pd.DataFrame()

        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
        
        # VWAP
        v = df['Volume'].replace(0, 1)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / (v.cumsum() + 1e-9)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # EMAs
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ==========================================
# 4. BIST TAVAN SKORLAMA ALGORİTMASI
# ==========================================
def calculate_tavan_score(df):
    if df.empty or len(df) < 20:
        return None
    
    last_c = float(df['Close'].iloc[-1])
    day_open = float(df['Open'].iloc[0])
    day_high = float(df['High'].max())
    vwap = float(df['VWAP'].iloc[-1])
    rsi = float(df['RSI'].iloc[-1])
    
    tavan_fiyat = round(day_open * 1.10, 2)
    if tavan_fiyat <= last_c:
        tavan_mesafe_pct = 0.0
    else:
        tavan_mesafe_pct = ((tavan_fiyat - last_c) / last_c) * 100

    recent_vol = df['Volume'].iloc[-3:].mean()
    hist_vol = df['Volume'].mean()
    rvol = recent_vol / (hist_vol + 1e-9)

    vwap_above = last_c > vwap
    rsi_momentum = 60 <= rsi <= 82
    high_breakout = (last_c >= day_high * 0.985)
    hacim_patlamasi = rvol >= 2.0
    tavan_potansiyel_bolge = 1.5 <= tavan_mesafe_pct <= 7.5

    score = 0
    if vwap_above: score += 20
    if rsi_momentum: score += 20
    if high_breakout: score += 25
    if hacim_patlamasi: score += 25
    if tavan_potansiyel_bolge: score += 10

    status = "NÖTR"
    if score >= 80:
        status = "🔥 YÜKSEK TAVAN POTANSİYELİ"
    elif score >= 50:
        status = "⚡ GÜÇLÜ İVME / TAKİP"
    else:
        status = "⚪ ZAYIF"

    return {
        "Son Fiyat": round(last_c, 2),
        "Tavan Hedef": tavan_fiyat,
        "Tavana Kalan (%)": round(tavan_mesafe_pct, 2),
        "RVOL (Hacim Katı)": round(rvol, 1),
        "RSI": round(rsi, 1),
        "Tavan Skoru (%)": score,
        "Durum": status
    }

# ==========================================
# 5. SOL ANA MENÜ (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Piyasa Seçimi")
selected_market = st.sidebar.radio(
    "Terminal Modu Seçin:",
    [
        "🇹🇷 BIST Hisseleri Terminali", 
        "🌍 Forex / Emtia / Kripto Terminali", 
        "🧮 Risk & Lot Hesaplayıcı"
    ]
)

# ==========================================
# MODÜL 1: BIST HİSSE TERMİNALİ
# ==========================================
if selected_market == "🇹🇷 BIST Hisseleri Terminali":
    st.title("🇹🇷 BIST Scalp & Tavan İhtimali Terminali")
    tab1, tab2 = st.tabs(["🚀 Gelişmiş Tavan Taraması", "📊 Detaylı Hisse Grafiği"])

    with tab1:
        st.subheader("🎯 Tavan Skorlama & Otomatik Bildirim Motoru")

        if st.button("🚀 BIST Tavan Taramasını Başlat"):
            results = []
            telegram_alerts = []
            bar = st.progress(0)

            for i, symbol in enumerate(BIST_TUM_LIST):
                data = fetch_market_data(symbol, period="2d", interval="5m")
                if not data.empty:
                    analysis = calculate_tavan_score(data)
                    if analysis:
                        hisse_kodu = symbol.replace(".IS", "")
                        analysis["Hisse"] = hisse_kodu
                        results.append(analysis)

                        if analysis["Tavan Skoru (%)"] >= 80:
                            msg = (
                                f"🔥 *BIST TAVAN SİNYALİ: #{hisse_kodu}*\n\n"
                                f"• **Tavan Skoru:** %{analysis['Tavan Skoru (%)']}\n"
                                f"• **Son Fiyat:** {analysis['Son Fiyat']} TL\n"
                                f"• **Tavan Hedef:** {analysis['Tavan Hedef']} TL\n"
                                f"• **Tavana Mesafe:** %{analysis['Tavana Kalan (%)']}\n"
                                f"• **Hacim Patlaması (RVOL):** {analysis['RVOL (Hacim Katı)']}x\n"
                                f"• **RSI (14):** {analysis['RSI']}\n\n"
                                f"⚡ *Açıklama:* Fiyat günün zirvesinde ve hacimli alım baskısı mevcut!"
                            )
                            telegram_alerts.append(msg)

                bar.progress((i + 1) / len(BIST_TUM_LIST))

            if results:
                df_res = pd.DataFrame(results)
                df_res = df_res[["Hisse", "Son Fiyat", "Tavan Hedef", "Tavana Kalan (%)", "RVOL (Hacim Katı)", "RSI", "Tavan Skoru (%)", "Durum"]]
                df_res = df_res.sort_values(by="Tavan Skoru (%)", ascending=False)
                
                st.dataframe(df_res, use_container_width=True)

                if telegram_alerts:
                    for alert_msg in telegram_alerts:
                        send_telegram_signal(alert_msg)
                    st.success(f"📱 Tavan Skoru %80 üzeri olan {len(telegram_alerts)} hisse Telegram'a gönderildi!")
                else:
                    st.info("ℹ️ Taramada %80 tavan skorunu aşan hisse bulunamadı.")

    with tab2:
        symbol = st.selectbox("Analiz Etmek İstediğiniz Hisse", BIST_TUM_LIST)
        interval = st.selectbox("Zaman Dilimi", ["1m", "5m", "15m"], index=1)
        df = fetch_market_data(symbol, period="5d", interval=interval)

        if not df.empty:
            c = float(df['Close'].iloc[-1])
            vwap = float(df['VWAP'].iloc[-1])
            atr = float(df['ATR'].iloc[-1])

            col1, col2, col3 = st.columns(3)
            col1.metric("Son Fiyat", f"{c:.2f} TL")
            col2.metric("VWAP", f"{vwap:.2f} TL")
            col3.metric("ATR Stop Riski", f"{c - (atr * 1.5):.2f} TL")

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
            fig.update_layout(title=f"{symbol} ({interval}) Grafiği", xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODÜL 2: FOREX & EMTİA TERMİNALİ
# ==========================================
elif selected_market == "🌍 Forex / Emtia / Kripto Terminali":
    st.title("🌍 Forex, Emtia & Kripto Terminali")
    tab1, tab2 = st.tabs(["🎯 Çift Yönlü Forex Taraması", "📊 Canlı Grafik & Trendler"])

    with tab1:
        st.subheader("Majör Pariteler Anlık Scalp Taraması")
        if st.button("🌍 Forex Piyasasını Tara"):
            f_results = []
            f_alerts = []
            bar = st.progress(0)

            for i, (name, symbol) in enumerate(FOREX_PAIRS.items()):
                data = fetch_market_data(symbol, period="2d", interval="5m")
                if not data.empty:
                    c = data['Close'].iloc[-1]
                    vwap = data['VWAP'].iloc[-1]
                    rsi = data['RSI'].iloc[-1]
                    ema20 = data['EMA20'].iloc[-1]
                    ema50 = data['EMA50'].iloc[-1]

                    long_setup = (c > vwap) and (ema20 > ema50) and (rsi > 55)
                    short_setup = (c < vwap) and (ema20 < ema50) and (rsi < 45)

                    signal = "NÖTR"
                    if long_setup:
                        signal = "🟢 LONG (ALIM)"
                        f_alerts.append(f"🟢 *FOREX LONG SİNYALİ: {name}*\n• **Fiyat:** {round(c, 4)}\n• **RSI:** {round(rsi, 1)}")
                    elif short_setup:
                        signal = "🔴 SHORT (SATIŞ)"
                        f_alerts.append(f"🔴 *FOREX SHORT SİNYALİ: {name}*\n• **Fiyat:** {round(c, 4)}\n• **RSI:** {round(rsi, 1)}")

                    f_results.append({
                        "Varlık": name,
                        "Fiyat": round(c, 4),
                        "VWAP": round(vwap, 4),
                        "RSI (14)": round(rsi, 1),
                        "Sinyal": signal
                    })
                bar.progress((i + 1) / len(FOREX_PAIRS))

            st.dataframe(pd.DataFrame(f_results), use_container_width=True)

            if f_alerts:
                for alert in f_alerts:
                    send_telegram_signal(alert)
                st.success(f"📱 {len(f_alerts)} adet Forex sinyali Telegram'a gönderildi!")

    with tab2:
        selected_pair = st.selectbox("Parite Seçin", list(FOREX_PAIRS.keys()))
        p_symbol = FOREX_PAIRS[selected_pair]
        interval = st.selectbox("Periyot", ["1m", "5m", "15m", "1h"], index=1)
        
        df_f = fetch_market_data(p_symbol, period="2d", interval=interval)
        if not df_f.empty:
            c = float(df_f['Close'].iloc[-1])
            vwap = float(df_f['VWAP'].iloc[-1])

            st.metric("Anlık Fiyat", f"{c:.4f}")

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_f.index, open=df_f['Open'], high=df_f['High'], low=df_f['Low'], close=df_f['Close'], name="Fiyat"))
            fig.add_trace(go.Scatter(x=df_f.index, y=df_f['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
            fig.add_trace(go.Scatter(x=df_f.index, y=df_f['EMA20'], mode='lines', name='EMA20', line=dict(color='cyan', width=1)))
            fig.add_trace(go.Scatter(x=df_f.index, y=df_f['EMA50'], mode='lines', name='EMA50', line=dict(color='magenta', width=1)))
            fig.update_layout(title=f"{selected_pair} ({interval}) Grafiği", xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODÜL 3: RİSK & LOT HESAPLAYICI
# ==========================================
elif selected_market == "🧮 Risk & Lot Hesaplayıcı":
    st.title("🧮 Pozisyon & Lot Büyüklüğü Hesaplayıcı")

    c1, c2, c3 = st.columns(3)
    balance = c1.number_input("Toplam Bakiye ($ veya TL)", value=5000.0, step=500.0)
    risk_pct = c2.number_input("İşlem Başı Risk (%)", value=1.0, step=0.5)
    stop_dist = c3.number_input("Stop Mesafesi (Pip veya TL)", value=15.0, step=1.0)

    if st.button("Hesapla"):
        max_risk = balance * (risk_pct / 100)
        forex_lot = max_risk / (stop_dist * 10)
        bist_adet = max_risk / stop_dist if stop_dist > 0 else 0

        st.success(f"💵 **İşlem Başı Göze Alınan Maksimum Zarar:** {max_risk:.2f}")
        st.info(f"📊 **Forex için Önerilen Pozisyon:** {forex_lot:.2f} Lot")
        st.info(f"📈 **BIST için Önerilen Lot (Adet):** {int(bist_adet)} Adet Hisse")
