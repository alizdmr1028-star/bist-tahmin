import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# 4. GELİŞMİŞ MUM GRAFİK ÇİZİCİ FONKSİYON
# ==========================================
def render_candlestick_chart(df, title_name):
    """Fiyat Mum Grafiği + Hacim + VWAP + EMA Gösterimi"""
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.75, 0.25]
    )

    # 1. MUM GRAFİK (CANDLESTICK)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Mum Fiyat",
            increasing_line_color='#26a69a', # Yükseliş Mumu (Yeşil)
            decreasing_line_color='#ef5350'  # Düşüş Mumu (Kırmızı)
        ),
        row=1, col=1
    )

    # Indikatör Çizgileri
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    if 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20', line=dict(color='#00bcd4', width=1)), row=1, col=1)
    if 'EMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', name='EMA50', line=dict(color='#e91e63', width=1)), row=1, col=1)

    # 2. HACİM BARI (VOLUME)
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['Volume'], 
            name="Hacim", 
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    # Grafik Düzeni & Teması
    fig.update_layout(
        title=f"📈 {title_name} - Canlı Mum Grafik",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. TAVAN SKORLAMA ALGORİTMASI
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
# 6. SOL ANA MENÜ (SIDEBAR)
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
    tab1, tab2 = st.tabs(["🚀 Gelişmiş Tavan Taraması", "📊 Detaylı Mum Grafiği"])

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
        col_sel, col_per = st.columns([3, 1])
        with col_sel:
            symbol = st.selectbox("Analiz Etmek İstediğiniz Hisse", BIST_TUM_LIST)
        with col_per:
            interval = st.selectbox("Zaman Dilimi", ["1m", "5m", "15m", "1h"], index=1)

        df = fetch_market_data(symbol, period="5d", interval=interval)

        if not df.empty:
            c = float(df['Close'].iloc[-1])
            vwap = float(df['VWAP'].iloc[-1])
            atr = float(df['ATR'].iloc[-1])

            m1, m2, m3 = st.columns(3)
            m1.metric("Son Fiyat", f"{c:.2f} TL")
            m2.metric("VWAP Seviyesi", f"{vwap:.2f} TL")
            m3.metric("ATR Stop Riski (-1.5x)", f"{c - (atr * 1.5):.2f} TL")

            # MUM GRAFİK RENDER
            render_candlestick_chart(df, f"{symbol} ({interval})")

# ==========================================
# MODÜL 2: FOREX & EMTİA TERMİNALİ
# ==========================================
elif selected_market == "🌍 Forex / Emtia / Kripto Terminali":
    st.title("🌍 Forex, Emtia & Kripto Terminali")
    tab1, tab2 = st.tabs(["🎯 Çift Yönlü Forex Taraması", "📊 Canlı Mum Grafik"])

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
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            selected_pair = st.selectbox("Parite Seçin", list(FOREX_PAIRS.keys()))
        with col_f2:
            interval = st.selectbox("Periyot", ["1m", "5m", "15m", "1h"], index=1)
        
        p_symbol = FOREX_PAIRS[selected_pair]
        df_f = fetch_market_data(p_symbol, period="5d", interval=interval)
        if not df_f.empty:
            c = float(df_f['Close'].iloc[-1])
            vwap = float(df_f['VWAP'].iloc[-1])

            st.metric("Anlık Fiyat", f"{c:.4f}")

            # MUM GRAFİK RENDER
            render_candlestick_chart(df_f, f"{selected_pair} ({interval})")

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
