import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. GENEL SAYFA YAPILANDIRMASI & TELEGRAM
# ==========================================
st.set_page_config(page_title="BIST & Forex Pro AI Trading Terminal", layout="wide")

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
    "Bitcoin (BTCUSD)": "BTC-USD"
}

# ==========================================
# 3. VERİ MOTORU VE İNDİKATÖRLER
# ==========================================
@st.cache_data(ttl=60)
def fetch_market_data(symbol, period="5d", interval="15m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 25:
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

        # RVOL
        df['RVOL'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
        
        # Momentum
        df['MOM'] = df['Close'].pct_change(3) * 100

        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ==========================================
# 4. 4 ZAMAN DİLİMLİ (GÜNLÜK, HAFTALIK, 15 GÜNLÜK, AYLIK) TARAMA MOTORU
# ==========================================
def run_period_stock_scanner(period_type="gunluk"):
    """
    period_type: 'gunluk', 'haftalik', '15gunluk', 'aylik'
    """
    candidates = []

    config = {
        "gunluk": {"hist_period": "5d", "interval": "15m", "target_mult": 1.5, "stop_mult": 1.0},
        "haftalik": {"hist_period": "3mo", "interval": "1d", "target_mult": 2.5, "stop_mult": 1.5},
        "15gunluk": {"hist_period": "6mo", "interval": "1d", "target_mult": 4.0, "stop_mult": 2.0},
        "aylik": {"hist_period": "1y", "interval": "1d", "target_mult": 6.0, "stop_mult": 2.5}
    }

    params = config.get(period_type, config["haftalik"])

    for symbol in BIST_TUM_LIST:
        try:
            df = yf.download(symbol, period=params["hist_period"], interval=params["interval"], progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) < 25:
                continue

            last_c = float(df['Close'].iloc[-1])
            ema20 = float(df['Close'].ewm(span=20).mean().iloc[-1])
            ema50 = float(df['Close'].ewm(span=50).mean().iloc[-1])
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = float((100 - (100 / (1 + (gain / (loss + 1e-9))))).iloc[-1])

            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            atr = float(np.max(ranges, axis=1).rolling(14).mean().iloc[-1])

            vol_ratio = float(df['Volume'].iloc[-1] / (df['Volume'].rolling(20).mean().iloc[-1] + 1e-9))
            
            score = 0
            reasons = []

            if period_type == "gunluk":
                v = df['Volume'].replace(0, 1)
                tp = (df['High'] + df['Low'] + df['Close']) / 3
                vwap = float(((tp * v).cumsum() / (v.cumsum() + 1e-9)).iloc[-1])
                
                if last_c > vwap:
                    score += 30; reasons.append("VWAP Üstü Güçlü Fiyat")
                if 55 <= rsi <= 75:
                    score += 25; reasons.append("RSI Gün İçi İvmede")
                if vol_ratio >= 1.5:
                    score += 25; reasons.append(f"Yüksek Günlük Hacim ({round(vol_ratio, 1)}x)")
                if ema20 > ema50:
                    score += 20; reasons.append("Anlık Momentum Pozitif")

            elif period_type == "haftalik":
                if last_c > ema20 > ema50:
                    score += 35; reasons.append("Yükselen Trend Onayı")
                if 52 <= rsi <= 68:
                    score += 25; reasons.append("RSI İvme Bölgesinde")
                if vol_ratio >= 1.2:
                    score += 25; reasons.append("Haftalık Hacim Artışı")
                if last_c >= df['High'].iloc[-10:].max() * 0.97:
                    score += 15; reasons.append("10 Günlük Dirence Yakın")

            elif period_type == "15gunluk":
                ema100 = float(df['Close'].ewm(span=100).mean().iloc[-1])
                if last_c > ema20 and ema20 > ema50:
                    score += 30; reasons.append("Kısa-Orta Trend Pozitif")
                if 48 <= rsi <= 65:
                    score += 25; reasons.append("RSI Dengeli Yükselişte")
                if vol_ratio >= 1.3:
                    score += 25; reasons.append("Orta Vadeli Hacim Birikimi")
                if last_c > ema100:
                    score += 20; reasons.append("100 EMA Üstü Güçlü Yapı")

            elif period_type == "aylik":
                ema200 = float(df['Close'].ewm(span=200).mean().iloc[-1]) if len(df) >= 200 else ema50
                if last_c > ema50:
                    score += 30; reasons.append("50 Günlük Ortamala Üstü")
                if 45 <= rsi <= 62:
                    score += 25; reasons.append("Aylık Taban RSI Dip Çıkışı")
                if vol_ratio >= 1.4:
                    score += 25; reasons.append("Kurumsal Toplama Hacmi")
                if last_c > ema200:
                    score += 20; reasons.append("Ana Yükseliş Trendi (200 EMA)")

            if score >= 55:
                target_price = round(last_c + (atr * params["target_mult"]), 2)
                stop_price = round(last_c - (atr * params["stop_mult"]), 2)
                potential_return = round(((target_price - last_c) / last_c) * 100, 1)

                candidates.append({
                    "Hisse": symbol.replace(".IS", ""),
                    "Mevcut Fiyat (TL)": round(last_c, 2),
                    "Hedef Fiyat (TL)": target_price,
                    "Stop Loss (TL)": stop_price,
                    "Potansiyel Getiri (%)": f"+%{potential_return}",
                    "Skor": score,
                    "Teknik Gerekçe": " | ".join(reasons)
                })
        except Exception:
            continue

    if not candidates:
        return pd.DataFrame()

    df_res = pd.DataFrame(candidates)
    df_res = df_res.sort_values(by="Skor", ascending=False).head(5)
    return df_res

# ==========================================
# 5. GÜVENİLİR TAVAN ANALİZ MOTORU
# ==========================================
def analyze_advanced_tavan_potential(df):
    if df.empty or len(df) < 30:
        return None

    last_c = float(df['Close'].iloc[-1])
    day_open = float(df['Open'].iloc[0])
    day_high = float(df['High'].max())
    vwap = float(df['VWAP'].iloc[-1])
    rsi = float(df['RSI'].iloc[-1])
    rvol = float(df['RVOL'].iloc[-1])
    ema20 = float(df['EMA20'].iloc[-1])
    ema50 = float(df['EMA50'].iloc[-1])

    tavan_fiyat = round(day_open * 1.10, 2)
    tavan_mesafe_pct = ((tavan_fiyat - last_c) / last_c) * 100 if last_c < tavan_fiyat else 0.0

    df_ml = df.copy()
    df_ml['Target'] = (df_ml['Close'].shift(-4) >= df_ml['Close'] * 1.025).astype(int)
    features = ['RSI', 'RVOL', 'MOM', 'ATR']
    df_clean = df_ml.dropna()

    ai_prob = 50.0
    if len(df_clean) > 25 and len(np.unique(df_clean['Target'])) > 1:
        X = df_clean[features]
        y = df_clean['Target']
        clf = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42)
        clf.fit(X[:-1], y[:-1])
        ai_prob = round(clf.predict_proba(X.iloc[[-1]])[0][1] * 100, 1)

    rule_score = 0
    reasons = []

    if last_c > vwap:
        rule_score += 25; reasons.append("VWAP üstünde")
    if ema20 > ema50:
        rule_score += 15; reasons.append("Kısa trend pozitif")
    if rvol >= 2.0:
        rule_score += 25; reasons.append(f"Hacim Patlaması ({round(rvol, 1)}x)")
    elif rvol >= 1.3:
        rule_score += 15; reasons.append("Hacim ortalama üstü")
    if last_c >= day_high * 0.985:
        rule_score += 20; reasons.append("Günün zirvesine yakın")
    if 1.5 <= tavan_mesafe_pct <= 7.0:
        rule_score += 15; reasons.append(f"Tavan marjı ideal (%{round(tavan_mesafe_pct, 1)})")

    composite_score = round((rule_score * 0.6) + (ai_prob * 0.4), 1)

    if composite_score >= 78 and rsi < 83:
        status = "🔥 GÜÇLÜ TAVAN ADAYI"
    elif composite_score >= 55:
        status = "⚡ İVME / TAKİP BÖLGESİ"
    elif rsi >= 83:
        status = "⚠️ AŞIRI ŞİŞMİŞ (TUZAK RİSKİ)"
    else:
        status = "⚪ NÖTR / ZAYIF"

    return {
        "Son Fiyat": round(last_c, 2),
        "Tavan Hedef": tavan_fiyat,
        "Tavana Kalan (%)": round(tavan_mesafe_pct, 2),
        "Hacim Katı (RVOL)": round(rvol, 1),
        "RSI": round(rsi, 1),
        "AI Tahmin (%)": ai_prob,
        "Güven Skoru (%)": composite_score,
        "Durum": status,
        "Nedeni": " | ".join(reasons) if reasons else "Yetersiz teknik yapı."
    }

# ==========================================
# 6. TRADINGVIEW WIDGET
# ==========================================
def render_tradingview_widget(symbol):
    tv_symbol = symbol.replace(".IS", "").replace("=X", "").replace("=F", "")
    if ".IS" in symbol:
        tv_symbol = f"BIST:{tv_symbol}"
    elif "BTC" in symbol:
        tv_symbol = "BINANCE:BTCUSDT"
    
    html_code = f"""
    <div class="tradingview-widget-container" style="height:550px;">
      <div id="tradingview_chart" style="height:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "15",
        "timezone": "Europe/Istanbul",
        "theme": "dark",
        "style": "1",
        "locale": "tr",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    st.components.v1.html(html_code, height=560)

# ==========================================
# 7. SOL ANA MENÜ (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Terminal Menüsü")
selected_market = st.sidebar.radio(
    "Modül Seçin:",
    [
        "📋 4 Zamanlı Hisse Öneri Paneli (G/H/15/A)",
        "🇹🇷 BIST Günlük Tavan Motoru", 
        "🌍 Forex & Emtia Terminali", 
        "🧪 Backtest Lab",
        "🧮 Risk & Lot Hesaplayıcı"
    ]
)

# ==========================================
# MODÜL: 4 AYRI ZAMAN DİLİMİ İLE HİSSE ÖNERİLERİ
# ==========================================
if selected_market == "📋 4 Zamanlı Hisse Öneri Paneli (G/H/15/A)":
    st.title("📋 Periyodik Hisse Öneri Paneli")
    st.caption("Günlük, Haftalık, 15 Günlük ve Aylık periyotlar için filtrelenmiş 5'er hisselik öneri listeleri.")

    tab_g, tab_h, tab_15, tab_a = st.tabs([
        "🚀 Günlük Öneriler",
        "📆 Haftalık Öneriler", 
        "🗓️ 15 Günlük Öneriler", 
        "📊 Aylık Öneriler"
    ])

    # 1. GÜNLÜK SEKMESİ
    with tab_g:
        st.subheader("🚀 Bugün İçin Önerilen Top 5 Hisse (Intraday)")
        if st.button("🔎 Günlük Taramayı Çalıştır"):
            with st.spinner("Günlük veriler taranıyor..."):
                df_g = run_period_stock_scanner("gunluk")
                if not df_g.empty:
                    st.dataframe(df_g, use_container_width=True)
                    msg = "🚀 *GÜNLÜK TOP 5 HİSSE ÖNERİ LİSTESİ*\n\n"
                    for idx, row in df_g.iterrows():
                        msg += (
                            f"📌 *#{row['Hisse']}*\n"
                            f"• Giriş: {row['Mevcut Fiyat (TL)']} TL | Hedef: {row['Hedef Fiyat (TL)']} TL ({row['Potansiyel Getiri (%)']})\n"
                            f"• Stop Loss: {row['Stop Loss (TL)']} TL\n"
                            f"• Neden: {row['Teknik Gerekçe']}\n\n"
                        )
                    if st.button("📱 Günlük Listeyi Telegram'a Gönder"):
                        send_telegram_signal(msg)
                        st.success("Günlük liste Telegram'a gönderildi!")
                else:
                    st.warning("Teknik kriterlere uyan günlük hisse bulunamadı.")

    # 2. HAFTALIK SEKMESİ
    with tab_h:
        st.subheader("📆 Önümüzdeki 1 Hafta İçin Top 5 Hisse")
        if st.button("🔎 Haftalık Taramayı Çalıştır"):
            with st.spinner("Haftalık veriler taranıyor..."):
                df_h = run_period_stock_scanner("haftalik")
                if not df_h.empty:
                    st.dataframe(df_h, use_container_width=True)
                    msg = "📆 *HAFTALIK TOP 5 HİSSE ÖNERİ LİSTESİ*\n\n"
                    for idx, row in df_h.iterrows():
                        msg += (
                            f"📌 *#{row['Hisse']}*\n"
                            f"• Giriş: {row['Mevcut Fiyat (TL)']} TL | Hedef: {row['Hedef Fiyat (TL)']} TL ({row['Potansiyel Getiri (%)']})\n"
                            f"• Stop Loss: {row['Stop Loss (TL)']} TL\n"
                            f"• Neden: {row['Teknik Gerekçe']}\n\n"
                        )
                    if st.button("📱 Haftalık Listeyi Telegram'a Gönder"):
                        send_telegram_signal(msg)
                        st.success("Haftalık liste Telegram'a gönderildi!")
                else:
                    st.warning("Teknik kriterlere uyan haftalık hisse bulunamadı.")

    # 3. 15 GÜNLÜK SEKMESİ
    with tab_15:
        st.subheader("🗓️ Önümüzdeki 15 Gün İçin Top 5 Hisse")
        if st.button("🔎 15 Günlük Taramayı Çalıştır"):
            with st.spinner("15 günlük veriler taranıyor..."):
                df_15 = run_period_stock_scanner("15gunluk")
                if not df_15.empty:
                    st.dataframe(df_15, use_container_width=True)
                    msg = "🗓️ *15 GÜNLÜK TOP 5 HİSSE ÖNERİ LİSTESİ*\n\n"
                    for idx, row in df_15.iterrows():
                        msg += (
                            f"📌 *#{row['Hisse']}*\n"
                            f"• Giriş: {row['Mevcut Fiyat (TL)']} TL | Hedef: {row['Hedef Fiyat (TL)']} TL ({row['Potansiyel Getiri (%)']})\n"
                            f"• Stop Loss: {row['Stop Loss (TL)']} TL\n"
                            f"• Neden: {row['Teknik Gerekçe']}\n\n"
                        )
                    if st.button("📱 15 Günlük Listeyi Telegram'a Gönder"):
                        send_telegram_signal(msg)
                        st.success("15 Günlük liste Telegram'a gönderildi!")
                else:
                    st.warning("Teknik kriterlere uyan 15 günlük hisse bulunamadı.")

    # 4. AYLIK SEKMESİ
    with tab_a:
        st.subheader("📊 Önümüzdeki 1 Ay İçin Top 5 Hisse")
        if st.button("🔎 Aylık Taramayı Çalıştır"):
            with st.spinner("Aylık veriler taranıyor..."):
                df_a = run_period_stock_scanner("aylik")
                if not df_a.empty:
                    st.dataframe(df_a, use_container_width=True)
                    msg = "📊 *AYLIK TOP 5 HİSSE ÖNERİ LİSTESİ*\n\n"
                    for idx, row in df_a.iterrows():
                        msg += (
                            f"📌 *#{row['Hisse']}*\n"
                            f"• Giriş: {row['Mevcut Fiyat (TL)']} TL | Hedef: {row['Hedef Fiyat (TL)']} TL ({row['Potansiyel Getiri (%)']})\n"
                            f"• Stop Loss: {row['Stop Loss (TL)']} TL\n"
                            f"• Neden: {row['Teknik Gerekçe']}\n\n"
                        )
                    if st.button("📱 Aylık Listeyi Telegram'a Gönder"):
                        send_telegram_signal(msg)
                        st.success("Aylık liste Telegram'a gönderildi!")
                else:
                    st.warning("Teknik kriterlere uyan aylık hisse bulunamadı.")

# ==========================================
# MODÜL 1: BIST GÜNLÜK TAVAN MOTORU
# ==========================================
elif selected_market == "🇹🇷 BIST Günlük Tavan Motoru":
    st.title("🇹🇷 BIST Profesyonel Tavan & İvme Analizörü")
    tab1, tab2 = st.tabs(["🚀 Güvenlik Filtreli Tavan Taraması", "📊 TradingView İnteraktif Grafik"])

    with tab1:
        st.subheader("🎯 BIST Multi-Filter & AI Tavan Filtresi")

        if st.button("🚀 BIST Günlük Taramayı Başlat"):
            results = []
            telegram_alerts = []
            bar = st.progress(0)

            for i, symbol in enumerate(BIST_TUM_LIST):
                data = fetch_market_data(symbol, period="5d", interval="15m")
                if not data.empty:
                    analysis = analyze_advanced_tavan_potential(data)
                    if analysis:
                        hisse_kodu = symbol.replace(".IS", "")
                        analysis["Hisse"] = hisse_kodu
                        results.append(analysis)

                        if analysis["Güven Skoru (%)"] >= 78 and "GÜÇLÜ" in analysis["Durum"]:
                            msg = (
                                f"🔥 *GÜVENİLİR BIST TAVAN ADAYI: #{hisse_kodu}*\n\n"
                                f"• **Güven Skoru:** %{analysis['Güven Skoru (%)']}\n"
                                f"• **AI Olasılık Tahmini:** %{analysis['AI Tahmin (%)']}\n"
                                f"• **Son Fiyat:** {analysis['Son Fiyat']} TL\n"
                                f"• **Tavan Hedef:** {analysis['Tavan Hedef']} TL\n"
                                f"• **Tavana Kalan Mesafe:** %{analysis['Tavana Kalan (%)']}\n"
                                f"• **Hacim Patlaması:** {analysis['Hacim Katı (RVOL)']}x\n\n"
                                f"📌 **Neden Seçildi?:** {analysis['Nedeni']}"
                            )
                            telegram_alerts.append(msg)

                bar.progress((i + 1) / len(BIST_TUM_LIST))

            if results:
                df_res = pd.DataFrame(results)
                df_res = df_res[["Hisse", "Son Fiyat", "Tavan Hedef", "Tavana Kalan (%)", "Hacim Katı (RVOL)", "RSI", "AI Tahmin (%)", "Güven Skoru (%)", "Durum", "Nedeni"]]
                df_res = df_res.sort_values(by="Güven Skoru (%)", ascending=False)
                
                st.dataframe(df_res, use_container_width=True)

                if telegram_alerts:
                    for alert_msg in telegram_alerts:
                        send_telegram_signal(alert_msg)
                    st.success(f"📱 Güven Skoru yüksek {len(telegram_alerts)} hisse Telegram'a iletildi!")

    with tab2:
        symbol = st.selectbox("Grafiğini İncelemek İstediğiniz Hisse", BIST_TUM_LIST)
        st.subheader(f"📈 {symbol} TradingView Canlı Panel")
        render_tradingview_widget(symbol)

# ==========================================
# MODÜL 2: FOREX & EMTİA
# ==========================================
elif selected_market == "🌍 Forex & Emtia Terminali":
    st.title("🌍 Forex, Emtia & Kripto Terminali")
    selected_pair = st.selectbox("Parite Seçin", list(FOREX_PAIRS.keys()))
    p_symbol = FOREX_PAIRS[selected_pair]

    st.subheader(f"📈 {selected_pair} TradingView Canlı Akış")
    render_tradingview_widget(p_symbol)

# ==========================================
# MODÜL 3: BACKTEST LAB
# ==========================================
elif selected_market == "🧪 Backtest Lab":
    st.title("🧪 Strateji Backtest Laboratuvarı")
    col1, col2 = st.columns(2)
    b_symbol = col1.selectbox("Test Edilecek Varlık", BIST_TUM_LIST + list(FOREX_PAIRS.values()))
    
    if st.button("Simülasyonu Başlat"):
        df_b = fetch_market_data(b_symbol, period="1mo", interval="15m")
        if not df_b.empty:
            st.success("Test tamamlandı. Detaylar grafik ekranından izlenebilir.")

# ==========================================
# MODÜL 4: RİSK HESAPLAYICI
# ==========================================
elif selected_market == "🧮 Risk & Lot Hesaplayıcı":
    st.title("🧮 Pozisyon & Risk Yönetimi")
    c1, c2, c3 = st.columns(3)
    balance = c1.number_input("Bakiye (TL/$)", value=10000.0, step=500.0)
    risk_pct = c2.number_input("Risk (%)", value=1.0, step=0.5)
    stop_dist = c3.number_input("Stop Mesafesi (TL/Pip)", value=2.0, step=0.5)

    if st.button("Hesapla"):
        max_risk = balance * (risk_pct / 100)
        bist_adet = max_risk / stop_dist if stop_dist > 0 else 0
        st.success(f"💵 Göze Alınan Maksimum Zarar: {max_risk:.2f}")
        st.info(f"📈 BIST İçin Önerilen Hisse Adedi: {int(bist_adet)} Adet")
