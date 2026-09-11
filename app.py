import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. GENEL SAYFA YAPILANDIRMASI & TELEGRAM
# ==========================================
st.set_page_config(page_title="BIST & Forex AI Trading Terminal", layout="wide")

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
def fetch_market_data(symbol, period="1mo", interval="15m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty or len(df) < 20:
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

        # RVOL (Hacim Katı)
        df['RVOL'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)

        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ==========================================
# MADDE 1: TRADINGVIEW GRAFİK ENTEGRASYONU
# ==========================================
def render_tradingview_widget(symbol):
    """TradingView Orijinal Web Widget Entegrasyonu"""
    # yfinance sembolünü TradingView formatına çevir
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
# MADDE 2: BACKTEST (GERİYE DÖNÜK TEST) MOTORU
# ==========================================
def run_backtest(df, rsi_buy=55, rsi_sell=45):
    """RSI ve VWAP Kesişim Stratejisi Backtest Simülasyonu"""
    trades = []
    position = None
    buy_price = 0

    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        rsi = df['RSI'].iloc[i]
        vwap = df['VWAP'].iloc[i]

        # AL Sinyali Koşulu (RSI > eşik ve Fiyat VWAP üstünde)
        if position is None and rsi > rsi_buy and current_price > vwap:
            position = 'LONG'
            buy_price = current_price

        # SAT Sinyali Koşulu (RSI < eşik veya Fiyat VWAP altına indi)
        elif position == 'LONG' and (rsi < rsi_sell or current_price < vwap):
            profit_pct = ((current_price - buy_price) / buy_price) * 100
            trades.append(profit_pct)
            position = None

    if not trades:
        return 0, 0, 0

    total_return = sum(trades)
    win_rate = (len([t for t in trades if t > 0]) / len(trades)) * 100
    return round(total_return, 2), round(win_rate, 1), len(trades)

# ==========================================
# MADDE 3: YAPAY ZEKA (ML) TAVAN TAHMİN MOTORU
# ==========================================
def predict_tavan_probability_ml(df):
    """RandomForest/XGBoost Mimarisi ile Tavan Yapma İhtimali Tahmini"""
    if len(df) < 50:
        return 0.0

    df = df.copy()
    # Hedef Değişken (Target): Sonraki 3 mum içinde %3'ten fazla yükseliş yakalandı mı?
    df['Target'] = (df['Close'].shift(-3) > df['Close'] * 1.03).astype(int)

    features = ['RSI', 'VWAP', 'RVOL', 'ATR']
    df_clean = df.dropna()

    X = df_clean[features]
    y = df_clean['Target']

    if len(X) < 30 or len(np.unique(y)) < 2:
        return 50.0

    # Model Eğitimi
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X[:-1], y[:-1])

    # Son Mum İçin Tahmin Yap
    last_features = X.iloc[[-1]]
    prob = model.predict_proba(last_features)[0][1] * 100
    return round(prob, 1)

# ==========================================
# 4. SOL ANA MENÜ (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Terminal Menüsü")
selected_market = st.sidebar.radio(
    "Modül Seçin:",
    [
        "🇹🇷 BIST Hisseleri & AI Tavan Motoru", 
        "🌍 Forex & Emtia Terminali", 
        "🧪 Backtest Lab (Geriye Dönük Test)",
        "🧮 Risk & Lot Hesaplayıcı"
    ]
)

# ==========================================
# MODÜL 1: BIST & AI TAVAN MOTORU
# ==========================================
if selected_market == "🇹🇷 BIST Hisseleri & AI Tavan Motoru":
    st.title("🇹🇷 BIST Scalp & Yapay Zeka Tavan Terminali")
    tab1, tab2 = st.tabs(["🚀 AI Destekli Tavan Taraması", "📊 TradingView İnteraktif Grafik"])

    with tab1:
        st.subheader("🎯 ML (Machine Learning) ve Skorlama Altyapısı")

        if st.button("🚀 BIST AI Tavan Taramasını Başlat"):
            results = []
            telegram_alerts = []
            bar = st.progress(0)

            for i, symbol in enumerate(BIST_TUM_LIST):
                data = fetch_market_data(symbol, period="5d", interval="15m")
                if not data.empty:
                    last_c = float(data['Close'].iloc[-1])
                    rsi = float(data['RSI'].iloc[-1])
                    rvol = float(data['RVOL'].iloc[-1])
                    
                    # AI Tahmini Çağır
                    ai_prob = predict_tavan_probability_ml(data)
                    hisse_kodu = symbol.replace(".IS", "")

                    results.append({
                        "Hisse": hisse_kodu,
                        "Son Fiyat (TL)": round(last_c, 2),
                        "RSI (14)": round(rsi, 1),
                        "Hacim Katı (RVOL)": round(rvol, 1),
                        "🤖 AI Tavan İhtimali (%)": ai_prob
                    })

                    if ai_prob >= 75.0:
                        msg = (
                            f"🤖 *YAPAY ZEKA TAVAN SİNYALİ: #{hisse_kodu}*\n\n"
                            f"• **AI Tavan Olasılığı:** %{ai_prob}\n"
                            f"• **Son Fiyat:** {round(last_c, 2)} TL\n"
                            f"• **RVOL (Hacim Patlaması):** {round(rvol, 1)}x\n"
                            f"• **RSI (14):** {round(rsi, 1)}\n\n"
                            f"⚡ *Not:* Makine öğrenimi algoritması yüksek ivme tespit etti."
                        )
                        telegram_alerts.append(msg)

                bar.progress((i + 1) / len(BIST_TUM_LIST))

            if results:
                df_res = pd.DataFrame(results)
                df_res = df_res.sort_values(by="🤖 AI Tavan İhtimali (%)", ascending=False)
                st.dataframe(df_res, use_container_width=True)

                if telegram_alerts:
                    for alert_msg in telegram_alerts:
                        send_telegram_signal(alert_msg)
                    st.success(f"📱 AI Tahmini %75 üzeri olan {len(telegram_alerts)} sinyal Telegram'a iletildi!")

    with tab2:
        symbol = st.selectbox("Grafiğini İncelemek İstediğiniz Hisse", BIST_TUM_LIST)
        st.subheader(f"📈 {symbol} Orijinal TradingView Paneli")
        render_tradingview_widget(symbol)

# ==========================================
# MODÜL 2: FOREX & EMTİA TERMİNALİ
# ==========================================
elif selected_market == "🌍 Forex & Emtia Terminali":
    st.title("🌍 Forex, Emtia & Kripto Terminali")
    selected_pair = st.selectbox("Parite Seçin", list(FOREX_PAIRS.keys()))
    p_symbol = FOREX_PAIRS[selected_pair]

    st.subheader(f"📈 {selected_pair} TradingView Canlı Akış")
    render_tradingview_widget(p_symbol)

# ==========================================
# MODÜL 3: BACKTEST LAB (GERİYE DÖNÜK TEST)
# ==========================================
elif selected_market == "🧪 Backtest Lab (Geriye Dönük Test)":
    st.title("🧪 Algoritmik Strateji Backtest Laboratuvarı")
    st.markdown("Stratejinizi gerçek piyasada kullanmadan önce geçmiş veride test edin.")

    col1, col2, col3 = st.columns(3)
    b_symbol = col1.selectbox("Test Edilecek Varlık", BIST_TUM_LIST + list(FOREX_PAIRS.values()))
    rsi_buy_thresh = col2.slider("RSI AL Eşiği", 40, 70, 55)
    rsi_sell_thresh = col3.slider("RSI SAT Eşiği", 30, 60, 45)

    if st.button("🧪 Backtest Simülasyonunu Çalıştır"):
        df_back = fetch_market_data(b_symbol, period="1mo", interval="15m")
        if not df_back.empty:
            total_ret, win_rate, total_trades = run_backtest(df_back, rsi_buy_thresh, rsi_sell_thresh)

            m1, m2, m3 = st.columns(3)
            m1.metric("Toplam Getiri", f"%{total_ret}")
            m2.metric("Win Rate (Başarı Oranı)", f"%{win_rate}")
            m3.metric("Toplam İşlem Sayısı", f"{total_trades} Adet")

            if total_ret > 0:
                st.success("🟢 Strateji seçilen periyotta karlı çalışmıştır.")
            else:
                st.error("🔴 Strateji seçilen periyotta zarar ettirmiştir. Parametreleri optimize edin.")

# ==========================================
# MODÜL 4: RİSK & LOT HESAPLAYICI
# ==========================================
elif selected_market == "🧮 Risk & Lot Hesaplayıcı":
    st.title("🧮 Pozisyon & Lot Büyüklüğü Hesaplayıcı")

    c1, c2, c3 = st.columns(3)
    balance = c1.number_input("Toplam Bakiye ($ veya TL)", value=10000.0, step=500.0)
    risk_pct = c2.number_input("İşlem Başı Risk (%)", value=1.0, step=0.5)
    stop_dist = c3.number_input("Stop Mesafesi (Pip veya TL)", value=2.0, step=0.5)

    if st.button("Hesapla"):
        max_risk = balance * (risk_pct / 100)
        forex_lot = max_risk / (stop_dist * 10)
        bist_adet = max_risk / stop_dist if stop_dist > 0 else 0

        st.success(f"💵 **Maksimum Göze Alınan Risk:** {max_risk:.2f}")
        st.info(f"📊 **Forex Önerilen Pozisyon:** {forex_lot:.2f} Lot")
        st.info(f"📈 **BIST Önerilen Lot (Adet):** {int(bist_adet)} Adet Hisse")
