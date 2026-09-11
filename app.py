import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="BIST Pro - Yapay Zeka Tahmin Platformu", layout="wide"
)

st.title("📊 BIST Pro - Yapay Zeka Analiz Platformu")
st.caption(
    "RSI, MACD, Dinamik ATR Stop-Loss, Kar-Al (Take-Profit) ve Random Forest"
    " Modeli"
)


# İndikatör Hesaplama Fonksiyonları
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


# BIST 30 Listesi
BIST30 = [
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

# BIST 100 Tam Liste
BIST100 = [
    "AEFES",
    "AGHOL",
    "AHGAZ",
    "AKBNK",
    "AKCNS",
    "AKFGY",
    "AKFYE",
    "AKSA",
    "AKSEN",
    "ALARK",
    "ALBRK",
    "ALFAS",
    "ANSGR",
    "ARCLK",
    "ASELS",
    "ASTOR",
    "BERA",
    "BIENY",
    "BIMAS",
    "BIOEN",
    "BOBET",
    "BRSAN",
    "BRYAT",
    "BUCIM",
    "CANTE",
    "CCOLA",
    "CIMSA",
    "CWENE",
    "DOAS",
    "DOHOL",
    "EBEBK",
    "ECILC",
    "ECZYT",
    "EGEEN",
    "EKGYO",
    "ENJSA",
    "ENKAI",
    "EREGL",
    "EUPWR",
    "EUREK",
    "FROTO",
    "GARAN",
    "GESAN",
    "GIOSB",
    "GUBRF",
    "GWIND",
    "HALKB",
    "HEKTS",
    "IFAVI",
    "IMASM",
    "INVES",
    "ISCTR",
    "ISGYO",
    "ISMEN",
    "KAYSE",
    "KCAER",
    "KCHOL",
    "KLSER",
    "KMPUR",
    "KONTR",
    "KORDS",
    "KOZAL",
    "KOZAA",
    "KRDMD",
    "KZGYO",
    "MAALT",
    "MAVI",
    "MGROS",
    "MIATK",
    "ODAS",
    "OTKAR",
    "OYAKC",
    "PASEU",
    "PENTA",
    "PETKM",
    "PGSUS",
    "QUAGR",
    "REEDR",
    "SAHOL",
    "SASA",
    "SDTTR",
    "SISE",
    "SKBNK",
    "SMRTG",
    "SOKM",
    "TABGD",
    "TAVHL",
    "TCELL",
    "THYAO",
    "TKFEN",
    "TOASO",
    "TSKB",
    "TTKOM",
    "TTRAK",
    "TUPRS",
    "TURSG",
    "ULKER",
    "VAKBN",
    "VESBE",
    "VESTL",
    "YEOTK",
    "YKBNK",
    "YYLGD",
]

# Yan Menü Kontrolleri
st.sidebar.header("⚙️ Tarama Ayarları")
kapsam = st.sidebar.radio(
    "Tarama Kapsamı:", ["BIST 30", "BIST 100 (Tümü)", "Özel Liste"]
)

ozel_hisseler_input = ""
if kapsam == "Özel Liste":
    ozel_hisseler_input = st.sidebar.text_input(
        "Hisseleri virgülle ayırarak girin:",
        value="THYAO, ASELS, GARAN, EREGL",
        help="Örnek: THYAO, ASELS, KCHOL",
    )

vade = st.sidebar.selectbox("Tahmin Vadesi:", ["1 Günlük Vade", "5 Günlük Vade"])
vade_gun = 1 if vade == "1 Günlük Vade" else 5

# Risk/Ödül Oranı Seçici
rr_ratio = st.sidebar.select_slider(
    "🎯 Risk / Ödül Oranı (Take-Profit):",
    options=[1.0, 1.5, 2.0, 2.5, 3.0],
    value=2.0,
    format_func=lambda x: f"1:{x}",
    help="Stop-Loss mesafesinin kaç katı Hedef Kar-Al fiyatı olsun?",
)

if st.sidebar.button("🚀 Taramayı Başlat"):
    if kapsam == "BIST 30":
        hisseler = BIST30
    elif kapsam == "BIST 100 (Tümü)":
        hisseler = BIST100
    else:
        hisseler = [
            h.strip().upper()
            for h in ozel_hisseler_input.split(",")
            if h.strip()
        ]

    if not hisseler:
        st.warning("Lütfen geçerli en az bir hisse kodu girin.")
    else:
        st.info(f"{len(hisseler)} adet hisse analiz ediliyor...")

        sonuclar = []
        progress_bar = st.progress(0)

        for idx, hisse in enumerate(hisseler):
            try:
                df = yf.download(
                    f"{hisse}.IS", period="1y", interval="1d", progress=False
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if len(df) > 60:
                    # Teknik Göstergeler
                    df["Returns"] = df["Close"].pct_change()
                    df["SMA_10"] = df["Close"].rolling(10).mean()
                    df["SMA_30"] = df["Close"].rolling(30).mean()
                    df["RSI"] = compute_rsi(df["Close"])
                    df["ATR"] = compute_atr(df)

                    # MACD
                    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
                    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
                    df["MACD"] = exp1 - exp2
                    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

                    # Target (Hedef)
                    df["Target"] = (
                        df["Close"].shift(-vade_gun) > df["Close"]
                    ).astype(int)
                    df_model = df.dropna()

                    features = [
                        "Returns",
                        "SMA_10",
                        "SMA_30",
                        "RSI",
                        "MACD",
                        "Signal",
                        "ATR",
                    ]
                    X = df_model[features]
                    y = df_model["Target"]

                    if len(X) > 30:
                        model = RandomForestClassifier(
                            n_estimators=100, random_state=42
                        )
                        model.fit(X[:-vade_gun], y[:-vade_gun])

                        son_veri = X.iloc[[-1]]
                        tahmin = model.predict(son_veri)[0]
                        olasilik = model.predict_proba(son_veri)[0][1]

                        son_fiyat = float(df_model["Close"].iloc[-1])
                        son_rsi = float(df_model["RSI"].iloc[-1])
                        son_atr = float(df_model["ATR"].iloc[-1])

                        # Dinamik ATR Stop-Loss ve Take-Profit (Hedef)
                        risk_mesafesi = 1.5 * son_atr
                        stop_loss = son_fiyat - risk_mesafesi
                        take_profit = son_fiyat + (risk_mesafesi * rr_ratio)

                        sonuclar.append({
                            "Hisse": hisse,
                            "Son Fiyat (TL)": round(son_fiyat, 2),
                            "ATR Stop-Loss (TL)": round(stop_loss, 2),
                            f"Hedef Kar-Al (1:{rr_ratio})": round(
                                take_profit, 2
                            ),
                            "Yükseliş İhtimali (%)": round(olasilik * 100, 1),
                            "RSI (14)": round(son_rsi, 1),
                            "ATR (14)": round(son_atr, 2),
                            "Sinyal": (
                                "🔥 Güçlü Yükseliş"
                                if olasilik > 0.65
                                else (
                                    "📈 Yükseliş"
                                    if olasilik > 0.50
                                    else "📉 Nötr / Düşüş"
                                )
                            ),
                        })
            except Exception:
                pass

            progress_bar.progress((idx + 1) / len(hisseler))

        if sonuclar:
            st.session_state["res_df"] = pd.DataFrame(sonuclar).sort_values(
                by="Yükseliş İhtimali (%)", ascending=False
            )
            st.session_state["rr_ratio"] = rr_ratio
        else:
            st.warning("Veri çekilemedi veya girilen hisseler bulunamadı.")

# Ekran Çıktısı Alanı
if "res_df" in st.session_state and not st.session_state["res_df"].empty:
    res_df = st.session_state["res_df"]
    rr_val = st.session_state.get("rr_ratio", 2.0)
    tp_col_name = f"Hedef Kar-Al (1:{rr_val})"

    st.success("Analiz Tamamlandı!")

    # Öne Çıkan Hisseler
    top_count = min(5, len(res_df))
    st.subheader(f"🔥 En Yüksek Yükseliş Potansiyelli İlk {top_count} Hisse")
    top_n = res_df.head(top_count)

    cols = st.columns(top_count)
    for i, (_, row) in enumerate(top_n.iterrows()):
        with cols[i]:
            st.metric(
                label=f"#{i+1} {row['Hisse']}",
                value=f"{row['Son Fiyat (TL)']} TL",
                delta=f"Hedef: {row[tp_col_name]} TL",
                delta_color="normal",
            )
            st.caption(f"🛑 Stop: **{row['ATR Stop-Loss (TL)']} TL**")

    st.divider()

    # Tüm Liste Tablosu ve CSV İndirme Butonu
    col_header, col_btn = st.columns([3, 1])

    with col_header:
        st.subheader("📋 Tüm Tarama Sonuçları")

    with col_btn:
        # Pandas DataFrame'ini UTF-8 CSV formatına çevirme
        csv_data = res_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )

        st.download_button(
            label="📥 Sonuçları CSV İndir",
            data=csv_data,
            file_name="bist_analiz_sonuclari.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(res_df, use_container_width=True)

    st.divider()

    # Plotly Interaktif Grafik Modülü
    st.subheader("📈 Hisse Detay ve Teknik Analiz Grafiği")

    secilen_hisse = st.selectbox(
        "Grafiğini incelemek istediğiniz hisseyi seçin:",
        res_df["Hisse"].tolist(),
    )

    if secilen_hisse:
        df_chart = yf.download(
            f"{secilen_hisse}.IS", period="6m", interval="1d", progress=False
        )
        if isinstance(df_chart.columns, pd.MultiIndex):
            df_chart.columns = df_chart.columns.get_level_values(0)

        df_chart["RSI"] = compute_rsi(df_chart["Close"])
        df_chart["ATR"] = compute_atr(df_chart)

        # Seçilen hissenin tablodaki stop ve hedef değerlerini alma
        hisse_row = res_df[res_df["Hisse"] == secilen_hisse].iloc[0]
        sel_stop = hisse_row["ATR Stop-Loss (TL)"]
        sel_tp = hisse_row[tp_col_name]

        # 3 Katmanlı Alt Grafik Yapısı (Mum Grafiği, RSI, ATR)
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f"{secilen_hisse} Fiyat & Mum Grafiği",
                "RSI (14)",
                "ATR (14)",
            ),
        )

        # 1. Mum Grafiği
        fig.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
                name="Fiyat",
            ),
            row=1,
            col=1,
        )

        # Stop-Loss ve Hedef Kar-Al Yatay Çizgileri
        fig.add_hline(
            y=sel_tp,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Hedef (Kar-Al): {sel_tp} TL",
            annotation_position="top right",
            row=1,
            col=1,
        )

        fig.add_hline(
            y=sel_stop,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Stop-Loss: {sel_stop} TL",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

        # 2. RSI Çizgisi
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple"),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # 3. ATR Çizgisi
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["ATR"],
                mode="lines",
                name="ATR",
                line=dict(color="orange"),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=850,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)
