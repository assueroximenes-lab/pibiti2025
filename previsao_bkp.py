import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import numpy as np

st.set_page_config(layout="wide")

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():

    arquivos = [
        "dengue2021.csv",
        "dengue2022.csv",
        "dengue2023.csv"
    ]

    dfs = []

    for arq in arquivos:
        df_temp = pd.concat(
            pd.read_csv(
                arq,
                usecols=["ID_MUNICIP", "ID_MN_RESI", "DT_SIN_PRI", "CLASSI_FIN", "CRITERIO"],
                dtype=str,
                encoding="latin1",
                na_values=["NA"],
                quotechar='"',
                chunksize=500000
            ),
            ignore_index=True
        )
        dfs.append(df_temp)

    dengue = pd.concat(dfs, ignore_index=True)

    dengue.columns = dengue.columns.str.strip().str.upper()

    col_data = "DT_SIN_PRI"
    dengue[col_data] = pd.to_datetime(dengue[col_data], errors="coerce")
    dengue = dengue.dropna(subset=[col_data])

    # 🔥 CLASSIFICAÇÃO CORRETA
    dengue["CLASSI_FIN"] = pd.to_numeric(dengue["CLASSI_FIN"], errors="coerce")
    dengue["CRITERIO"] = pd.to_numeric(dengue["CRITERIO"], errors="coerce")

    dengue["CONFIRMADO"] = dengue["CLASSI_FIN"].isin([10, 11, 12])
    dengue["CONFIRMADO"] = dengue["CONFIRMADO"] & dengue["CRITERIO"].isin([1, 2])

    # limpar códigos
    for col in ["ID_MUNICIP", "ID_MN_RESI"]:
        dengue[col] = (
            dengue[col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.strip()
        )

    dengue["ID6"] = dengue["ID_MN_RESI"]
    dengue.loc[
        (dengue["ID6"].isna()) | (dengue["ID6"] == ""),
        "ID6"
    ] = dengue["ID_MUNICIP"]

    dengue["ID6"] = dengue["ID6"].str.zfill(6)

    # =========================
    # MUNICÍPIOS (IBGE)
    # =========================
    municipios = pd.read_csv("municipios.csv", dtype=str)
    municipios.columns = municipios.columns.str.lower().str.strip()

    col_id = [c for c in municipios.columns if "ibge" in c or "codigo" in c][0]
    col_nome = [c for c in municipios.columns if "nome" in c][0]
    col_lat = [c for c in municipios.columns if "lat" in c][0]
    col_lon = [c for c in municipios.columns if "lon" in c][0]

    municipios = municipios.rename(columns={
        col_id: "ID7",
        col_nome: "MUNICIPIO_NOME",
        col_lat: "LATITUDE",
        col_lon: "LONGITUDE"
    })

    municipios["ID7"] = (
        municipios["ID7"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(7)
    )

    municipios["ID6"] = municipios["ID7"].str[:6]

    df = dengue.merge(
        municipios[["ID6", "MUNICIPIO_NOME", "LATITUDE", "LONGITUDE"]],
        on="ID6",
        how="left"
    )

    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    # =========================
    # UF
    # =========================
    uf_map = {
        '11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO',
        '21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA',
        '31':'MG','32':'ES','33':'RJ','35':'SP',
        '41':'PR','42':'SC','43':'RS',
        '50':'MS','51':'MT','52':'GO','53':'DF'
    }

    uf_nome = {
        'RO':'Rondônia','AC':'Acre','AM':'Amazonas','RR':'Roraima','PA':'Pará','AP':'Amapá','TO':'Tocantins',
        'MA':'Maranhão','PI':'Piauí','CE':'Ceará','RN':'Rio Grande do Norte','PB':'Paraíba','PE':'Pernambuco',
        'AL':'Alagoas','SE':'Sergipe','BA':'Bahia',
        'MG':'Minas Gerais','ES':'Espírito Santo','RJ':'Rio de Janeiro','SP':'São Paulo',
        'PR':'Paraná','SC':'Santa Catarina','RS':'Rio Grande do Sul',
        'MS':'Mato Grosso do Sul','MT':'Mato Grosso','GO':'Goiás','DF':'Distrito Federal'
    }

    df["UF"] = df["ID6"].str[:2].map(uf_map)
    df["UF_NOME"] = df["UF"].map(uf_nome)

    return df, col_data


def interpretar_previsao(forecast, df_p, titulo=""):

    st.subheader(f"🧠 Interpretação da Previsão {titulo}")

    forecast_futuro = forecast[forecast["ds"] > df_p["ds"].max()].copy()

    if len(forecast_futuro) == 0:
        st.warning("Sem dados futuros para interpretação")
        return

    inicio = forecast_futuro["yhat"].iloc[0]
    fim = forecast_futuro["yhat"].iloc[-1]

    if fim > inicio * 1.2:
        tendencia = "crescimento"
    elif fim < inicio * 0.8:
        tendencia = "redução"
    else:
        tendencia = "estabilidade"

    forecast_futuro["mes"] = forecast_futuro["ds"].dt.month

    mes_pico = forecast_futuro.groupby("mes")["yhat"].mean().idxmax()

    meses_nome = {
        1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",
        5:"Maio",6:"Junho",7:"Julho",8:"Agosto",
        9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
    }

    mes_pico_nome = meses_nome[mes_pico]

    media_prevista = forecast_futuro["yhat"].mean()

    if "48" in titulo:
        complemento = "Por se tratar de longo prazo, interpretar com cautela."
    else:
        complemento = "Boa confiabilidade para curto prazo."

    texto = f"""
A previsão indica uma tendência de **{tendencia}**.

Mês de pico: **{mes_pico_nome}**

Média estimada: **{int(media_prevista)} casos**

{complemento}
"""
    st.info(texto)


#################################################################################
## INÍCIO
#################################################################################
df, col_data = load_data()

# KPI INICIAL
st.title("🦟 Dashboard Epidemiológico de Dengue")

colA, colB, colC = st.columns(3)
colA.metric("Total Registros", len(df))
colB.metric("Confirmados", int(df["CONFIRMADO"].sum()))
colC.metric("Não Confirmados", len(df) - int(df["CONFIRMADO"].sum()))

# 🔥 FILTRAR CONFIRMADOS
df = df[df["CONFIRMADO"]].copy()

# ========================
# FILTROS
# ========================
st.sidebar.image("piviti.png", use_container_width=True)
st.sidebar.title("Filtros")

frequencia = st.sidebar.selectbox(
    "Agregação Temporal",
    ["Semanal", "Mensal", "Anual"]
)

uf_nome = dict(zip(df["UF"], df["UF_NOME"]))

uf = st.sidebar.selectbox(
    "Estado",
    ["Todos"] + sorted(df["UF"].dropna().unique()),
    format_func=lambda x: f"{x} - {uf_nome.get(x,'')}" if x != "Todos" else x
)

if uf != "Todos":
    municipios_filtrados = sorted(
        df[df["UF"] == uf]["MUNICIPIO_NOME"].dropna().unique()
    )
else:
    municipios_filtrados = sorted(df["MUNICIPIO_NOME"].dropna().unique())

municipio = st.sidebar.selectbox(
    "Município",
    ["Todos"] + list(municipios_filtrados)
)

ano = st.sidebar.selectbox(
    "Ano",
    ["Todos"] + sorted(df[col_data].dt.year.dropna().unique())
)

# ========================
# FILTRO
# ========================
df_f = df.copy()

if uf != "Todos":
    df_f = df_f[df_f["UF"] == uf]

if municipio != "Todos":
    df_f = df_f[df_f["MUNICIPIO_NOME"] == municipio]

if ano != "Todos":
    df_f = df_f[df_f[col_data].dt.year == ano]

# ========================
# KPI
# ========================
col1, col2, col3 = st.columns(3)

col1.metric("Casos", len(df_f))
col2.metric("Municípios", df_f["MUNICIPIO_NOME"].nunique())

if len(df_f) > 0:
    col3.metric(
        "Período",
        f"{df_f[col_data].min().strftime('%d/%m/%Y')} - {df_f[col_data].max().strftime('%d/%m/%Y')}"
    )

# ========================
# SÉRIE
# ========================
freq_map = {"Semanal": "W", "Mensal": "M", "Anual": "Y"}
freq = freq_map[frequencia]

serie = (
    df_f.groupby(pd.Grouper(key=col_data, freq=freq))
    .size()
    .reset_index(name="casos")
)

serie["media_movel"] = serie["casos"].rolling(4).mean()

fig = px.line(serie, x=col_data, y=["casos", "media_movel"])
st.plotly_chart(fig, use_container_width=True)

# ========================
# MAPA DE CASOS
# ========================
st.subheader("🗺️ Mapa de Casos")

mapa = (
    df_f.dropna(subset=["LATITUDE", "LONGITUDE"])
    .groupby(["MUNICIPIO_NOME", "LATITUDE", "LONGITUDE"])
    .size()
    .reset_index(name="casos")
)

if len(mapa) > 0:

    q1 = mapa["casos"].quantile(0.33)
    q2 = mapa["casos"].quantile(0.66)

    def classificar_risco(valor):
        if valor <= q1:
            return "green"
        elif valor <= q2:
            return "orange"
        else:
            return "red"

    centro = [mapa["LATITUDE"].mean(), mapa["LONGITUDE"].mean()]
else:
    centro = [-14.2350, -51.9253]

m = folium.Map(location=centro, zoom_start=5)

for _, row in mapa.iterrows():

    cor = classificar_risco(row["casos"]) if len(mapa) > 0 else "blue"

    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=8,
        color=cor,
        fill=True,
        fill_color=cor,
        fill_opacity=0.7,
        popup=f"{row['MUNICIPIO_NOME']}<br>Casos: {row['casos']}"
    ).add_to(m)

    folium.Marker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        icon=folium.DivIcon(
            html=f"<div style='font-size:10px;color:white;text-align:center;font-weight:bold;'>{int(row['casos'])}</div>"
        )
    ).add_to(m)

st_folium(m, width=1000, height=600)

# ========================
# TOP 10
# ========================
st.subheader("Top 10 Municípios")

ranking = (
    df_f.groupby("MUNICIPIO_NOME")
    .size()
    .reset_index(name="casos")
    .sort_values(by="casos", ascending=False)
    .head(10)
)

fig_rank = px.bar(ranking, x="casos", y="MUNICIPIO_NOME", orientation="h")
st.plotly_chart(fig_rank, use_container_width=True)

# ========================
# PREVISÃO 24
# ========================
st.subheader("🔮 Previsão (24 meses)")

serie_prev = (
    df_f.groupby(pd.Grouper(key=col_data, freq="M"))
    .size()
    .reset_index(name="casos")
)

serie_prev = serie_prev.set_index(col_data).asfreq("M").fillna(0).reset_index()

df_p = serie_prev.rename(columns={col_data: "ds", "casos": "y"})

if len(df_p) > 12:

    df_p["y"] = np.log1p(df_p["y"])

    model = Prophet(yearly_seasonality=True, interval_width=0.8)
    model.fit(df_p)

    future = model.make_future_dataframe(periods=24, freq="M")
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = np.expm1(forecast[col]).clip(lower=0)

    fig_prev = px.line()
    fig_prev.add_scatter(x=serie_prev[col_data], y=serie_prev["casos"], name="Real")
    fig_prev.add_scatter(x=forecast["ds"], y=forecast["yhat"], name="Previsto")

    st.plotly_chart(fig_prev, use_container_width=True)
    interpretar_previsao(forecast, df_p, "(24 meses)")
else:
    st.warning("Poucos dados para previsão de 24 meses")
# ========================
# PREVISÃO 48
# ========================
st.subheader("🔮 Previsão (48 meses)")

if len(df_p) > 24:

    model = Prophet(yearly_seasonality=True)
    model.fit(df_p)

    future = model.make_future_dataframe(periods=48, freq="M")
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = np.expm1(forecast[col]).clip(lower=0)

    fig_prev = px.line()
    fig_prev.add_scatter(x=df_p["ds"], y=np.expm1(df_p["y"]), name="Real")
    fig_prev.add_scatter(x=forecast["ds"], y=forecast["yhat"], name="Previsto")

    st.plotly_chart(fig_prev, use_container_width=True)
    interpretar_previsao(forecast, df_p, "(48 meses)")
else:
    st.warning("Poucos dados para previsão de 48 meses")