from google.cloud import bigquery
import pandas as pd
import os

PROJECT_ID = "pibiti-dengue"

client = bigquery.Client(project=PROJECT_ID)

PASTA = "dados_dengue"
os.makedirs(PASTA, exist_ok=True)

ANOS = range(2015, 2026)

for ano in ANOS:
    print(f"\n🔄 Baixando {ano}...")

    query = f"""
    SELECT
      CAST(id_municipio AS STRING) AS ID_MUNICIP,
      CAST(id_mn_resi AS STRING) AS ID_MN_RESI,
      data_notificacao AS DT_SIN_PRI
    FROM `basededados.br_ms_sinan.microdados_dengue`
    WHERE EXTRACT(YEAR FROM data_notificacao) = {ano}
      AND data_notificacao IS NOT NULL
    """

    try:
        df = client.query(query).to_dataframe()  # 🔥 sem destination

        caminho = f"{PASTA}/dengue_{ano}.csv"
        df.to_csv(caminho, index=False)

        print(f"✅ {ano} OK ({len(df)} linhas)")

    except Exception as e:
        print(f"❌ {ano}: {e}")