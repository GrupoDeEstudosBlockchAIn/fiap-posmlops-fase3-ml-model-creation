import os
import pandas as pd
from datetime import datetime
from src.core.models import CryptoData

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def save_crypto_data(crypto_data: CryptoData):

    date_str = datetime.now().strftime("%Y-%m-%d")
    nome_arq_str = f"crypto_data_{date_str}"

    # Caminho para o arquivo Excel
    file_path_xlsx = os.path.join(DATA_DIR, f"{nome_arq_str}.xlsx")

    # Caminho para o arquivo Parquet
    file_path_parquet = os.path.join(DATA_DIR, f"{nome_arq_str}.parquet")

    # Cria um DataFrame com os dados recebidos
    df = pd.DataFrame([crypto_data.dict()])

    # Salvar dados em Excel
    if os.path.exists(file_path_xlsx):
        try:
            existing_df = pd.read_excel(file_path_xlsx, engine="openpyxl")
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Erro ao ler o arquivo Excel: {e}")
    df.to_excel(file_path_xlsx, index=False, engine="openpyxl")

    # Salvar dados em Parquet
    if os.path.exists(file_path_parquet):
        try:
            existing_df = pd.read_parquet(file_path_parquet)
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Erro ao ler o arquivo Parquet: {e}")
    df.to_parquet(file_path_parquet, index=False, engine="pyarrow")
