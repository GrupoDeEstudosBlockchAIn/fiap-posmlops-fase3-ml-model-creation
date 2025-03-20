import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.data.data_fetcher import fetch_crypto_data
from app.data.data_processor import process_crypto_data
from app.model.train import train_and_predict
import time

def main():
    while True:
        # Faz o scraping e obtém os dados brutos
        raw_file = fetch_crypto_data()

        # Faz o refinamento dos dados brutos
        process_crypto_data(raw_file)

        # Faz o treinamento do modelo de ML de Previsão
        train_and_predict()

        print("⏳ Aguardando 1 hora para a próxima execução...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
