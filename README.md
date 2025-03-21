# README.md
# Yahoo Finance API

Este projeto é uma API para buscar as melhores cotações das 100 criptomoedas favoritas no Yahoo Finance e armazenar os dados nos formatos `.csv` e `.parquet`, seguindo a arquitetura Clean Architecture.

## 1. Criando ambiente virtual
-> Instalação do ambiente VENV:
```bash
python -m venv venv 
```

-> Ativação do ambiente VENV:
```bash
venv\Scripts\activate 
```


## 2. Dependências
-> Instale as dependências:
```bash
pip install -r requirements.txt
```


## 3. Executar a API
-> Setar o Path:
```bash
cd financeinsight/app/src
```

-> Executar a API
```bash
python main.py
``` 


## 4. Arquitetura do Projeto
.github
|── workflows
|   |── pipeline.yml
financeinsight/
│── app/
│   ├── src/
│   │   ├── main.py  # Ponto de entrada principal
│   │   ├── config.py  # Configurações gerais da aplicação
│   │   ├── __init__.py
│   │
│   ├── data/
│   │   ├── data_fetcher.py  # Coleta de dados do Yahoo Finance
│   │   ├── data_processor.py  # Processamento e refinamento de dados
│   │   ├── feature_engineering.py  # Criação de indicadores técnicos (RSI, MACD, Bollinger)
│   │   ├── __init__.py
│   │
│   ├── model/
│   │   ├── train.py  # Treinamento do modelo
│   │   ├── predict.py  # Previsões com o modelo treinado
│   │   ├── __init__.py
│   │
│   ├── report/
│   │   ├── report_generator.py  # Geração de relatórios e dashboards
│   │   ├── __init__.py
│   │
│   ├── dashboard/
│   │   ├── dashboard_generator.py  # Geração do dashboard interativo
│   │   ├── __init__.py
│   │
│   ├── __init__.py
│
├── data_lake/
│   ├── raw/  # Dados brutos coletados
│   ├── refined/  # Dados processados e prontos para uso
│
├── trained_model/  # Modelos treinados
│
├── generated_dashboards/  # Dashboards gerados
│
├── generated_reports/  # Relatórios gerados
│
|── requirements.txt  # Dependências do projeto
|── .gitignore
|── README.md  


## 5. Documentação completa do Projeto
-> Doc_Previsao_Preco_Cripto.md