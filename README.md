# README.md
# Yahoo Finance API

Este projeto é uma API para buscar cotações de ações no Yahoo Finance e armazenar os dados nos formatos `.parquet` e `.csv`, seguindo a arquitetura Clean Architecture.

## Como executar

1. Instale as dependências:
```sh
pip install -r requirements.txt
```

2. Execute a API:
```sh
uvicorn src.api.main:app --reload
```

3. Acesse a documentação interativa no Swagger:
```
http://127.0.0.1:8000/docs
```


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
│   │   ├── evaluate.py  # Avaliação do modelo
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
