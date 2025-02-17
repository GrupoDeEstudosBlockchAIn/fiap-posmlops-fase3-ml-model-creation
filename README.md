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
    |── pipeline.yml
financeinsight  
│── app  
│   ├── src  
│   │   ├── main.py  
│   │   ├── webscraping  
│   │   │   ├── api  
│   │   │   │   ├── routes.py  
│   │   │   ├── core  
│   │   │   │   ├── models.py  
│   │   │   │   ├── usecase.py  
│   │   │   ├── infrastructure  
│   │   │   │   ├── cripto_scraper.py  
│   │   │   │   ├── storage.py  
│   │   ├── machine_learning
│   │   │   ├── models  
│   │   │   │   ├── model.py  
│   │   │   │   ├── preprocess.py  
│   │   │   │   ├── predict.py 
│   │   │   ├── training  
│   │   │   │   ├── train.py   
│   │   │   │   ├── evaluation.py  
│   │   │   │   ├── hyperparameter_tuning.py  
│   │   │   ├── dashboards 
│   │   │   │   ├── dashboard.py  
│   │   │   │   ├── templates/ 
│   │   │   ├── tests  
│   │   │   │   ├── test_ml_model.py  
│   ├── data_lake  
│   │   ├── raw/
│   │   ├── refined/
|── tests    
│── requirements.txt
.gitignore
README.md  
