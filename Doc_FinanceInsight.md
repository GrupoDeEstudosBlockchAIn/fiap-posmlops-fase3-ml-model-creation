# **Documentação do Projeto FinanceInsight**

## **Visão Geral**
O projeto **FinanceInsight** tem como objetivo a coleta, processamento, análise e previsão de preços das **Criptomoedas com Maior Potencial de Valorização**, utilizando Machine Learning. Os dados são obtidos do Yahoo Finance e refinados para a extração de indicadores técnicos como **RSI, MACD e Bandas de Bollinger**. O modelo de Machine Learning faz previsões e gera relatórios e dashboards interativos para suporte à tomada de decisão.

---

## **Estrutura do Projeto**

```
.github/
│── workflows/
│   │── pipeline.yml  # Configuração do CI/CD via GitHub Actions
financeinsight/
│── app/
│   ├── src/
│   │   ├── main.py  # Ponto de entrada principal da aplicação
│   │   ├── config.py  # Configurações gerais da aplicação
│   │   ├── __init__.py
│   │
│   ├── data/
│   │   ├── data_fetcher.py  # Coleta de dados do Yahoo Finance
│   │   ├── data_processor.py  # Processamento e refinamento de dados
│   │   ├── feature_engineering.py  # Criação de indicadores técnicos (RSI, MACD e Bollinger)
│   │   ├── __init__.py
│   │
│   ├── model/
│   │   ├── train.py  # Treinamento do modelo
│   │   ├── predict.py  # Previsões e Avaliação com o modelo treinado
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
│   ├── raw/  # Armazena os dados brutos coletados
│   ├── refined/  # Armazena os dados processados e prontos para uso
│
├── trained_model/  # Diretório para os modelos treinados
│
├── generated_dashboards/  # Dashboards gerados
│
├── generated_reports/  # Relatórios gerados
│
|── requirements.txt  # Dependências do projeto
|── .gitignore
|── README.md
```

---

## **Fluxo de Funcionamento**

### **1. Coleta de Dados**
**Arquivo:** `data/data_fetcher.py`  
🔹 Obtém dados do Yahoo Finance via `yfinance`.  
🔹 Coleta informações de **100 criptomoedas** e salva os dados no diretório **`data_lake/raw`** em `.csv` e `.xlsx`.  
🔹 Logs de erros e sucessos são gravados em **`fetch_crypto_data.log`**.

### **2. Processamento e Refinamento**
**Arquivo:** `data/data_processor.py`  
🔹 Lê os arquivos brutos do **Yahoo Finance** e trata os dados.  
🔹 Calcula indicadores técnicos (**RSI, MACD, Bandas de Bollinger**).  
🔹 Gera novas features e salva os dados refinados em `.parquet` no diretório **`data_lake/refined`**.

### **3. Treinamento do Modelo**
**Arquivo:** `model/train.py`  
🔹 Lê os dados refinados e treina um modelo de **XGBoost**.  
🔹 Remove outliers com **IQR** e normaliza os dados.  
🔹 Aplica **RandomizedSearchCV** para otimização de hiperparâmetros.  
🔹 Salva o modelo treinado em **`trained_model/crypto_price_model.pkl`**.

### **4. Previsões**
**Arquivo:** `model/predict.py`  
🔹 Utiliza o modelo treinado para prever os preços das **Criptomoedas com Maior Potencial de Valorização**.  
🔹 Gera relatório de métricas (MAE, MSE, RMSE, SMAPE e R²).  
🔹 Salva previsões e feature importances.

### **5. Geração de Dashboards**
**Arquivo:** `dashboard/dashboard_generator.py`  
🔹 Utiliza **Plotly** para criar gráficos interativos.  
🔹 Salva os dashboards em HTML no diretório **`generated_dashboards/`**.

### **6. Geração de Relatórios**
**Arquivo:** `report/report_generator.py`  
🔹 Cria relatórios de performance do modelo.  
🔹 Gera gráficos estatísticos e análises detalhadas.  
🔹 Salva relatórios no diretório **`generated_reports/`**.

---

## **Tecnologias Utilizadas**
🔹 **Python** - Linguagem principal do projeto  
🔹 **yFinance** - Coleta de dados financeiros  
🔹 **Pandas** - Manipulação e análise de dados  
🔹 **Scikit-learn** - Pré-processamento e métricas do modelo  
🔹 **XGBoost** - Modelo de Machine Learning  
🔹 **Plotly** - Dashboards interativos  
🔹 **Joblib** - Salvamento e carregamento do modelo treinado  

---

## **Execução do Projeto**

### **1. Clonar o Repositório**
```bash
git clone https://github.com/GrupoDeEstudosBlockchAIn/fiap-posmlops-fase3-ml-model-creation.git
cd financeinsight
```

### **2. Criar e Ativar Ambiente Virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

### **3. Instalar Dependências**
```bash
pip install -r requirements.txt
```

### **4. Rodar a Coleta de Dados**
```bash
cd financeinsight/app/src
python main.py
```
---

## **Exemplo de Reports Gerado**
financeinsight/generated_reports
//-> Visualizar os relatórios mais recentes
---

## **Exemplo de Dashboards Gerado**
financeinsight/generated_dashboards
//-> Visualizar os dashboards mais recentes
---

## **Automação com GitHub Actions**
**Arquivo:** `.github/workflows/pipeline.yml`  
🔹 **Executa os seguintes passos automaticamente em cada commit**:
1. Instala dependências
2. Testa a coleta de dados
3. Testa o pipeline de dados
4. Testa o treinamento e previsões do modelo

---

## **Métricas de Avaliação**
O modelo é avaliado com as seguintes métricas:

🔹 **MAE** - Erro absoluto médio  
🔹 **MSE** - Erro quadrático médio  
🔹 **RMSE** - Raiz do erro quadrático médio  
🔹 **SMAPE** - Erro percentual médio absoluto simétrico  
🔹 **R²** - Coeficiente de determinação  

Exemplo de saída:
```bash
MAE: 0.01, MSE: 0.00, RMSE: 0.03, SMAPE: 15.92, R²: 0.99
```

---

## **Contribuição**
🔹 **Fork o repositório**  
🔹 **Crie uma branch:** `git checkout -b minha-branch`  
🔹 **Faça suas alterações e commit:** `git commit -m "Minha melhoria"`  
🔹 **Envie o código:** `git push origin minha-branch`  
🔹 **Abra um Pull Request**

---

## **Contato**
**Desenvolvedor:** Alexandro de Paula Barros  
**Email:** cittamap77@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/alexandro-de-paula-barros-92484170/

---

**FinanceInsight - Inteligência Financeira com Machine Learning!**