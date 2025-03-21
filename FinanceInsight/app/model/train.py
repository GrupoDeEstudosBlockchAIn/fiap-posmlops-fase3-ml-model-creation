import pandas as pd
from app.src.config import DATA_LAKE_REFINED
from app.model.predict import best_model_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb

# Função para remover outliers usando o método IQR
def remove_outliers(df, columns):

    """Remove outliers das colunas especificadas usando o método do Intervalo Interquartil (IQR)."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def train_and_predict():

    """Treina o modelo e faz previsões para as 10 criptomoedas favoritas."""
    files = list(DATA_LAKE_REFINED.glob("*.parquet"))
    if not files:
        print("Nenhum dado refinado disponível.")
        return
    
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    
    df["Calculation Date"] = pd.to_datetime(df["Calculation Date"])
    df.sort_values("Calculation Date", inplace=True)
    
    features = ["Market Cap", "Volume", "SMA_7", "SMA_30", "EMA_7", "Volatilidade_7"]
    target = "Price"
    
    # Removendo outliers
    df = remove_outliers(df, features + [target])
    
    X = df[features]
    y = df[target]
    
    # Normalizando os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Converter X_train e X_test para DataFrames com os mesmos nomes de colunas
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    
    # Definição do modelo
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Espaço de busca dos hiperparâmetros
    param_dist = {
        "n_estimators": [50, 100, 200, 500],  # Número de árvores
        "max_depth": [3, 5, 7, 10],  # Profundidade máxima
        "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Taxa de aprendizado
        "subsample": [0.6, 0.8, 1.0],  # Amostragem de dados para evitar overfitting
        "colsample_bytree": [0.6, 0.8, 1.0],  # Amostragem de colunas para cada árvore
        "gamma": [0, 0.1, 0.2, 0.3],  # Redução mínima na perda para dividir um nó
        "reg_lambda": [0.01, 0.1, 1, 10],  # Regularização L2
        "reg_alpha": [0.01, 0.1, 1, 10]  # Regularização L1
    }

    # RandomizedSearchCV para encontrar os melhores hiperparâmetros
    random_search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, 
        n_iter=20, cv=3, scoring="neg_mean_absolute_error", 
        n_jobs=-1, verbose=1, random_state=42
    )

    # Ajustando o modelo com os melhores hiperparâmetros
    random_search.fit(X_train, y_train)

    # Obtendo os melhores hiperparâmetros encontrados
    best_params = random_search.best_params_
    print("Melhores Hiperparâmetros Encontrados:", best_params)

    # Treinar modelo final com os melhores hiperparâmetros
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    
    #############
    # Previsões #
    #############
    best_model_predict(best_model, scaler, X_test, y_test, df, features)

