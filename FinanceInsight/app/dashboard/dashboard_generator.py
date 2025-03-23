
import datetime
from app.src.config import DASH_DIR
import plotly.graph_objects as go

def dashboards_cryptocurrency_forecast(df_predictions):
    """Gera e salva dashboards das previsões recebidas."""
    now = datetime.datetime.now()
    forecast_date = now.strftime('%d/%m/%Y')
    forecast_time = now.strftime('%H:%M')
    
    import random
    colors = [f'rgba({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)}, 0.8)' for _ in range(len(df_predictions))]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_predictions["Nome"],
        y=df_predictions["Preço Previsto"].str.replace("$", "").astype(float),
        marker=dict(color=colors),
        hovertext=[f"{symbol}, {price}" for symbol, price in zip(df_predictions["Símbolo"], df_predictions["Preço Previsto"])],
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title=f"Previsão de Preços das Top 10 Criptomoedas ({forecast_date} - {forecast_time})",
        xaxis_title="Criptomoedas",
        yaxis_title="Preço Previsto (USD)",
        template="plotly_dark"
    )
    
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    dash_filename = DASH_DIR / f"dash_{timestamp}.html"
    fig.write_html(dash_filename)
    
    return dash_filename
