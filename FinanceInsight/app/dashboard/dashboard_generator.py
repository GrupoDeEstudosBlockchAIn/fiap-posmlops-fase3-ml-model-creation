import datetime
from app.src.config import DASH_DIR
import plotly.graph_objects as go

def dashboards_cryptocurrency_forecast(df_predictions):
    """Gera e salva dashboards das previsões recebidas, incluindo Preço Atual e Preço Previsto no tooltip."""
    now = datetime.datetime.now()
    forecast_date = now.strftime('%d/%m/%Y')
    forecast_time = now.strftime('%H:%M')
    
    import random
    colors = [f'rgba({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)}, 0.8)' for _ in range(len(df_predictions))]
    
    # Convertendo os preços formatados ("$10.00") em valores numéricos sem adicionar colunas extras
    preco_atual_numerico = df_predictions["Preço Atual"].str.replace("$", "").astype(float)
    preco_previsto_numerico = df_predictions["Preço Previsto"].str.replace("$", "").astype(float)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_predictions["Nome"],
        y=preco_previsto_numerico,  # Usa a variável temporária em vez de uma nova coluna
        marker=dict(color=colors),
        hovertext=[
            f"<b>Símbolo:</b> {symbol}<br>"
            f"<b>Preço Atual:</b> {current_price}<br>"
            f"<b>Preço Previsto:</b> {predicted_price}"
            for symbol, current_price, predicted_price in zip(
                df_predictions["Símbolo"], 
                df_predictions["Preço Atual"], 
                df_predictions["Preço Previsto"]
            )
        ],
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title=f"Criptomoedas com Maior Potencial de Valorização ({forecast_date} - {forecast_time})",
        xaxis_title="Criptomoedas",
        yaxis_title="Preço Previsto (USD)",
        template="plotly_dark"
    )
    
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    dash_filename = DASH_DIR / f"dash_{timestamp}.html"
    fig.write_html(dash_filename)
    
    return dash_filename
