from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.routes import router

app = FastAPI(
    title="CoinGecko API",
    description="API para obter e armazenar cotações de criptomoedas",
    version="1.0",
)

# Inclui as rotas de criptomoedas
app.include_router(router)

# Redireciona a rota raiz para `/cryptos?limit=100`
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/cryptos?limit=100")

