# quorex-prices

Market data microservice for [Quorex Trading Intelligence App](https://quorex-amber.vercel.app).

Built with **FastAPI + yfinance + pandas-ta**, deployed on Railway.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `GET` | `/quote/{ticker}` | Market data + technicals |
| `GET` | `/cache/stats` | Cache status |
| `DELETE` | `/cache/clear` | Clear cache |

### `GET /quote/{ticker}`

**Example:** `GET /quote/NVDA`

```json
{
  "ticker": "NVDA",
  "price": 824.50,
  "change_pct": 2.14,
  "volume": 48000000,
  "rsi": 62.4,
  "ma50": 798.20,
  "ma200": 721.50,
  "high52w": 974.00,
  "low52w": 462.00,
  "news": [
    "NVIDIA beats earnings expectations",
    "New AI chip announced at GTC",
    "Data center revenue hits record high"
  ],
  "cached_at": "2025-01-15T14:30:00"
}
```

**Error responses:**
- `404` — Ticker not found or no market data
- `500` — Internal error fetching data

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000

# Test
curl http://localhost:8000/quote/NVDA
curl http://localhost:8000/quote/AAPL
curl http://localhost:8000/quote/BTC-USD
```

Interactive docs: http://localhost:8000/docs

---

## Deploy to Railway

### Option A — Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login & deploy
railway login
railway init        # select "quorex-prices" as project name
railway up
```

### Option B — GitHub Integration (recommended)

1. Push this repo to GitHub as `quorex-prices`
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select `quorex-prices`
4. Railway auto-detects Python via Nixpacks — no extra config needed
5. Set environment variables if needed (none required by default)
6. Your service URL: `https://quorex-prices-production.up.railway.app`

---

## Integration with .NET 8 API (quorex-api)

Add this to your `appsettings.json`:

```json
{
  "PricesApi": {
    "BaseUrl": "https://quorex-prices-production.up.railway.app"
  }
}
```

Example C# service call:

```csharp
public class PricesService
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;

    public PricesService(HttpClient http, IConfiguration config)
    {
        _http = http;
        _baseUrl = config["PricesApi:BaseUrl"];
    }

    public async Task<QuoteDto?> GetQuoteAsync(string ticker)
    {
        var response = await _http.GetAsync($"{_baseUrl}/quote/{ticker}");
        if (!response.IsSuccessStatusCode) return null;
        return await response.Content.ReadFromJsonAsync<QuoteDto>();
    }
}

public record QuoteDto(
    string Ticker,
    decimal Price,
    decimal ChangePct,
    long Volume,
    decimal? Rsi,
    decimal? Ma50,
    decimal? Ma200,
    decimal? High52w,
    decimal? Low52w,
    List<string> News
);
```

Register in `Program.cs`:

```csharp
builder.Services.AddHttpClient<PricesService>();
```

---

## Cache

- TTL: **15 minutes** per ticker
- Storage: in-memory (resets on restart)
- Monitor: `GET /cache/stats`
- Flush: `DELETE /cache/clear`

---

## Supported Tickers

Any ticker supported by Yahoo Finance:
- US Stocks: `NVDA`, `AAPL`, `TSLA`, `MSFT`
- Crypto: `BTC-USD`, `ETH-USD`
- ETFs: `SPY`, `QQQ`, `ARKK`
- Indices: `^GSPC`, `^NDX`
- Forex: `EURUSD=X`

---

## Tech Stack

- **Python 3.12**
- **FastAPI 0.115** — REST API framework
- **yfinance 0.2.50** — Yahoo Finance data
- **pandas-ta 0.3.14** — RSI, SMA indicators
- **Railway** — Deployment platform
