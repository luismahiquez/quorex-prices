import time
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quorex Prices API",
    description="Market data microservice for Quorex Trading Intelligence App",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache
cache: dict = {}
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes


class QuoteResponse(BaseModel):
    ticker: str
    price: float
    change_pct: float
    volume: int
    rsi: Optional[float]
    ma50: Optional[float]
    ma200: Optional[float]
    high52w: Optional[float]
    low52w: Optional[float]
    news: list[str]
    cached_at: Optional[str] = None


def get_cached(ticker: str) -> Optional[QuoteResponse]:
    if ticker in cache:
        entry = cache[ticker]
        if time.time() - entry["timestamp"] < CACHE_TTL_SECONDS:
            logger.info(f"Cache hit for {ticker}")
            return entry["data"]
        else:
            del cache[ticker]
    return None


def set_cache(ticker: str, data: QuoteResponse):
    cache[ticker] = {
        "timestamp": time.time(),
        "data": data
    }


def safe_float(value, decimals: int = 2) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return round(float(value), decimals)
    except Exception:
        return None


@app.get("/")
def root():
    return {
        "service": "Quorex Prices API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/quote/{ticker}", "/health", "/cache/clear"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/quote/{ticker}", response_model=QuoteResponse)
def get_quote(ticker: str):
    ticker = ticker.upper().strip()

    # Check cache first
    cached = get_cached(ticker)
    if cached:
        return cached

    logger.info(f"Fetching fresh data for {ticker}")

    try:
        stock = yf.Ticker(ticker)

        # Get basic info
        info = stock.info
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Try to validate ticker with fast_info
            fast = stock.fast_info
            if not hasattr(fast, "last_price") or fast.last_price is None:
                raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found or has no market data")

        # Price & change
        price = safe_float(
            info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        )
        prev_close = safe_float(info.get("previousClose") or info.get("regularMarketPreviousClose"))

        if price is None:
            raise HTTPException(status_code=404, detail=f"No price data available for '{ticker}'")

        change_pct = 0.0
        if prev_close and prev_close != 0:
            change_pct = round(((price - prev_close) / prev_close) * 100, 2)

        # Volume
        volume = int(info.get("regularMarketVolume") or info.get("volume") or 0)

        # 52-week range
        high52w = safe_float(info.get("fiftyTwoWeekHigh"))
        low52w = safe_float(info.get("fiftyTwoWeekLow"))

        # Technical indicators: fetch 1 year of daily data
        hist = stock.history(period="1y", interval="1d")

        rsi_val = None
        ma50_val = None
        ma200_val = None

        if hist is not None and not hist.empty and len(hist) >= 14:
            close = hist["Close"]

            # RSI (14)
            try:
                rsi_series = ta.rsi(close, length=14)
                if rsi_series is not None and not rsi_series.empty:
                    rsi_val = safe_float(rsi_series.iloc[-1])
            except Exception as e:
                logger.warning(f"RSI calculation failed for {ticker}: {e}")

            # MA50
            if len(close) >= 50:
                try:
                    ma50_series = ta.sma(close, length=50)
                    if ma50_series is not None and not ma50_series.empty:
                        ma50_val = safe_float(ma50_series.iloc[-1])
                except Exception as e:
                    logger.warning(f"MA50 calculation failed for {ticker}: {e}")

            # MA200
            if len(close) >= 200:
                try:
                    ma200_series = ta.sma(close, length=200)
                    if ma200_series is not None and not ma200_series.empty:
                        ma200_val = safe_float(ma200_series.iloc[-1])
                except Exception as e:
                    logger.warning(f"MA200 calculation failed for {ticker}: {e}")

        # News headlines (up to 5)
        news_headlines = []
        try:
            news_items = stock.news or []
            for item in news_items[:5]:
                title = item.get("content", {}).get("title") or item.get("title", "")
                if title:
                    news_headlines.append(title)
        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")

        result = QuoteResponse(
            ticker=ticker,
            price=price,
            change_pct=change_pct,
            volume=volume,
            rsi=rsi_val,
            ma50=ma50_val,
            ma200=ma200_val,
            high52w=high52w,
            low52w=low52w,
            news=news_headlines,
            cached_at=datetime.utcnow().isoformat()
        )

        set_cache(ticker, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data for '{ticker}': {str(e)}")


@app.delete("/cache/clear")
def clear_cache():
    count = len(cache)
    cache.clear()
    return {"cleared": count, "message": f"Removed {count} cached entries"}


@app.get("/cache/stats")
def cache_stats():
    now = time.time()
    entries = []
    for ticker, entry in cache.items():
        age_seconds = int(now - entry["timestamp"])
        entries.append({
            "ticker": ticker,
            "age_seconds": age_seconds,
            "expires_in_seconds": max(0, CACHE_TTL_SECONDS - age_seconds)
        })
    return {"total": len(cache), "entries": entries}

@app.get("/search")
def search_tickers(q: str):
    q = q.strip().upper()
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    try:
        # Single character: try exact ticker lookup
        if len(q) == 1:
            stock = yf.Ticker(q)
            info = stock.info
            name = info.get("longName") or info.get("shortName")
            if name:
                return [{
                    "ticker": q,
                    "name": name,
                    "exchange": info.get("exchange", ""),
                    "type": info.get("quoteType", "EQUITY")
                }]
            return []

        # 2+ characters: use yfinance Search
        search = yf.Search(q, max_results=8)
        quotes = search.quotes or []
        results = []
        for item in quotes:
            ticker = item.get("symbol")
            name = item.get("longname") or item.get("shortname")
            if ticker and name:
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "exchange": item.get("exchange", ""),
                    "type": item.get("quoteType", "EQUITY")
                })
        return results

    except Exception as e:
        logger.error(f"Search error for '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/vix")
def get_vix():
    return get_quote_data("^VIX")
