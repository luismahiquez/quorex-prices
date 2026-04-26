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

@app.get("/vix")
def get_vix():
    return get_quote(ticker="^VIX")

class TechnicalContext(BaseModel):
    rsi: Optional[float]
    sma20: Optional[float]
    sma50: Optional[float]
    sma200: Optional[float]
    trend: str
    volumeStatus: str
    supportLevels: list[float]
    resistanceLevels: list[float]

class PriceContext(BaseModel):
    currentPrice: float
    changePercent: float
    volume: int
    averageVolume: Optional[int]
    fiftyTwoWeekHigh: Optional[float]
    fiftyTwoWeekLow: Optional[float]

class FundamentalContext(BaseModel):
    marketCap: Optional[float]
    peRatio: Optional[float]
    eps: Optional[float]
    revenueGrowth: Optional[float]
    profitMargins: Optional[float]
    debtToEquity: Optional[float]

class NewsItem(BaseModel):
    title: str
    publisher: Optional[str]
    date: Optional[str]

class NewsContext(BaseModel):
    sentiment: str
    mainCatalyst: str
    headlines: list[NewsItem]

class MarketContextResponse(BaseModel):
    ticker: str
    companyName: str
    assetType: str
    analysisDate: str
    priceContext: PriceContext
    technicalContext: TechnicalContext
    fundamentalContext: FundamentalContext
    newsContext: NewsContext
    missingData: list[str]
    cached_at: Optional[str] = None

# Context cache (5 min TTL — more aggressive refresh for analysis)
context_cache: dict = {}
CONTEXT_CACHE_TTL = 5 * 60


def get_cached_context(ticker: str) -> Optional[MarketContextResponse]:
    if ticker in context_cache:
        entry = context_cache[ticker]
        if time.time() - entry["timestamp"] < CONTEXT_CACHE_TTL:
            return entry["data"]
        else:
            del context_cache[ticker]
    return None


def set_context_cache(ticker: str, data: MarketContextResponse):
    context_cache[ticker] = {
        "timestamp": time.time(),
        "data": data
    }


def detect_asset_type(info: dict) -> str:
    quote_type = (info.get("quoteType") or "").upper()
    if quote_type == "CRYPTOCURRENCY":
        return "Crypto"
    if quote_type == "ETF":
        return "ETF"
    if quote_type == "EQUITY":
        return "Stock"
    return "Stock"


def calculate_trend(price: float, sma20: Optional[float], sma50: Optional[float], sma200: Optional[float]) -> str:
    above = []
    below = []
    if sma20:
        (above if price > sma20 else below).append("SMA20")
    if sma50:
        (above if price > sma50 else below).append("SMA50")
    if sma200:
        (above if price > sma200 else below).append("SMA200")

    if not above and not below:
        return "Insufficient data for trend analysis"
    if above and not below:
        return f"Price is above {', '.join(above)} — bullish structure"
    if below and not above:
        return f"Price is below {', '.join(below)} — bearish structure"
    return f"Price is above {', '.join(above)} but below {', '.join(below)} — mixed trend"


def calculate_volume_status(volume: int, avg_volume: Optional[int]) -> str:
    if not avg_volume or avg_volume == 0:
        return "Average volume data not available"
    ratio = volume / avg_volume
    if ratio > 1.5:
        return f"Above average volume ({ratio:.1f}x) — elevated activity"
    if ratio < 0.7:
        return f"Below average volume ({ratio:.1f}x) — low activity"
    return f"Normal volume ({ratio:.1f}x average)"


def calculate_support_resistance(hist, current_price: float) -> tuple[list[float], list[float]]:
    support = []
    resistance = []
    try:
        if hist is None or hist.empty:
            return [], []
        closes = hist["Close"].dropna()
        if len(closes) < 20:
            return [], []

        # Simple support/resistance from recent lows and highs
        recent = closes.tail(60)
        low_20  = safe_float(recent.tail(20).min())
        low_60  = safe_float(recent.min())
        high_20 = safe_float(recent.tail(20).max())
        high_60 = safe_float(recent.max())

        if low_20 and low_20 < current_price:
            support.append(low_20)
        if low_60 and low_60 < current_price and low_60 != low_20:
            support.append(low_60)

        if high_20 and high_20 > current_price:
            resistance.append(high_20)
        if high_60 and high_60 > current_price and high_60 != high_20:
            resistance.append(high_60)

    except Exception as e:
        logger.warning(f"Support/resistance calculation failed: {e}")

    return sorted(set(support), reverse=True)[:2], sorted(set(resistance))[:2]


def calculate_news_sentiment(headlines: list[str]) -> tuple[str, str]:
    if not headlines:
        return "Neutral", "No recent news available"

    positive_words = ["upgrade", "beat", "strong", "growth", "bullish", "buy", "surge", "rally", "record", "positive"]
    negative_words = ["downgrade", "miss", "weak", "decline", "bearish", "sell", "drop", "fall", "concern", "risk", "warning"]

    pos_count = sum(1 for h in headlines for w in positive_words if w in h.lower())
    neg_count = sum(1 for h in headlines for w in negative_words if w in h.lower())

    if pos_count > neg_count + 1:
        sentiment = "Positive"
    elif neg_count > pos_count + 1:
        sentiment = "Negative"
    elif pos_count > 0 or neg_count > 0:
        sentiment = "Mixed"
    else:
        sentiment = "Neutral"

    catalyst = headlines[0] if headlines else "No major catalyst identified"
    return sentiment, catalyst


@app.get("/context/{ticker}", response_model=MarketContextResponse)
def get_market_context(ticker: str):
    ticker = ticker.upper().strip()

    cached = get_cached_context(ticker)
    if cached:
        return cached

    logger.info(f"Building market context for {ticker}")
    missing_data = []

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

        # --- Price Context ---
        price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"))
        if price is None:
            raise HTTPException(status_code=404, detail=f"No price data for '{ticker}'")

        prev_close  = safe_float(info.get("previousClose") or info.get("regularMarketPreviousClose"))
        change_pct  = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
        volume      = int(info.get("regularMarketVolume") or info.get("volume") or 0)
        avg_volume  = int(info.get("averageVolume") or info.get("averageDailyVolume10Day") or 0) or None
        high52w     = safe_float(info.get("fiftyTwoWeekHigh"))
        low52w      = safe_float(info.get("fiftyTwoWeekLow"))

        price_ctx = PriceContext(
            currentPrice=price,
            changePercent=change_pct,
            volume=volume,
            averageVolume=avg_volume,
            fiftyTwoWeekHigh=high52w,
            fiftyTwoWeekLow=low52w
        )

        # --- Technical Context ---
        hist = stock.history(period="1y", interval="1d")
        rsi_val = sma20_val = sma50_val = sma200_val = None

        if hist is not None and not hist.empty and len(hist) >= 14:
            close = hist["Close"]

            try:
                rsi_s = ta.rsi(close, length=14)
                if rsi_s is not None and not rsi_s.empty:
                    rsi_val = safe_float(rsi_s.iloc[-1])
            except:
                missing_data.append("RSI")

            if len(close) >= 20:
                try:
                    sma20_s = ta.sma(close, length=20)
                    if sma20_s is not None and not sma20_s.empty:
                        sma20_val = safe_float(sma20_s.iloc[-1])
                except:
                    missing_data.append("SMA20")

            if len(close) >= 50:
                try:
                    sma50_s = ta.sma(close, length=50)
                    if sma50_s is not None and not sma50_s.empty:
                        sma50_val = safe_float(sma50_s.iloc[-1])
                except:
                    missing_data.append("SMA50")

            if len(close) >= 200:
                try:
                    sma200_s = ta.sma(close, length=200)
                    if sma200_s is not None and not sma200_s.empty:
                        sma200_val = safe_float(sma200_s.iloc[-1])
                except:
                    missing_data.append("SMA200")
        else:
            missing_data.extend(["RSI", "SMA20", "SMA50", "SMA200"])

        support, resistance = calculate_support_resistance(hist, price)
        trend        = calculate_trend(price, sma20_val, sma50_val, sma200_val)
        volume_status = calculate_volume_status(volume, avg_volume)

        tech_ctx = TechnicalContext(
            rsi=rsi_val,
            sma20=sma20_val,
            sma50=sma50_val,
            sma200=sma200_val,
            trend=trend,
            volumeStatus=volume_status,
            supportLevels=support,
            resistanceLevels=resistance
        )

        # --- Fundamental Context ---
        market_cap     = safe_float(info.get("marketCap"), 0)
        pe_ratio       = safe_float(info.get("trailingPE") or info.get("forwardPE"))
        eps            = safe_float(info.get("trailingEps") or info.get("forwardEps"))
        revenue_growth = safe_float(info.get("revenueGrowth"))
        profit_margins = safe_float(info.get("profitMargins"))
        debt_to_equity = safe_float(info.get("debtToEquity"))

        if pe_ratio is None:     missing_data.append("P/E Ratio")
        if eps is None:          missing_data.append("EPS")
        if revenue_growth is None: missing_data.append("Revenue Growth")
        if profit_margins is None: missing_data.append("Profit Margins")

        fund_ctx = FundamentalContext(
            marketCap=market_cap,
            peRatio=pe_ratio,
            eps=eps,
            revenueGrowth=revenue_growth,
            profitMargins=profit_margins,
            debtToEquity=debt_to_equity
        )

        # --- News Context ---
        headlines = []
        news_items_raw = []
        try:
            news_items_raw = stock.news or []
            for item in news_items_raw[:5]:
                title     = item.get("content", {}).get("title") or item.get("title", "")
                publisher = item.get("content", {}).get("provider", {}).get("displayName") or item.get("publisher", "")
                pub_date  = item.get("content", {}).get("pubDate") or item.get("providerPublishTime", "")
                if isinstance(pub_date, int):
                    pub_date = datetime.utcfromtimestamp(pub_date).strftime("%Y-%m-%d")
                if title:
                    headlines.append(NewsItem(title=title, publisher=publisher, date=str(pub_date)[:10]))
        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")
            missing_data.append("Recent News")

        headline_texts  = [h.title for h in headlines]
        sentiment, catalyst = calculate_news_sentiment(headline_texts)

        news_ctx = NewsContext(
            sentiment=sentiment,
            mainCatalyst=catalyst,
            headlines=headlines
        )

        # --- Asset type & company name ---
        asset_type   = detect_asset_type(info)
        company_name = info.get("longName") or info.get("shortName") or ticker

        result = MarketContextResponse(
            ticker=ticker,
            companyName=company_name,
            assetType=asset_type,
            analysisDate=datetime.utcnow().strftime("%Y-%m-%d"),
            priceContext=price_ctx,
            technicalContext=tech_ctx,
            fundamentalContext=fund_ctx,
            newsContext=news_ctx,
            missingData=missing_data,
            cached_at=datetime.utcnow().isoformat()
        )

        set_context_cache(ticker, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building context for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build market context for '{ticker}': {str(e)}")

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



