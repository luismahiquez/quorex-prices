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
import re

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

# ============================================================
# AI Market Context Section
# Add this section without modifying /quote, /vix, /search, etc.
# ============================================================

class ContextPrice(BaseModel):
    currentPrice: float
    previousClose: Optional[float]
    changePercent: float
    volume: int
    averageVolume: Optional[int]
    volumeRatio: Optional[float]
    fiftyTwoWeekHigh: Optional[float]
    fiftyTwoWeekLow: Optional[float]
    distanceFrom52WeekHighPercent: Optional[float]


class ContextTechnical(BaseModel):
    rsi: Optional[float]
    rsiStatus: str
    sma20: Optional[float]
    sma50: Optional[float]
    sma200: Optional[float]
    trend: str
    momentumStatus: str
    volumeStatus: str
    supportLevels: list[float]
    resistanceLevels: list[float]


class ContextFundamental(BaseModel):
    marketCap: Optional[float]
    peRatio: Optional[float]
    eps: Optional[float]
    revenueGrowth: Optional[float]
    profitMargins: Optional[float]
    debtToEquity: Optional[float]
    fundamentalSummary: str


class ContextNewsItem(BaseModel):
    title: str
    publisher: Optional[str]
    date: Optional[str]
    relevanceScore: Optional[float]
    category: Optional[str]


class ContextNews(BaseModel):
    sentiment: str
    mainCatalyst: str
    headlines: list[ContextNewsItem]


class MarketContextResponse(BaseModel):
    ticker: str
    companyName: str
    assetType: str
    analysisDate: str
    marketStatus: str
    lastTradingSession: Optional[str]
    priceAsOf: Optional[str]
    priceContext: ContextPrice
    technicalContext: ContextTechnical
    fundamentalContext: ContextFundamental
    newsContext: ContextNews
    missingData: list[str]
    cached_at: Optional[str] = None


# ============================================================
# Context helpers
# ============================================================

def ctx_percent_distance(current: Optional[float], reference: Optional[float]) -> Optional[float]:
    if current is None or reference is None or reference == 0:
        return None

    return round(((current - reference) / reference) * 100, 2)


def ctx_asset_type(info: dict) -> str:
    quote_type = (info.get("quoteType") or "").upper()

    if quote_type == "CRYPTOCURRENCY":
        return "Crypto"

    if quote_type == "ETF":
        return "ETF"

    return "Stock"


def ctx_market_status(info: dict) -> str:
    state = (info.get("marketState") or "").upper()

    if state in ["REGULAR", "OPEN"]:
        return "open"

    if state in ["PRE", "PREMARKET"]:
        return "premarket"

    if state in ["POST", "POSTMARKET"]:
        return "postmarket"

    return "closed"


def ctx_last_trading_session(hist) -> Optional[str]:
    try:
        if hist is None or hist.empty:
            return None

        last_index = hist.index[-1]

        if hasattr(last_index, "strftime"):
            return last_index.strftime("%Y-%m-%d")

        return str(last_index)[:10]

    except Exception:
        return None


def ctx_rsi_status(rsi: Optional[float]) -> str:
    if rsi is None:
        return "Unavailable"

    if rsi >= 70:
        return "Overbought"

    if rsi >= 60:
        return "Bullish"

    if rsi >= 45:
        return "Neutral"

    if rsi >= 30:
        return "Weak"

    return "Oversold"


def ctx_volume_ratio(volume: int, avg_volume: Optional[int]) -> Optional[float]:
    if not avg_volume or avg_volume == 0:
        return None

    return round(volume / avg_volume, 2)


def ctx_volume_status(volume: int, avg_volume: Optional[int]) -> str:
    ratio = ctx_volume_ratio(volume, avg_volume)

    if ratio is None:
        return "Average volume data not available"

    if ratio >= 2.0:
        return f"Very high volume ({ratio:.1f}x average)"

    if ratio >= 1.3:
        return f"Above average volume ({ratio:.1f}x average)"

    if ratio >= 1.1:
        return f"Slightly above average volume ({ratio:.1f}x average)"

    if ratio >= 0.8:
        return f"Normal volume ({ratio:.1f}x average)"

    return f"Below average volume ({ratio:.1f}x average)"


def ctx_trend(
    price: float,
    sma20: Optional[float],
    sma50: Optional[float],
    sma200: Optional[float]
) -> str:
    above = []
    below = []

    if sma20:
        (above if price > sma20 else below).append("SMA20")

    if sma50:
        (above if price > sma50 else below).append("SMA50")

    if sma200:
        (above if price > sma200 else below).append("SMA200")

    if above and not below:
        return f"Price is above {', '.join(above)} — bullish structure"

    if below and not above:
        return f"Price is below {', '.join(below)} — bearish structure"

    if above and below:
        return f"Price is above {', '.join(above)} but below {', '.join(below)} — mixed trend"

    return "Insufficient data for trend analysis"


def ctx_momentum_status(
    price: float,
    sma20: Optional[float],
    rsi: Optional[float]
) -> str:
    if sma20 is None or rsi is None:
        return "Unavailable"

    distance_from_sma20 = ctx_percent_distance(price, sma20)

    if rsi >= 70 and distance_from_sma20 is not None and distance_from_sma20 > 7:
        return "Strong but extended"

    if price > sma20 and rsi >= 60:
        return "Bullish momentum"

    if price > sma20:
        return "Positive trend"

    if price < sma20 and rsi < 45:
        return "Weak momentum"

    return "Neutral"


def ctx_support_resistance(
    hist,
    price: float,
    sma20: Optional[float],
    sma50: Optional[float],
    sma200: Optional[float],
    high52w: Optional[float],
    low52w: Optional[float]
) -> tuple[list[float], list[float]]:
    support = []
    resistance = []

    def add_level(target: list[float], level: Optional[float]):
        if level is None or level <= 0:
            return

        level = round(float(level), 2)

        for existing in target:
            if abs(existing - level) / level < 0.003:
                return

        target.append(level)

    try:
        if hist is not None and not hist.empty and len(hist) >= 20:
            recent = hist.tail(60)

            low_20 = safe_float(recent["Low"].tail(20).min())
            low_60 = safe_float(recent["Low"].min())

            high_20 = safe_float(recent["High"].tail(20).max())
            high_60 = safe_float(recent["High"].max())

            for level in [low_20, low_60, sma20, sma50, sma200, low52w]:
                if level and level < price:
                    add_level(support, level)

            for level in [high_20, high_60, high52w]:
                if level and level > price:
                    add_level(resistance, level)

        # Psychological levels every $5 for stocks above $20
        if price >= 20:
            below = round(price / 5) * 5

            if below >= price:
                below -= 5

            above = below + 5

            if below > 0:
                add_level(support, below)

            if above > price:
                add_level(resistance, above)

    except Exception as e:
        logger.warning(f"Support/resistance failed: {e}")

    support = sorted(support, reverse=True)[:4]
    resistance = sorted(resistance)[:4]

    return support, resistance


def ctx_fundamental_summary(
    market_cap: Optional[float],
    pe_ratio: Optional[float],
    revenue_growth: Optional[float],
    profit_margins: Optional[float]
) -> str:
    parts = []

    if market_cap:
        if market_cap >= 200_000_000_000:
            parts.append("mega-cap company")
        elif market_cap >= 10_000_000_000:
            parts.append("large-cap company")
        else:
            parts.append("smaller-cap company")

    if revenue_growth is not None:
        if revenue_growth >= 0.30:
            parts.append("strong revenue growth")
        elif revenue_growth >= 0.10:
            parts.append("moderate revenue growth")
        elif revenue_growth >= 0:
            parts.append("low revenue growth")
        else:
            parts.append("negative revenue growth")

    if profit_margins is not None:
        if profit_margins >= 0.20:
            parts.append("high profitability")
        elif profit_margins >= 0.05:
            parts.append("positive profitability")
        else:
            parts.append("weak profitability")

    if pe_ratio is not None:
        if pe_ratio >= 40:
            parts.append("premium valuation")
        elif pe_ratio >= 20:
            parts.append("moderate-to-high valuation")
        elif pe_ratio > 0:
            parts.append("lower valuation")

    if not parts:
        return "Insufficient fundamental data."

    return ", ".join(parts).capitalize() + "."


# ============================================================
# News helpers
# ============================================================

def ctx_keyword_match(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False

    text_lower = text.lower()
    keyword_lower = keyword.lower().strip()

    if len(keyword_lower) <= 3:
        return re.search(rf"\b{re.escape(keyword_lower)}\b", text_lower) is not None

    return keyword_lower in text_lower


def ctx_company_tokens(company_name: str) -> list[str]:
    ignored_words = {
        "inc",
        "corp",
        "corporation",
        "company",
        "class",
        "plc",
        "limited",
        "holdings",
        "group",
        "ordinary",
        "shares",
        "the",
        "and",
        "interactive"
    }

    words = re.split(r"[^a-zA-Z0-9]+", company_name.lower())

    return [
        word
        for word in words
        if len(word) > 3 and word not in ignored_words
    ]


def ctx_title_mentions_current_asset(title: str, ticker: str, company_name: str) -> bool:
    title_lower = title.lower()
    ticker_lower = ticker.lower()

    if ctx_keyword_match(title_lower, ticker_lower):
        return True

    company_words = ctx_company_tokens(company_name)

    return any(ctx_keyword_match(title_lower, word) for word in company_words)


def ctx_mentions_different_ticker_only(title: str, ticker: str, company_name: str) -> bool:
    matches = re.findall(r"\(([A-Z]{1,6})\)", title)

    if not matches:
        return False

    # Keep it if it clearly mentions the current asset.
    # Example: "What Makes Peloton (PTON) a New Buy Stock"
    if ctx_title_mentions_current_asset(title, ticker, company_name):
        return False

    # Skip it if it has another ticker and does not mention the current asset.
    # Example: analyzing PTON but title says "Pool Corp. (POOL)..."
    return any(match.upper() != ticker.upper() for match in matches)


def ctx_news_relevance(title: str, ticker: str, company_name: str) -> tuple[float, str]:
    title_lower = title.lower()

    # Asset-specific news
    if ctx_title_mentions_current_asset(title, ticker, company_name):
        return 0.95, "asset"

    # Generic sector/theme news
    sector_words = [
        "ai",
        "artificial intelligence",
        "openai",
        "cloud",
        "software",
        "semiconductor",
        "semiconductors",
        "chip",
        "chips",
        "gpu",
        "data center",
        "datacenter",
        "big tech",
        "mag 7",
        "magnificent seven",
        "earnings",
        "guidance",
        "analyst",
        "upgrade",
        "downgrade",
        "fitness",
        "consumer",
        "retail",
        "subscription",
        "streaming",
        "energy",
        "oil",
        "banking",
        "crypto",
        "bitcoin",
        "biotech",
        "healthcare"
    ]

    if any(ctx_keyword_match(title_lower, word) for word in sector_words):
        return 0.60, "sector"

    # Macro / market-wide news
    macro_words = [
        "fed",
        "powell",
        "inflation",
        "rates",
        "rate cut",
        "rate hike",
        "treasury",
        "yields",
        "wall street",
        "market",
        "stocks",
        "s&p",
        "nasdaq",
        "dow",
        "economy",
        "cpi",
        "ppi",
        "jobs",
        "gdp",
        "recession"
    ]

    if any(ctx_keyword_match(title_lower, word) for word in macro_words):
        return 0.55, "macro"

    return 0.20, "irrelevant"


def ctx_news_sentiment(headlines: list[str]) -> tuple[str, str]:
    if not headlines:
        return "Neutral", "No recent relevant news available"

    positive_words = [
        "upgrade",
        "beat",
        "strong",
        "growth",
        "bullish",
        "buy",
        "surge",
        "rally",
        "record",
        "positive",
        "optimism",
        "outperform",
        "raise",
        "raised",
        "boost",
        "rebound",
        "upside",
        "tops",
        "beats"
    ]

    negative_words = [
        "downgrade",
        "miss",
        "weak",
        "decline",
        "bearish",
        "sell",
        "drop",
        "fall",
        "concern",
        "risk",
        "warning",
        "cut",
        "lawsuit",
        "probe",
        "investigation",
        "slowdown",
        "value trap"
    ]

    pos_count = sum(
        1
        for headline in headlines
        for word in positive_words
        if word in headline.lower()
    )

    neg_count = sum(
        1
        for headline in headlines
        for word in negative_words
        if word in headline.lower()
    )

    if pos_count > neg_count + 1:
        sentiment = "Positive"
    elif neg_count > pos_count + 1:
        sentiment = "Negative"
    elif pos_count > 0 or neg_count > 0:
        sentiment = "Mixed"
    else:
        sentiment = "Neutral"

    return sentiment, headlines[0]


# ============================================================
# AI Market Context Endpoint
# ============================================================

@app.get("/context/{ticker}", response_model=MarketContextResponse)
def get_market_context(ticker: str):
    ticker = ticker.upper().strip()
    missing_data = []

    logger.info(f"Building AI market context for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        hist = stock.history(period="1y", interval="1d")

        if not info and (hist is None or hist.empty):
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

        company_name = info.get("longName") or info.get("shortName") or ticker
        asset_type = ctx_asset_type(info)
        market_status = ctx_market_status(info)
        last_session = ctx_last_trading_session(hist)
        price_as_of = f"{last_session} 16:00:00 ET" if last_session else None

        # -------------------------
        # Price
        # -------------------------

        price = safe_float(
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )

        if price is None and hist is not None and not hist.empty:
            price = safe_float(hist["Close"].dropna().iloc[-1])

        if price is None:
            raise HTTPException(status_code=404, detail=f"No price data for '{ticker}'")

        prev_close = safe_float(
            info.get("previousClose")
            or info.get("regularMarketPreviousClose")
        )

        if prev_close is None and hist is not None and not hist.empty:
            closes = hist["Close"].dropna()
            if len(closes) >= 2:
                prev_close = safe_float(closes.iloc[-2])

        change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0

        volume = int(info.get("regularMarketVolume") or info.get("volume") or 0)

        avg_volume = int(
            info.get("averageVolume")
            or info.get("averageDailyVolume10Day")
            or 0
        ) or None

        volume_ratio = ctx_volume_ratio(volume, avg_volume)

        high52w = safe_float(info.get("fiftyTwoWeekHigh"))
        low52w = safe_float(info.get("fiftyTwoWeekLow"))
        distance_52w_high = ctx_percent_distance(price, high52w)

        if avg_volume is None:
            missing_data.append("Average Volume")

        if high52w is None:
            missing_data.append("52-Week High")

        if low52w is None:
            missing_data.append("52-Week Low")

        price_ctx = ContextPrice(
            currentPrice=price,
            previousClose=prev_close,
            changePercent=change_pct,
            volume=volume,
            averageVolume=avg_volume,
            volumeRatio=volume_ratio,
            fiftyTwoWeekHigh=high52w,
            fiftyTwoWeekLow=low52w,
            distanceFrom52WeekHighPercent=distance_52w_high
        )

        # -------------------------
        # Technical
        # -------------------------

        rsi_val = None
        sma20_val = None
        sma50_val = None
        sma200_val = None

        if hist is not None and not hist.empty and len(hist) >= 14:
            close = hist["Close"].dropna()

            try:
                rsi_series = ta.rsi(close, length=14)
                if rsi_series is not None and not rsi_series.empty:
                    rsi_val = safe_float(rsi_series.iloc[-1])
            except Exception:
                missing_data.append("RSI")

            if len(close) >= 20:
                try:
                    sma20_val = safe_float(ta.sma(close, length=20).iloc[-1])
                except Exception:
                    missing_data.append("SMA20")
            else:
                missing_data.append("SMA20")

            if len(close) >= 50:
                try:
                    sma50_val = safe_float(ta.sma(close, length=50).iloc[-1])
                except Exception:
                    missing_data.append("SMA50")
            else:
                missing_data.append("SMA50")

            if len(close) >= 200:
                try:
                    sma200_val = safe_float(ta.sma(close, length=200).iloc[-1])
                except Exception:
                    missing_data.append("SMA200")
            else:
                missing_data.append("SMA200")

        else:
            missing_data.extend(["RSI", "SMA20", "SMA50", "SMA200"])

        support, resistance = ctx_support_resistance(
            hist=hist,
            price=price,
            sma20=sma20_val,
            sma50=sma50_val,
            sma200=sma200_val,
            high52w=high52w,
            low52w=low52w
        )

        tech_ctx = ContextTechnical(
            rsi=rsi_val,
            rsiStatus=ctx_rsi_status(rsi_val),
            sma20=sma20_val,
            sma50=sma50_val,
            sma200=sma200_val,
            trend=ctx_trend(price, sma20_val, sma50_val, sma200_val),
            momentumStatus=ctx_momentum_status(price, sma20_val, rsi_val),
            volumeStatus=ctx_volume_status(volume, avg_volume),
            supportLevels=support,
            resistanceLevels=resistance
        )

        # -------------------------
        # Fundamentals
        # -------------------------

        market_cap = safe_float(info.get("marketCap"))
        pe_ratio = safe_float(info.get("trailingPE") or info.get("forwardPE"))
        eps = safe_float(info.get("trailingEps") or info.get("forwardEps"))
        revenue_growth = safe_float(info.get("revenueGrowth"))
        profit_margins = safe_float(info.get("profitMargins"))
        debt_to_equity = safe_float(info.get("debtToEquity"))

        if market_cap is None:
            missing_data.append("Market Cap")

        if pe_ratio is None:
            missing_data.append("P/E Ratio")

        if eps is None:
            missing_data.append("EPS")

        if revenue_growth is None:
            missing_data.append("Revenue Growth")

        if profit_margins is None:
            missing_data.append("Profit Margins")

        if debt_to_equity is None:
            missing_data.append("Debt To Equity")

        fund_ctx = ContextFundamental(
            marketCap=market_cap,
            peRatio=pe_ratio,
            eps=eps,
            revenueGrowth=revenue_growth,
            profitMargins=profit_margins,
            debtToEquity=debt_to_equity,
            fundamentalSummary=ctx_fundamental_summary(
                market_cap,
                pe_ratio,
                revenue_growth,
                profit_margins
            )
        )

        # -------------------------
        # News
        # -------------------------

        raw_news = []
        all_news_count = 0

        try:
            news_items = stock.news or []

            # Fallback: sometimes yf.Ticker(ticker).news returns empty.
            if not news_items:
                try:
                    search = yf.Search(ticker, max_results=10)
                    news_items = getattr(search, "news", None) or []
                except Exception as search_ex:
                    logger.warning(f"Search news fallback failed for {ticker}: {search_ex}")

            all_news_count = len(news_items)
            logger.info(f"{ticker} news items found: {all_news_count}")

            for item in news_items[:10]:
                content = item.get("content", {}) if isinstance(item, dict) else {}

                title = (
                    content.get("title")
                    or item.get("title", "")
                )

                publisher = (
                    content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else None
                ) or item.get("publisher", "")

                pub_date = (
                    content.get("pubDate")
                    or item.get("providerPublishTime", "")
                )

                if isinstance(pub_date, int):
                    pub_date = datetime.utcfromtimestamp(pub_date).strftime("%Y-%m-%d")
                else:
                    pub_date = str(pub_date)[:10] if pub_date else None

                if not title:
                    continue

                # Skip headlines clearly about another ticker.
                # Example: analyzing PTON but title says "Pool Corp. (POOL)..."
                if ctx_mentions_different_ticker_only(title, ticker, company_name):
                    logger.info(f"{ticker} skipped different ticker news: {title}")
                    continue

                relevance, category = ctx_news_relevance(
                    title=title,
                    ticker=ticker,
                    company_name=company_name
                )

                logger.info(
                    f"{ticker} news relevance={relevance}, "
                    f"category={category}, title={title}"
                )

                if relevance >= 0.55:
                    raw_news.append(
                        ContextNewsItem(
                            title=title,
                            publisher=publisher,
                            date=pub_date,
                            relevanceScore=relevance,
                            category=category
                        )
                    )

        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")
            missing_data.append("Recent News")

        if all_news_count == 0:
            missing_data.append("Recent News")

        if all_news_count > 0 and not raw_news:
            # Fallback: include best available news sorted by relevance
            fallback_news = []
            for item in news_items[:10]:
                content = item.get("content", {}) if isinstance(item, dict) else {}
                title = content.get("title") or item.get("title", "")
                publisher = (
                    content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else None
                ) or item.get("publisher", "")
                pub_date = content.get("pubDate") or item.get("providerPublishTime", "")
                if isinstance(pub_date, int):
                    pub_date = datetime.utcfromtimestamp(pub_date).strftime("%Y-%m-%d")
                else:
                    pub_date = str(pub_date)[:10] if pub_date else None
                if not title:
                    continue
                if ctx_mentions_different_ticker_only(title, ticker, company_name):
                    continue
                relevance, category = ctx_news_relevance(title, ticker, company_name)
                fallback_news.append(ContextNewsItem(
                    title=title,
                    publisher=publisher,
                    date=pub_date,
                    relevanceScore=relevance,
                    category=category
                ))
            # Sort by relevance and take top 3
            fallback_news.sort(key=lambda x: x.relevanceScore or 0, reverse=True)
            raw_news = fallback_news[:3]
            if raw_news:
                missing_data.append("Asset-specific News")
            else:
                missing_data.append("Relevant News")

        headline_texts = [n.title for n in raw_news]
        sentiment, _ = ctx_news_sentiment(headline_texts)

        asset_news = [n for n in raw_news if n.category == "asset"]

        if asset_news:
            catalyst = asset_news[0].title
        elif raw_news:
            catalyst = f"No asset-specific news. Best available: {raw_news[0].title}"
            if "Asset-specific News" not in missing_data:
                missing_data.append("Asset-specific News")
        else:
            catalyst = "No relevant news available at this time."

        news_ctx = ContextNews(
            sentiment=sentiment,
            mainCatalyst=catalyst,
            headlines=raw_news[:5]
        )

        return MarketContextResponse(
            ticker=ticker,
            companyName=company_name,
            assetType=asset_type,
            analysisDate=datetime.utcnow().strftime("%Y-%m-%d"),
            marketStatus=market_status,
            lastTradingSession=last_session,
            priceAsOf=price_as_of,
            priceContext=price_ctx,
            technicalContext=tech_ctx,
            fundamentalContext=fund_ctx,
            newsContext=news_ctx,
            missingData=list(sorted(set(missing_data))),
            cached_at=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error building context for {ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build market context for '{ticker}': {str(e)}"
        )

def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def get_trend(change_pct: float, threshold: float = 0.10):
    if change_pct > threshold:
        return "up"
    if change_pct < -threshold:
        return "down"
    return "neutral"


def get_latest_intraday_price(ticker: yf.Ticker):
    """
    Latest available intraday price.
    """
    try:
        intraday = ticker.history(period="1d", interval="5m")

        if intraday is not None and not intraday.empty:
            closes = intraday["Close"].dropna()

            if len(closes) > 0:
                return safe_float(closes.iloc[-1])

    except Exception as e:
        logger.warning(f"Failed to get intraday price: {e}")

    return None


def get_info_price(ticker: yf.Ticker):
    """
    Fallback current price from ticker.info.
    """
    try:
        info = ticker.info or {}

        return safe_float(
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )

    except Exception as e:
        logger.warning(f"Failed to get info price: {e}")
        return None


def get_value_from_obj(obj, key):
    """
    Safely read a key from dict-like or object-like yfinance structures.
    """
    try:
        if hasattr(obj, "get"):
            value = obj.get(key)
            if value is not None:
                return value
    except Exception:
        pass

    try:
        value = obj[key]
        if value is not None:
            return value
    except Exception:
        pass

    try:
        value = getattr(obj, key)
        if value is not None:
            return value
    except Exception:
        pass

    return None


def get_previous_close_from_fast_info(ticker: yf.Ticker):
    """
    First attempt: yfinance fast_info.
    """
    try:
        fast_info = ticker.fast_info

        value = (
            get_value_from_obj(fast_info, "previous_close")
            or get_value_from_obj(fast_info, "regular_market_previous_close")
            or get_value_from_obj(fast_info, "last_close")
        )

        return safe_float(value)

    except Exception as e:
        logger.warning(f"Failed to get previous close from fast_info: {e}")
        return None


def get_previous_close_from_info(ticker: yf.Ticker):
    """
    Second attempt: ticker.info.
    """
    try:
        info = ticker.info or {}

        value = (
            info.get("regularMarketPreviousClose")
            or info.get("previousClose")
        )

        return safe_float(value)

    except Exception as e:
        logger.warning(f"Failed to get previous close from info: {e}")
        return None


def get_previous_close_from_daily_history(ticker: yf.Ticker):
    """
    Third attempt: daily history.
    Uses 1mo instead of 7d because futures sometimes return empty daily data
    with short periods.
    """
    try:
        hist = ticker.history(period="1mo", interval="1d")

        if hist is None or hist.empty:
            return None

        closes = hist["Close"].dropna()

        if len(closes) == 0:
            return None

        today_et = datetime.now(ET).date()
        last_index_date = closes.index[-1].date()

        # If the last daily candle is today, use previous completed candle.
        if last_index_date == today_et and len(closes) >= 2:
            return safe_float(closes.iloc[-2])

        # Otherwise, use latest completed daily close.
        return safe_float(closes.iloc[-1])

    except Exception as e:
        logger.warning(f"Failed to get previous close from daily history: {e}")
        return None


def get_futures_session_start_et(now_et: datetime):
    """
    Futures generally trade from 6:00 PM ET to 5:00 PM ET next day.
    This gives us the current futures session start.
    """
    today_6pm = now_et.replace(hour=18, minute=0, second=0, microsecond=0)

    if now_et >= today_6pm:
        return today_6pm

    return today_6pm - timedelta(days=1)


def get_reference_from_intraday_before_session(ticker: yf.Ticker):
    """
    Final fallback:
    Use intraday candles and pick the last close before the current futures session started.

    Example:
    Monday 12:09 AM ET -> current futures session started Sunday 6:00 PM ET.
    Reference should be the last available close before Sunday 6:00 PM ET.
    """
    try:
        now_et = datetime.now(ET)
        session_start = get_futures_session_start_et(now_et)

        hist = ticker.history(period="10d", interval="60m")

        if hist is None or hist.empty:
            return None

        closes = hist["Close"].dropna()

        if len(closes) == 0:
            return None

        # Ensure timezone is Eastern for comparison.
        if closes.index.tz is None:
            closes.index = closes.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            closes.index = closes.index.tz_convert("America/New_York")

        before_session = closes[closes.index < session_start]

        if len(before_session) == 0:
            return None

        return safe_float(before_session.iloc[-1])

    except Exception as e:
        logger.warning(f"Failed to get reference from intraday before session: {e}")
        return None


def get_reference_price(ticker: yf.Ticker):
    """
    Robust reference price resolver.
    """
    reference = get_previous_close_from_fast_info(ticker)
    if reference is not None and reference > 0:
        return reference

    reference = get_previous_close_from_info(ticker)
    if reference is not None and reference > 0:
        return reference

    reference = get_previous_close_from_daily_history(ticker)
    if reference is not None and reference > 0:
        return reference

    reference = get_reference_from_intraday_before_session(ticker)
    if reference is not None and reference > 0:
        return reference

    return None


def normalize_tnx_value(value):
    """
    Yahoo ^TNX can come as 43.10 to represent 4.31%.
    """
    if value is None:
        return None

    if value > 20:
        return value / 10

    return value


def get_vix_level(price):
    if price is None:
        return "unknown"

    if price < 15:
        return "low"

    if price < 20:
        return "normal"

    if price < 25:
        return "elevated"

    return "high"


def build_market_item(
    symbol: str,
    threshold: float = 0.10,
    normalize_tnx: bool = False
):
    ticker = yf.Ticker(symbol)

    current_price = get_latest_intraday_price(ticker)

    if current_price is None:
        current_price = get_info_price(ticker)

    reference_price = get_reference_price(ticker)

    if normalize_tnx:
        current_price = normalize_tnx_value(current_price)
        reference_price = normalize_tnx_value(reference_price)

    change = 0.0
    change_pct = 0.0
    change_source = "calculated"

    if (
        current_price is not None
        and reference_price is not None
        and reference_price != 0
    ):
        change = current_price - reference_price
        change_pct = round((change / reference_price) * 100, 2)
    else:
        change_source = "missing_reference"

    return {
        "symbol": symbol,
        "price": round(current_price, 2) if current_price is not None else None,
        "reference_price": round(reference_price, 2) if reference_price is not None else None,
        "change": round(change, 2),
        "change_pct": change_pct,
        "trend": get_trend(change_pct, threshold),
        "change_source": change_source
    }

def get_relative_strength(asset_change_pct, benchmark_change_pct):
    """
    Compares one asset against the S&P 500 futures change.
    Used to know if Nasdaq, Dow, Russell are leading or lagging.
    """
    if asset_change_pct is None or benchmark_change_pct is None:
        return "unknown"

    diff = asset_change_pct - benchmark_change_pct

    if diff >= 0.15:
        return "leading"

    if diff <= -0.15:
        return "lagging"

    return "in_line"

def build_futures_structure(results):
    """
    Builds a simple market structure summary from futures data.
    This helps the AI understand the market faster.
    """

    sp500 = results.get("sp500", {})
    nasdaq = results.get("nasdaq", {})
    dow = results.get("dow", {})
    russell = results.get("russell", {})
    oil = results.get("oil", {})
    gold = results.get("gold", {})

    sp500_change = sp500.get("change_pct", 0)
    nasdaq_change = nasdaq.get("change_pct", 0)
    dow_change = dow.get("change_pct", 0)
    russell_change = russell.get("change_pct", 0)
    oil_change = oil.get("change_pct", 0)
    gold_change = gold.get("change_pct", 0)

    positive_count = sum([
        sp500_change > 0.10,
        nasdaq_change > 0.10,
        dow_change > 0.10,
        russell_change > 0.10
    ])

    negative_count = sum([
        sp500_change < -0.10,
        nasdaq_change < -0.10,
        dow_change < -0.10,
        russell_change < -0.10
    ])

    if positive_count >= 3:
        futures_tone = "bullish"
    elif negative_count >= 3:
        futures_tone = "bearish"
    else:
        futures_tone = "mixed"

    tech_leadership = get_relative_strength(nasdaq_change, sp500_change)
    small_caps_tone = "confirming" if russell_change > 0.10 else "weak" if russell_change < -0.10 else "neutral"

    if oil_change < -0.50:
        oil_tone = "down_supportive_for_inflation"
    elif oil_change > 0.50:
        oil_tone = "up_inflation_risk"
    else:
        oil_tone = "neutral"

    if gold_change > 0.50:
        gold_tone = "strong_defensive_bid"
    elif gold_change < -0.50:
        gold_tone = "weak"
    else:
        gold_tone = "neutral"

    if futures_tone == "bullish" and oil_tone == "down_supportive_for_inflation":
        overall_tone = "risk_on"
    elif futures_tone == "bullish" and gold_tone == "strong_defensive_bid":
        overall_tone = "risk_on_with_defensive_hedging"
    elif futures_tone == "bearish":
        overall_tone = "risk_off"
    else:
        overall_tone = "mixed"

    return {
        "futuresTone": futures_tone,
        "techLeadership": tech_leadership,
        "smallCapsTone": small_caps_tone,
        "oilTone": oil_tone,
        "goldTone": gold_tone,
        "overallTone": overall_tone
    }


@app.get("/futures")
def get_futures():
    symbols = {
        "sp500": "ES=F",
        "nasdaq": "NQ=F",
        "dow": "YM=F",
        "russell": "RTY=F",
        "oil": "CL=F",
        "gold": "GC=F"
    }

    results = {}

    for name, symbol in symbols.items():
        try:
            results[name] = build_market_item(
                symbol=symbol,
                threshold=0.10
            )

        except Exception as e:
            logger.warning(f"Failed to get futures for {symbol}: {e}")

            results[name] = {
                "symbol": symbol,
                "price": None,
                "reference_price": None,
                "change": 0.0,
                "change_pct": 0.0,
                "trend": "neutral",
                "change_source": "error"
            }

    # Add relative strength vs S&P 500 futures
    sp500_change = results.get("sp500", {}).get("change_pct", 0)

    for key in ["nasdaq", "dow", "russell"]:
        asset_change = results.get(key, {}).get("change_pct", 0)
        results[key]["relative_strength_vs_sp500"] = get_relative_strength(
            asset_change,
            sp500_change
        )

    # Add market structure summary
    market_structure = build_futures_structure(results)

    return {
        "items": results,
        "marketStructure": market_structure
    }


@app.get("/macro")
def get_macro():
    symbols = {
        "vix": "^VIX",
        "tenYearYield": "^TNX",
        "dollarIndex": "DX-Y.NYB"
    }

    results = {}

    for name, symbol in symbols.items():
        try:
            normalize_tnx = symbol == "^TNX"

            if symbol == "^VIX":
                threshold = 0.05
            elif symbol == "DX-Y.NYB":
                threshold = 0.05
            elif symbol == "^TNX":
                threshold = 0.02
            else:
                threshold = 0.10

            item = build_market_item(
                symbol=symbol,
                threshold=threshold,
                normalize_tnx=normalize_tnx
            )

            if symbol == "^VIX":
                item["level"] = get_vix_level(item["price"])

            results[name] = item

        except Exception as e:
            logger.warning(f"Failed to get macro for {symbol}: {e}")

            results[name] = {
                "symbol": symbol,
                "price": None,
                "reference_price": None,
                "change": 0.0,
                "change_pct": 0.0,
                "trend": "neutral",
                "change_source": "error"
            }

    return results

from datetime import datetime, timedelta, timezone
import yfinance as yf


def get_crypto_trend(change_pct: float):
    if change_pct > 1.0:
        return "up"
    if change_pct < -1.0:
        return "down"
    if change_pct > 0.25:
        return "slightly_up"
    if change_pct < -0.25:
        return "slightly_down"
    return "neutral"


def get_crypto_risk_appetite(btc_change, eth_change, sol_change):
    changes = [
        btc_change if btc_change is not None else 0,
        eth_change if eth_change is not None else 0,
        sol_change if sol_change is not None else 0
    ]

    avg = sum(changes) / len(changes)

    positive_count = sum(1 for x in changes if x > 0.5)
    negative_count = sum(1 for x in changes if x < -0.5)

    if avg >= 2.0 and positive_count >= 2:
        return "strong"
    if avg >= 0.5 and positive_count >= 2:
        return "improving"
    if avg <= -2.0 and negative_count >= 2:
        return "risk-off"
    if avg <= -0.5 and negative_count >= 2:
        return "weakening"

    return "neutral"


def build_crypto_item(symbol: str):
    ticker = yf.Ticker(symbol)

    hist = ticker.history(period="2d", interval="5m")

    if hist is None or hist.empty:
        return {
            "symbol": symbol,
            "price": None,
            "price_24h_ago": None,
            "change": 0.0,
            "change_pct_24h": 0.0,
            "trend": "neutral",
            "change_source": "missing_data"
        }

    closes = hist["Close"].dropna()

    if len(closes) == 0:
        return {
            "symbol": symbol,
            "price": None,
            "price_24h_ago": None,
            "change": 0.0,
            "change_pct_24h": 0.0,
            "trend": "neutral",
            "change_source": "missing_data"
        }

    current_price = float(closes.iloc[-1])

    now_utc = datetime.now(timezone.utc)
    target_time = now_utc - timedelta(hours=24)

    # Make index timezone-aware if needed
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC")
    else:
        closes.index = closes.index.tz_convert("UTC")

    before_or_at_24h = closes[closes.index <= target_time]

    if len(before_or_at_24h) > 0:
        price_24h_ago = float(before_or_at_24h.iloc[-1])
    else:
        # fallback: use oldest available close
        price_24h_ago = float(closes.iloc[0])

    change = current_price - price_24h_ago

    if price_24h_ago != 0:
        change_pct = round((change / price_24h_ago) * 100, 2)
    else:
        change_pct = 0.0

    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "price_24h_ago": round(price_24h_ago, 2),
        "change": round(change, 2),
        "change_pct_24h": change_pct,
        "trend": get_crypto_trend(change_pct),
        "change_source": "calculated_24h"
    }


@app.get("/crypto")
def get_crypto():
    symbols = {
        "btc": "BTC-USD",
        "eth": "ETH-USD",
        "sol": "SOL-USD"
    }

    results = {}

    for name, symbol in symbols.items():
        try:
            results[name] = build_crypto_item(symbol)
        except Exception as e:
            logger.warning(f"Failed to get crypto for {symbol}: {e}")
            results[name] = {
                "symbol": symbol,
                "price": None,
                "price_24h_ago": None,
                "change": 0.0,
                "change_pct_24h": 0.0,
                "trend": "neutral",
                "change_source": "error"
            }

    btc_change = results["btc"]["change_pct_24h"]
    eth_change = results["eth"]["change_pct_24h"]
    sol_change = results["sol"]["change_pct_24h"]

    return {
        "items": results,
        "cryptoRiskAppetite": get_crypto_risk_appetite(
            btc_change,
            eth_change,
            sol_change
        )
    }

@app.get("/sectors")
def get_sectors():
    sectors = {
        "technology": {
            "symbol": "XLK",
            "name": "Technology"
        },
        "financials": {
            "symbol": "XLF",
            "name": "Financials"
        },
        "energy": {
            "symbol": "XLE",
            "name": "Energy"
        },
        "healthcare": {
            "symbol": "XLV",
            "name": "Healthcare"
        },
        "consumer_discretionary": {
            "symbol": "XLY",
            "name": "Consumer Discretionary"
        },
        "consumer_staples": {
            "symbol": "XLP",
            "name": "Consumer Staples"
        },
        "industrials": {
            "symbol": "XLI",
            "name": "Industrials"
        },
        "communication_services": {
            "symbol": "XLC",
            "name": "Communication Services"
        },
        "utilities": {
            "symbol": "XLU",
            "name": "Utilities"
        },
        "materials": {
            "symbol": "XLB",
            "name": "Materials"
        },
        "real_estate": {
            "symbol": "XLRE",
            "name": "Real Estate"
        }
    }

    items = {}

    for key, sector in sectors.items():
        symbol = sector["symbol"]

        try:
            ticker = yf.Ticker(symbol)

            # Current / intraday price
            intraday = ticker.history(period="2d", interval="5m")
            daily = ticker.history(period="5d", interval="1d")
            info = ticker.info or {}

            current_price = None
            reference_price = None

            if intraday is not None and not intraday.empty:
                closes_5m = intraday["Close"].dropna()
                if len(closes_5m) > 0:
                    current_price = float(closes_5m.iloc[-1])

            if current_price is None:
                current_price = safe_float(
                    info.get("regularMarketPrice") or
                    info.get("currentPrice")
                )

            # Previous close / reference price
            reference_price = safe_float(
                info.get("regularMarketPreviousClose") or
                info.get("previousClose")
            )

            if reference_price is None and daily is not None and not daily.empty:
                closes_daily = daily["Close"].dropna()

                if len(closes_daily) >= 2:
                    # Use previous daily close as reference
                    reference_price = float(closes_daily.iloc[-2])
                elif len(closes_daily) == 1:
                    reference_price = float(closes_daily.iloc[0])

            if current_price is None and daily is not None and not daily.empty:
                closes_daily = daily["Close"].dropna()
                if len(closes_daily) > 0:
                    current_price = float(closes_daily.iloc[-1])

            change = 0.0
            change_pct = 0.0

            if current_price is not None and reference_price is not None and reference_price != 0:
                change = round(current_price - reference_price, 2)
                change_pct = round((change / reference_price) * 100, 2)

            trend = (
                "up" if change_pct > 0.15 else
                "down" if change_pct < -0.15 else
                "neutral"
            )

            items[key] = {
                "symbol": symbol,
                "name": sector["name"],
                "price": round(current_price, 2) if current_price is not None else None,
                "reference_price": round(reference_price, 2) if reference_price is not None else None,
                "change": change,
                "change_pct": change_pct,
                "trend": trend,
                "change_source": "calculated"
            }

        except Exception as e:
            logger.warning(f"Failed to get sector data for {symbol}: {e}")

            items[key] = {
                "symbol": symbol,
                "name": sector["name"],
                "price": None,
                "reference_price": None,
                "change": 0.0,
                "change_pct": 0.0,
                "trend": "neutral",
                "change_source": "error"
            }

    ranked = sorted(
        items.values(),
        key=lambda x: x.get("change_pct", 0),
        reverse=True
    )

    leaders = ranked[:3]
    laggards = list(reversed(ranked[-3:]))

    sector_tone = get_sector_tone(leaders, laggards)

    return {
        "items": items,
        "leaders": leaders,
        "laggards": laggards,
        "sectorTone": sector_tone
    }


def get_sector_tone(leaders, laggards):
    leader_names = [x["name"] for x in leaders]
    laggard_names = [x["name"] for x in laggards]

    defensive = {"Utilities", "Consumer Staples", "Healthcare", "Real Estate"}
    growth = {"Technology", "Consumer Discretionary", "Communication Services"}
    cyclicals = {"Financials", "Industrials", "Materials", "Energy"}

    defensive_leaders = sum(1 for name in leader_names if name in defensive)
    growth_leaders = sum(1 for name in leader_names if name in growth)
    cyclical_leaders = sum(1 for name in leader_names if name in cyclicals)

    growth_laggards = sum(1 for name in laggard_names if name in growth)
    defensive_laggards = sum(1 for name in laggard_names if name in defensive)

    if growth_leaders >= 2:
        return "growth_led_risk_on"

    if defensive_leaders >= 2 and growth_laggards >= 1:
        return "defensive_rotation"

    if cyclical_leaders >= 2:
        return "cyclical_rotation"

    if growth_laggards >= 2:
        return "growth_under_pressure"

    if defensive_laggards >= 2 and growth_leaders >= 1:
        return "risk_on_rotation"

    return "mixed_rotation"

@app.get("/asset-profile/{ticker}")
def get_asset_profile(ticker: str):
    symbol = ticker.strip().upper()

    if not symbol:
        raise HTTPException(status_code=400, detail="Ticker is required")

    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}

        name = (
            info.get("longName")
            or info.get("shortName")
            or info.get("displayName")
            or symbol
        )

        quote_type = info.get("quoteType", "EQUITY")

        sector = info.get("sector")
        industry = info.get("industry")

        # Fallbacks para assets que no tienen sector en Yahoo
        if not sector:
            if quote_type in ["ETF", "MUTUALFUND"]:
                sector = "ETF"
            elif quote_type in ["CRYPTOCURRENCY"]:
                sector = "Crypto"
            elif quote_type in ["INDEX"]:
                sector = "Index"
            elif quote_type in ["FUTURE"]:
                sector = "Futures"
            elif quote_type in ["CURRENCY"]:
                sector = "Currency"
            else:
                sector = "Unknown"

        return {
            "ticker": symbol,
            "name": name,
            "exchange": info.get("exchange", ""),
            "type": quote_type,
            "sector": sector,
            "industry": industry or "",
            "currency": info.get("currency", ""),
            "marketCap": info.get("marketCap")
        }

    except Exception as e:
        logger.error(f"Asset profile error for '{symbol}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Asset profile failed: {str(e)}"
        )

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



