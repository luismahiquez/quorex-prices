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
            missing_data.append("Relevant News")

        headline_texts = [n.title for n in raw_news]
        sentiment, _ = ctx_news_sentiment(headline_texts)

        asset_news = [n for n in raw_news if n.category == "asset"]

        if asset_news:
            catalyst = asset_news[0].title
        else:
            catalyst = "No strong asset-specific catalyst detected; current news is mostly sector or macro related."

            if raw_news:
                missing_data.append("Asset-specific News")

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



