# ===========================================
# REBALANCEAI - FastAPI Backend
# Combines all Colab blocks into a deployable app
# ===========================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import warnings
warnings.filterwarnings('ignore')

# ── Core imports ──────────────────────────────────────────────────────────────
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import stats

# ── Sentiment ─────────────────────────────────────────────────────────────────
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

# ===========================================
# BLOCK 1: API Keys & Risk Profiles
# ===========================================

NEWS_API_KEY   = os.environ.get('NewsAPI')
OPENAI_API_KEY = os.environ.get('GPT')

RISK_PROFILES = {
    "conservative": {
        "max_single_stock": 0.15,
        "max_sector": 0.30,
        "min_stocks": 5,
        "beta_preference": "low",
        "sentiment_multiplier": 0.3,
        "risk_aversion": 5.0,
        "target_sharpe": 1.0,
    },
    "moderate": {
        "max_single_stock": 0.20,
        "max_sector": 0.40,
        "min_stocks": 5,
        "beta_preference": "neutral",
        "sentiment_multiplier": 0.5,
        "risk_aversion": 3.0,
        "target_sharpe": 1.2,
    },
    "aggressive": {
        "max_single_stock": 0.30,
        "max_sector": 0.50,
        "min_stocks": 3,
        "beta_preference": "high",
        "sentiment_multiplier": 0.8,
        "risk_aversion": 1.5,
        "target_sharpe": 1.5,
    }
}

# ===========================================
# BLOCK 2: Stock Analyzer
# ===========================================

vader_analyzer = SentimentIntensityAnalyzer()

TOXIC_KEYWORDS = [
    'lawsuit','fraud','scandal','investigation','miss','missed',
    'decline','loss','plunge','crash','warning','risk','threat',
    'downgrade','cut','concern','worry','fear','volatile','uncertain',
    'breach','hack','fail','failure','disappointing','weak'
]

def safe_div(a, b):
    try:
        if b is None or (isinstance(b, (int, float, np.floating)) and (b == 0 or not np.isfinite(b))):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def normalize_metric(value, ideal_direction='higher', min_val=None, max_val=None):
    if not np.isfinite(value):
        return 50.0
    if ideal_direction == 'lower':
        if min_val is not None and max_val is not None:
            return np.clip(100 * (1 - (value - min_val) / (max_val - min_val)), 0, 100)
        return np.clip(100 / (1 + abs(value)), 0, 100)
    if min_val is not None and max_val is not None:
        return np.clip(100 * (value - min_val) / (max_val - min_val), 0, 100)
    return np.clip(50 + 50 * np.tanh(value), 0, 100)

def latest(s):
    try:
        return float(pd.Series(s).dropna().iloc[-1])
    except Exception:
        return np.nan

def col(df, key):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    candidates = [key, key.replace("_"," "), key.replace(" ","_"),
                  key.replace("-"," "), key.upper(), key.lower()]
    for k in candidates:
        if k in df.index:   return df.loc[k]
        if k in df.columns: return df[k]
    for idx in df.index:
        if str(idx).lower() == key.lower():
            return df.loc[idx]
    return pd.Series(dtype=float)

def sum_last_quarters(df, row_name, q=4):
    s = col(df, row_name)
    if isinstance(s, pd.Series) and not s.empty:
        return float(s.dropna().iloc[:q].sum())
    return np.nan

def latest_point(df, row_name):
    s = col(df, row_name)
    if isinstance(s, pd.Series) and not s.empty:
        return float(s.dropna().iloc[0])
    return np.nan

def analyze_stock(ticker, days_lookback=14, verbose=False):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2y", interval="1d", auto_adjust=False).dropna()
        if hist.empty:
            raise ValueError(f"No price history for {ticker}")

        info         = dict(getattr(t, "info", {}) or {})
        company_name = info.get('longName', ticker)
        sector       = info.get('sector', 'Unknown')
        price        = info.get('currentPrice') or float(hist["Close"].iloc[-1])
        market_cap   = info.get('marketCap')
        shares_out   = info.get('sharesOutstanding')

        inc_q = t.quarterly_income_stmt
        bs_q  = t.quarterly_balance_sheet
        cf_q  = t.quarterly_cashflow

        Revenue_TTM  = sum_last_quarters(inc_q, "Total Revenue")
        NetInc_TTM   = sum_last_quarters(inc_q, "Net Income Common Stockholders")
        OCF_TTM      = sum_last_quarters(cf_q,  "Total Cash From Operating Activities")
        Capex_TTM    = sum_last_quarters(cf_q,  "Capital Expenditures")
        if not np.isfinite(Capex_TTM):
            Capex_TTM = sum_last_quarters(cf_q, "Capital Expenditure")

        TotalAssets  = latest_point(bs_q, "Total Assets")
        Equity       = latest_point(bs_q, "Total Stockholder Equity")
        CurrentAssets= latest_point(bs_q, "Total Current Assets")
        CurrentLiab  = latest_point(bs_q, "Total Current Liabilities")
        TotalDebt    = latest_point(bs_q, "Total Debt")

        PE             = safe_div(price, safe_div(NetInc_TTM, shares_out)) if shares_out else info.get('trailingPE')
        ROE            = safe_div(NetInc_TTM, Equity)
        ROA            = safe_div(NetInc_TTM, TotalAssets)
        DebtToEquity   = safe_div(TotalDebt, Equity)
        CurrentRatio   = safe_div(CurrentAssets, CurrentLiab)

        close = hist["Close"].copy()
        ret   = close.pct_change()

        def rsi(series, n=14):
            delta    = series.diff()
            up       = delta.clip(lower=0.0)
            down     = -delta.clip(upper=0.0)
            roll_up  = up.ewm(alpha=1/n, adjust=False).mean()
            roll_dn  = down.ewm(alpha=1/n, adjust=False).mean()
            rs       = roll_up / roll_dn
            return 100 - (100 / (1 + rs))

        RSI14            = rsi(close, 14)
        Volatility20_Ann = ret.rolling(20).std() * np.sqrt(252)
        Sharpe_252       = (ret.rolling(252).mean() / ret.rolling(252).std()) * np.sqrt(252)

        try:
            spx_data  = yf.download("^GSPC", period="2y", interval="1d", progress=False, auto_adjust=False)
            spx_close = spx_data["Close"].squeeze().dropna()
            stk_ret   = close.pct_change()
            spx_ret   = spx_close.pct_change()
            if hasattr(stk_ret.index, 'tz') and stk_ret.index.tz is not None:
                stk_ret.index = stk_ret.index.tz_localize(None)
            if hasattr(spx_ret.index, 'tz') and spx_ret.index.tz is not None:
                spx_ret.index = spx_ret.index.tz_localize(None)
            ret_df = pd.DataFrame({"stock": stk_ret, "spx": spx_ret}).dropna()
            if len(ret_df) >= 60:
                cov_mat = ret_df.cov()
                Beta    = cov_mat.loc["stock","spx"] / cov_mat.loc["spx","spx"]
            else:
                Beta = info.get('beta', np.nan)
        except:
            Beta = info.get('beta', np.nan)

        cummax   = close.cummax()
        Drawdown = (cummax - close) / cummax

        latest_rsi        = latest(RSI14)
        latest_volatility = latest(Volatility20_Ann)
        latest_sharpe     = latest(Sharpe_252)
        latest_drawdown   = latest(Drawdown)

        # Sentiment
        sentiment_score   = None
        sentiment_metrics = {}
        try:
            query     = f"{ticker} stock OR {company_name}"
            from_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
            url       = "https://newsapi.org/v2/everything"
            params    = {'q': query,'from': from_date,'language':'en',
                         'sortBy':'publishedAt','pageSize':100,'apiKey': NEWS_API_KEY}
            response  = requests.get(url, params=params)
            data      = response.json()

            if data['status'] == 'ok' and len(data['articles']) > 0:
                processed = []
                for article in data['articles']:
                    text        = f"{article.get('title','')} {article.get('description','')}"
                    vader_score = vader_analyzer.polarity_scores(text)['compound']
                    blob_score  = TextBlob(text).sentiment.polarity
                    sentiment   = 0.7 * vader_score + 0.3 * blob_score
                    text_lower  = text.lower()
                    toxic_count = sum(1 for kw in TOXIC_KEYWORDS if kw in text_lower)
                    word_count  = len(text.split())
                    toxicity    = (toxic_count / word_count) * 100 if word_count > 0 else 0
                    processed.append({'sentiment_score': sentiment, 'toxicity': toxicity,
                                      'published_at': article.get('publishedAt','')})

                df_articles             = pd.DataFrame(processed)
                df_articles['published_at'] = pd.to_datetime(df_articles['published_at'], utc=True)
                avg_sentiment           = float(df_articles['sentiment_score'].mean())
                avg_toxicity            = float(df_articles['toxicity'].mean())
                attention               = len(df_articles)
                cutoff                  = pd.Timestamp(datetime.now() - timedelta(days=7)).tz_localize('UTC')
                current_week            = df_articles[df_articles['published_at'] >= cutoff]
                prior_cutoff            = cutoff - pd.Timedelta(days=7)
                prior_week              = df_articles[(df_articles['published_at'] >= prior_cutoff) &
                                                      (df_articles['published_at'] < cutoff)]
                current_avg             = float(current_week['sentiment_score'].mean()) if len(current_week) > 0 else 0
                prior_avg               = float(prior_week['sentiment_score'].mean())   if len(prior_week)   > 0 else 0
                sentiment_momentum      = current_avg - prior_avg
                sentiment_score         = (
                    normalize_metric(avg_sentiment,       'higher', -0.3, 0.3) * 0.40 +
                    normalize_metric(sentiment_momentum,  'higher', -0.2, 0.2) * 0.30 +
                    normalize_metric(avg_toxicity,        'lower',   0,   5)   * 0.20 +
                    normalize_metric(attention,           'higher',  10, 100)  * 0.10
                )
                sentiment_metrics = {'avg_sentiment': avg_sentiment,
                                     'sentiment_momentum': sentiment_momentum,
                                     'toxicity': avg_toxicity, 'attention': attention}
        except:
            sentiment_score = 50.0

        fund_scores = {
            'ROE':          normalize_metric(ROE,          'higher', 0.05, 0.30) * 0.25,
            'ROA':          normalize_metric(ROA,          'higher', 0.02, 0.15) * 0.15,
            'PE':           normalize_metric(PE,           'lower',  10,   40)   * 0.20,
            'DebtToEquity': normalize_metric(DebtToEquity, 'lower',  0,    2)    * 0.20,
            'CurrentRatio': normalize_metric(CurrentRatio, 'higher', 1.0,  3.0)  * 0.20,
        }
        fundamental_score = sum(fund_scores.values())

        tech_scores = {
            'Sharpe':     normalize_metric(latest_sharpe,              'higher', 0.5, 2.0)  * 0.35,
            'RSI':        normalize_metric(abs(latest_rsi - 50),       'lower',  0,   30)   * 0.25,
            'Volatility': normalize_metric(latest_volatility,          'lower',  0.15,0.50) * 0.25,
            'Drawdown':   normalize_metric(latest_drawdown,            'lower',  0,   0.30) * 0.15,
        }
        technical_score = sum(tech_scores.values())

        if sentiment_score is not None:
            alpha_score = 0.50 * fundamental_score + 0.30 * technical_score + 0.20 * sentiment_score
        else:
            alpha_score = 0.60 * fundamental_score + 0.40 * technical_score

        risk_flags = []
        if DebtToEquity > 1.5:             risk_flags.append("High debt-to-equity ratio")
        if CurrentRatio < 1.5:             risk_flags.append("Low liquidity (current ratio)")
        if latest_volatility > 0.40:       risk_flags.append("High volatility")
        if latest_rsi > 70:                risk_flags.append("Overbought (RSI > 70)")
        elif latest_rsi < 30:              risk_flags.append("Oversold (RSI < 30)")
        if latest_drawdown > 0.15:         risk_flags.append("Significant drawdown from peak")
        if sentiment_metrics.get('toxicity', 0) > 3.0:
            risk_flags.append("High negative news coverage")

        return {
            'ticker': ticker, 'company_name': company_name, 'sector': sector,
            'price': price, 'market_cap': market_cap,
            'alpha_score': alpha_score, 'fundamental_score': fundamental_score,
            'technical_score': technical_score,
            'sentiment_score': sentiment_score if sentiment_score else 50.0,
            'expected_return': latest_sharpe * latest_volatility if np.isfinite(latest_sharpe) and np.isfinite(latest_volatility) else 0.10,
            'volatility': latest_volatility if np.isfinite(latest_volatility) else 0.25,
            'sharpe':     latest_sharpe     if np.isfinite(latest_sharpe)     else 1.0,
            'beta':       Beta              if np.isfinite(Beta)              else 1.0,
            'PE':         PE                if np.isfinite(PE)                else np.nan,
            'ROE':        ROE               if np.isfinite(ROE)               else np.nan,
            'ROA':        ROA               if np.isfinite(ROA)               else np.nan,
            'debt_to_equity': DebtToEquity  if np.isfinite(DebtToEquity)      else np.nan,
            'current_ratio':  CurrentRatio  if np.isfinite(CurrentRatio)      else np.nan,
            'rsi':        latest_rsi        if np.isfinite(latest_rsi)        else 50,
            'drawdown':   latest_drawdown   if np.isfinite(latest_drawdown)   else 0,
            'sentiment_metrics': sentiment_metrics, 'risk_flags': risk_flags,
            'returns_series': ret.dropna().tail(252).values,
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None

# ===========================================
# BLOCK 3: Batch Analyzer
# ===========================================

def validate_and_fix_ticker(ticker):
    corrections = {'FB': 'META', 'GOOG': 'GOOGL', 'BRK.B': 'BRK-B'}
    t = ticker.upper().strip()
    return corrections.get(t, t)

def batch_analyze_stocks(tickers, verbose_individual=False):
    results, failed = [], []
    for ticker in tickers:
        ticker = validate_and_fix_ticker(ticker)
        try:
            result = analyze_stock(ticker, verbose=verbose_individual)
            if result is not None:
                results.append(result)
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values('alpha_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df

def get_sp500_top_stocks(n=50):
    top_50 = [
        'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','LLY','V',
        'UNH','XOM','WMT','JPM','MA','JNJ','PG','AVGO','HD','CVX',
        'MRK','ABBV','COST','PEP','KO','ADBE','CRM','NFLX','TMO','MCD',
        'ACN','CSCO','ABT','NKE','DHR','TXN','WFC','DIS','VZ','CMCSA',
        'NEE','PM','ORCL','INTC','BMY','QCOM','UPS','RTX','HON','AMD'
    ]
    return top_50[:n]

# ===========================================
# BLOCK 4: Portfolio Optimizer
# ===========================================

def optimize_portfolio(stocks_df, risk_profile='moderate', portfolio_value=10000, preferences=None):
    if risk_profile not in RISK_PROFILES:
        risk_profile = 'moderate'
    profile     = RISK_PROFILES[risk_profile].copy()
    preferences = preferences or {}
    df          = stocks_df.copy()

    min_alpha = preferences.get('min_alpha', 40.0)
    df = df[df['alpha_score'] >= min_alpha]

    if preferences.get('specific_tickers'):
        df = df[df['ticker'].isin(preferences['specific_tickers'])]
    if preferences.get('exclude_tickers'):
        df = df[~df['ticker'].isin(preferences['exclude_tickers'])]
    if preferences.get('sector_preference'):
        df.loc[df['sector'] == preferences['sector_preference'], 'alpha_score'] *= 1.2
    if profile['beta_preference'] == 'low':
        df = df[df['beta'] <= 1.2]

    if preferences.get('specific_tickers') and len(df) < profile['min_stocks']:
        profile['min_stocks'] = max(2, len(df))
    elif len(df) < profile['min_stocks']:
        df = stocks_df[stocks_df['alpha_score'] >= 30.0]
        if len(df) < profile['min_stocks']:
            return None

    df       = df.reset_index(drop=True)
    n_stocks = len(df)

    expected_returns = df['alpha_score'].values / 100.0
    if 'expected_return' in df.columns:
        expected_returns = 0.5 * expected_returns + 0.5 * df['expected_return'].values
    expected_returns *= 0.15

    volatilities  = np.where(np.isfinite(df['volatility'].values), df['volatility'].values, 0.25)
    returns_data  = []
    for idx, row in df.iterrows():
        rs = row.get('returns_series', None)
        returns_data.append(rs if rs is not None and len(rs) > 0
                            else np.random.randn(252) * volatilities[idx] / np.sqrt(252))

    min_len        = min(len(r) for r in returns_data)
    returns_matrix = np.array([r[:min_len] for r in returns_data])
    if returns_matrix.shape[1] > 1:
        corr_matrix = np.corrcoef(returns_matrix)
        corr_matrix = np.where(np.isfinite(corr_matrix), corr_matrix, 0)
        np.fill_diagonal(corr_matrix, 1.0)
    else:
        corr_matrix = np.eye(n_stocks)

    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

    def neg_sharpe(w):
        ret = np.dot(w, expected_returns)
        vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        return -((ret - 0.04) / vol) if vol > 0 else 0

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    sectors     = df['sector'].values
    for sector in df['sector'].unique():
        mask = (sectors == sector)
        max_w = profile['max_sector']
        constraints.append({'type': 'ineq', 'fun': lambda w, m=mask, mx=max_w: mx - np.sum(w[m])})

    bounds     = [(0.0, profile['max_single_stock'])] * n_stocks
    opt_result = minimize(neg_sharpe, np.ones(n_stocks)/n_stocks, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-9})

    w = opt_result.x
    w[w < 0.01] = 0
    w /= w.sum()

    if np.sum(w > 0.02) < profile['min_stocks']:
        n_inc      = min(max(profile['min_stocks'], int(n_stocks*0.4)), n_stocks)
        top_idx    = np.argsort(df['alpha_score'].values)[-n_inc:]
        new_w      = np.zeros(n_stocks)
        new_w[top_idx] = w[top_idx]
        if np.sum(new_w > 0) < profile['min_stocks']:
            new_w = np.zeros(n_stocks)
            new_w[top_idx[:profile['min_stocks']]] = 1.0 / profile['min_stocks']
        else:
            new_w /= new_w.sum()
        w = new_w

    final_ret  = np.dot(w, expected_returns)
    final_vol  = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    final_shp  = (final_ret - 0.04) / final_vol if final_vol > 0 else 0
    final_beta = np.dot(w, df['beta'].values)
    final_alp  = np.dot(w, df['alpha_score'].values)

    allocation_details = sorted([
        {'ticker': df.iloc[i]['ticker'], 'company_name': df.iloc[i]['company_name'],
         'sector': df.iloc[i]['sector'], 'percentage': w[i]*100,
         'dollar_amount': w[i]*portfolio_value, 'alpha_score': df.iloc[i]['alpha_score'],
         'beta': df.iloc[i]['beta'], 'sharpe': df.iloc[i]['sharpe'],
         'risk_flags': df.iloc[i]['risk_flags']}
        for i, weight in enumerate(w) if weight > 0.001
    ], key=lambda x: x['percentage'], reverse=True)

    sector_allocation = {}
    for d in allocation_details:
        sector_allocation[d['sector']] = sector_allocation.get(d['sector'], 0) + d['percentage']

    return {
        'allocations': {d['ticker']: d['dollar_amount'] for d in allocation_details},
        'percentages': {d['ticker']: d['percentage']    for d in allocation_details},
        'allocation_details': allocation_details,
        'portfolio_metrics': {
            'expected_return': final_ret, 'volatility': final_vol,
            'sharpe_ratio': final_shp, 'beta': final_beta,
            'weighted_alpha': final_alp, 'n_positions': len(allocation_details),
            'sector_allocation': sector_allocation,
        },
        'risk_profile': risk_profile,
        'portfolio_value': portfolio_value,
        'stocks_dataframe': df,
    }

# ===========================================
# BLOCK 8: Monte Carlo Simulation
# ===========================================

def monte_carlo_simulation(portfolio_result, days_to_simulate=252,
                            num_simulations=5000, use_realistic_adjustments=True):
    portfolio_value    = portfolio_result['portfolio_value']
    allocation_details = portfolio_result['allocation_details']
    tickers  = [d['ticker']          for d in allocation_details]
    weights  = np.array([d['percentage']/100 for d in allocation_details])

    returns_data = []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period='5y')
            if hist.empty:
                returns_data.append(pd.Series([0.0004]*252))
                continue
            returns_data.append(hist['Close'].pct_change().dropna())
        except:
            returns_data.append(pd.Series([0.0004]*252))

    mean_returns = np.array([r.mean() for r in returns_data])
    if use_realistic_adjustments:
        mean_returns = mean_returns * 0.6 + (0.10/252) * 0.4

    min_len        = min(len(r) for r in returns_data)
    returns_matrix = pd.DataFrame({t: returns_data[i].iloc[-min_len:].values
                                   for i, t in enumerate(tickers)})
    cov_matrix     = returns_matrix.cov().values

    portfolio_return   = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol      = np.sqrt(portfolio_variance)

    all_paths       = np.zeros((num_simulations, days_to_simulate + 1))
    all_paths[:, 0] = portfolio_value
    L               = np.linalg.cholesky(cov_matrix)

    for sim in range(num_simulations):
        for day in range(1, days_to_simulate + 1):
            rnd         = np.random.normal(0, 1, len(tickers))
            corr_ret    = mean_returns + np.dot(L, rnd)
            if np.random.random() < 0.005:
                corr_ret += np.random.uniform(-0.05, -0.02)
            port_ret    = np.dot(weights, corr_ret)
            all_paths[sim, day] = all_paths[sim, day-1] * (1 + port_ret)

    final_values    = all_paths[:, -1]
    confidence_lvls = [0.05, 0.50, 0.95]
    percentiles     = {c: np.percentile(final_values, c*100) for c in confidence_lvls}

    prob_loss    = np.sum(final_values < portfolio_value) / num_simulations * 100
    prob_gain_20 = np.sum(final_values > portfolio_value*1.2) / num_simulations * 100
    expected_val = np.mean(final_values)
    exp_return   = (expected_val - portfolio_value) / portfolio_value * 100
    annual_vol   = portfolio_vol * np.sqrt(252) * 100

    if prob_loss < 10:
        risk_level     = "🟢 LOW RISK"
        interpretation = "Very low probability of loss. Portfolio has strong downside protection."
    elif prob_loss < 25:
        risk_level     = "🟡 MODERATE RISK"
        interpretation = "Some downside risk, but portfolio is well-positioned for positive returns."
    else:
        risk_level     = "🔴 HIGH RISK"
        interpretation = "Significant probability of loss. Consider a more conservative allocation."

    return {
        'expected_value': expected_val, 'expected_return': exp_return,
        'percentiles': percentiles, 'probability_of_loss': prob_loss,
        'probability_gain_20': prob_gain_20, 'risk_level': risk_level,
        'interpretation': interpretation,
    }

# ===========================================
# BLOCK 5: LangChain Tools & Agent
# ===========================================

def lookup_ticker_tool(company_name: str) -> str:
    mapping = {
        'apple':'AAPL','microsoft':'MSFT','google':'GOOGL','alphabet':'GOOGL',
        'amazon':'AMZN','meta':'META','nvidia':'NVDA','facebook':'META',
        'tesla':'TSLA','netflix':'NFLX','adobe':'ADBE','salesforce':'CRM',
        'oracle':'ORCL','intel':'INTC','amd':'AMD','cisco':'CSCO',
        'jpmorgan':'JPM','jp morgan':'JPM','chase':'JPM',
        'bank of america':'BAC','wells fargo':'WFC','goldman sachs':'GS',
        'morgan stanley':'MS','visa':'V','mastercard':'MA',
        'american express':'AXP','berkshire':'BRK-B','berkshire hathaway':'BRK-B',
        'walmart':'WMT','costco':'COST','target':'TGT','home depot':'HD',
        'mcdonalds':'MCD',"mcdonald's":'MCD','starbucks':'SBUX','nike':'NKE',
        'coca cola':'KO','coke':'KO','pepsi':'PEP','disney':'DIS',
        'johnson and johnson':'JNJ','johnson & johnson':'JNJ','pfizer':'PFE',
        'merck':'MRK','abbvie':'ABBV','united health':'UNH','eli lilly':'LLY',
        'exxon':'XOM','chevron':'CVX','ups':'UPS','fedex':'FDX','boeing':'BA',
    }
    n = company_name.lower().strip()
    if n in mapping: return mapping[n]
    for k, v in mapping.items():
        if k in n or n in k: return v
    return f"Could not find ticker for '{company_name}'. Please use ticker symbol."

def analyze_single_stock_tool(ticker: str) -> str:
    result = analyze_stock(ticker, verbose=False)
    if result is None:
        return f"Error: Could not analyze {ticker}"
    pe_str  = f"{result['PE']:.2f}"  if np.isfinite(result['PE'])  else 'N/A'
    roe_str = f"{result['ROE']*100:.2f}%" if np.isfinite(result['ROE']) else 'N/A'
    alpha   = result['alpha_score']
    if   alpha >= 70: rec, why = "🟢 STRONG BUY", "Excellent fundamentals, technicals, and sentiment align."
    elif alpha >= 60: rec, why = "🟢 BUY",         "Strong indicators across multiple dimensions."
    elif alpha >= 50: rec, why = "🟡 HOLD",         "Mixed signals. Monitor closely."
    elif alpha >= 40: rec, why = "🔴 SELL",         "Weak performance indicators."
    else:             rec, why = "🔴 STRONG SELL",  "Poor fundamentals, technicals, or sentiment."
    if result['risk_flags'] and 'High volatility' in result['risk_flags']:
        rec += " (⚠️ HIGH VOLATILITY)"
    return (f"Stock Analysis: {result['ticker']} - {result['company_name']}\n"
            f"Sector: {result['sector']} | Price: ${result['price']:.2f}\n\n"
            f"🎯 ALPHA SCORE: {alpha:.2f}/100\n"
            f"   • Fundamental: {result['fundamental_score']:.2f}/100\n"
            f"   • Technical:   {result['technical_score']:.2f}/100\n"
            f"   • Sentiment:   {result['sentiment_score']:.2f}/100\n\n"
            f"📊 RECOMMENDATION: {rec}\n💡 Why: {why}\n\n"
            f"Key Metrics:\n"
            f"   • P/E: {pe_str} | ROE: {roe_str} | Beta: {result['beta']:.2f}\n"
            f"   • Sharpe: {result['sharpe']:.2f} | Volatility: {result['volatility']*100:.2f}%\n\n"
            f"Risk Flags: {', '.join(result['risk_flags']) if result['risk_flags'] else 'None'}")

def compare_stocks_tool(tickers_str: str) -> str:
    tickers = [t.strip().upper() for t in tickers_str.strip().strip("'\"").split(',')]
    if len(tickers) < 2:
        return "Error: Need at least 2 tickers. Use format: 'AAPL,MSFT'"
    results = [r for t in tickers if (r := analyze_stock(t, verbose=False)) is not None]
    if len(results) < 2:
        return "Error: Could not analyze enough stocks"
    out = f"\n📊 STOCK COMPARISON ({len(results)} stocks)\n{'='*60}\n\n"
    for r in results:
        a   = r['alpha_score']
        rec = ("🟢 STRONG BUY" if a>=70 else "🟢 BUY" if a>=60 else
               "🟡 HOLD" if a>=50 else "🔴 SELL" if a>=40 else "🔴 STRONG SELL")
        out += (f"{r['ticker']:<6} | {r['company_name'][:30]:<30} | Rec: {rec}\n"
                f"       Alpha: {a:>5.1f} | Beta: {r['beta']:.2f} | Sharpe: {r['sharpe']:.2f}\n\n")
    best = max(results, key=lambda x: x['alpha_score'])
    out += f"🏆 Highest Alpha: {best['ticker']} ({best['alpha_score']:.1f}/100)"
    return out

def build_portfolio_tool(input_str: str) -> str:
    parts = input_str.strip().strip("'\"").split('|')
    if len(parts) < 2:
        return "Error: Format should be 'risk_profile|portfolio_value'"
    risk_profile = parts[0].strip().lower()
    try:
        portfolio_value = float(parts[1].strip().replace(',',''))
    except:
        return "Error: Portfolio value must be a number"

    preferences = {}
    if len(parts) > 2:
        pref_str = parts[2].strip().strip("'\"")
        if ':' in pref_str:
            key, value = pref_str.split(':', 1)
            key, value = key.strip(), value.strip()
            if key == 'sector_preference':
                preferences['sector_preference'] = value
            elif key == 'specific_tickers':
                preferences['specific_tickers'] = [validate_and_fix_ticker(t.strip()) for t in value.split(',')]
            elif key == 'exclude_tickers':
                preferences['exclude_tickers'] = [validate_and_fix_ticker(t.strip()) for t in value.split(',')]

    tickers = preferences.get('specific_tickers', get_sp500_top_stocks(50))
    df      = batch_analyze_stocks(tickers)
    if df.empty:
        return "Error: Could not analyze any stocks"

    result = optimize_portfolio(df, risk_profile, portfolio_value, preferences)
    if result is None:
        return "Error: Could not optimize portfolio"

    metrics = result['portfolio_metrics']
    out  = (f"\n💼 PORTFOLIO RECOMMENDATION\n{'='*60}\n"
            f"Risk Profile: {risk_profile.upper()} | Value: ${portfolio_value:,.0f}\n\n"
            f"📊 Portfolio Metrics:\n"
            f"   • Expected Return: {metrics['expected_return']*100:.2f}%\n"
            f"   • Volatility:      {metrics['volatility']*100:.2f}%\n"
            f"   • Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}\n"
            f"   • Beta:            {metrics['beta']:.2f}\n"
            f"   • Weighted Alpha:  {metrics['weighted_alpha']:.2f}/100\n"
            f"   • Positions:       {metrics['n_positions']}\n\n"
            f"💰 Allocation:\n")
    for d in result['allocation_details']:
        out += f"   {d['ticker']:<6} ${d['dollar_amount']:>8,.0f} ({d['percentage']:>5.2f}%) | {d['company_name'][:25]}\n"
    out += "\n🏢 Sector Breakdown:\n"
    for s, p in metrics['sector_allocation'].items():
        out += f"   {s:<30} {p:>6.2f}%\n"

    try:
        mc = monte_carlo_simulation(result, days_to_simulate=252, num_simulations=5000)
        out += (f"\n\n🎲 PROBABILISTIC FORECAST (1 Year):\n"
                f"   Expected Value:      ${mc['expected_value']:>12,.0f}\n"
                f"   Expected Return:     {mc['expected_return']:>12.2f}%\n"
                f"   Worst Case (5%):     ${mc['percentiles'][0.05]:>12,.0f}\n"
                f"   Best Case (95%):     ${mc['percentiles'][0.95]:>12,.0f}\n"
                f"   Probability of Loss: {mc['probability_of_loss']:>12.2f}%\n"
                f"   Prob of 20%+ Gain:   {mc['probability_gain_20']:>12.2f}%\n"
                f"   Risk Level:          {mc['risk_level']}\n"
                f"\n💡 {mc['interpretation']}")
    except Exception as e:
        out += f"\n\n⚠️ Monte Carlo unavailable: {e}"
    return out

# ── Initialize Agent ──────────────────────────────────────────────────────────
llm    = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

tools = [
    Tool(name="LookupTicker",   func=lookup_ticker_tool,
         description="Find stock ticker from company name. Input: 'Tesla' → Returns: 'TSLA'"),
    Tool(name="AnalyzeStock",   func=analyze_single_stock_tool,
         description="Analyze single stock with BUY/SELL/HOLD recommendation. Input: 'AAPL'"),
    Tool(name="CompareStocks",  func=compare_stocks_tool,
         description="Compare stocks side-by-side. Input: 'AAPL,MSFT,GOOGL'"),
    Tool(name="BuildPortfolio", func=build_portfolio_tool,
         description=("Build optimized portfolio with Monte Carlo risk analysis. "
                       "Format: 'risk_profile|portfolio_value|preferences'. "
                       "Examples: 'moderate|10000' or 'aggressive|50000|sector_preference:Technology'")),
]

agent = initialize_agent(tools=tools, llm=llm,
                         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                         memory=memory, verbose=False,
                         max_iterations=5, early_stopping_method="generate")

# ===========================================
# FastAPI App
# ===========================================

app = FastAPI(title="RebalanceAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = agent.run(req.message)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error processing request: {str(e)}"}