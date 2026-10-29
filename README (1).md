# Quantitative Trading Research Portfolio

A self-directed research program in quantitative trading: option pricing, volatility, risk, backtesting, market making and statistical arbitrage — built in Python and tested on real market data.

The emphasis throughout is **honest evaluation**. Every strategy is judged on data it was not fit to, after realistic trading costs. Most of them fail that test, and the write-ups say so.

---

## Key findings

| Question | Answer from the data |
|---|---|
| Does a rules-based Fibonacci + ADX pullback strategy have an edge? | **No.** −0.20R expectancy over 69 trades and 9.7 years; −1.5% a year vs. +15.6% for holding SPY. |
| Do parameter settings tuned on past data hold up? | **No.** The best in-sample settings (+0.13R) fell to −0.89R on unseen data — a textbook case of overfitting. |
| Are highly correlated stock pairs actually tied together? | **Mostly not.** Of 5 pairs with correlation ≥ 0.6, only Visa/Mastercard was cointegrated. Coca-Cola/Pepsi (p = 0.82) and Exxon/Chevron (p = 0.85) were not. |
| Does the one cointegrated pair keep working? | **No.** Out of sample, V/MA lost money and its cointegration weakened (p-value 0.013 → 0.182). |
| Does a 15-pair basket with a daily-updated (Kalman filter) hedge ratio fix that? | **It reduces risk but doesn't create an edge.** Volatility fell from 3.1% to 1.9%, but the basket returned −2.0% a year — roughly breakeven before about 2% a year in trading costs. |
| What decides market-making profitability? | **Adverse selection and inventory control.** In simulation, 5% informed order flow turned a strategy that was profitable in 100% of runs into one that lost in 97%. Volatility-adaptive spreads recovered about half the edge. |
| How fat are the tails? | **Much fatter than the bell curve.** SPY had 6 moves beyond 4 standard deviations in 5 years — 76 times more than a normal distribution predicts. |

---

## Projects

All notebooks run in Google Colab with **Runtime → Run all**. Replace `YOUR-USERNAME` in the links below with your GitHub username to make the Colab buttons work.

### 1. Expected value, edge and Monte Carlo simulation
[`Quant_Lesson_1_Expected_Value.ipynb`](Quant_Lesson_1_Expected_Value.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/quant-trading-portfolio/blob/main/Quant_Lesson_1_Expected_Value.ipynb)

Expected value, breakeven win rates, and trade statistics (Sharpe, drawdown) computed on a live trading log. A 10,000-path Monte Carlo shows how position size changes outcomes — and why sizing must survive being wrong about your own edge: at 20% risk per trade, a trader who believes they win 58% of the time but actually wins 42% has a 57% chance of losing 90% of the account. At 1% risk, the same trader finishes the year roughly flat.

### 2. Market data and volatility
[`Quant_Lesson_2_Volatility.ipynb`](Quant_Lesson_2_Volatility.ipynb)

Realized volatility, expected moves, drawdowns, and an implied-vs-realized volatility comparison. Example finding: SPY options priced at 18% implied volatility while SPY was realizing 9.8% — a measurable premium for option sellers.

### 3. Option pricing and the Greeks
[`Quant_Lesson_3_Options_Pricing.ipynb`](Quant_Lesson_3_Options_Pricing.ipynb)

Black-Scholes and all major Greeks built from scratch and verified with put-call parity. Locates 16-delta strikes, prices credit spreads, and visualizes theta acceleration and gamma near expiration.

### 4. Fat tails, correlation and beta
[`Quant_Lesson_4_Statistics.ipynb`](Quant_Lesson_4_Statistics.ipynb)

Measures how far real returns depart from the normal distribution, how correlations change on down days, and how beta converts a stock portfolio into market-equivalent exposure. AMZN showed the fattest tails of the tickers tested (7 moves beyond 4σ — 88× the normal prediction); NVDA's beta to SPY was 2.14.

### 5. Backtesting a discretionary setup — and overfitting
[`Quant_Lesson_5_Backtesting.ipynb`](Quant_Lesson_5_Backtesting.ipynb)

A Fibonacci golden-pocket entry filtered by ADX/DMI, formalized as exact rules and backtested over 10 years on SPY, QQQ, NVDA and AMZN with costs, gap-aware fills and no look-ahead bias.

| | Result |
|---|---|
| Trades | 69 |
| Win rate | 27.5% (breakeven ≈ 34%) |
| Average win / loss | +2.12R / −1.08R |
| Expectancy | **−0.20R** |
| Profit factor | 0.75 |
| CAGR | −1.5% (SPY: +15.6%) |

An in-sample / out-of-sample parameter sweep shows the overfitting trap directly: the best-tuned settings went from +0.134R to −0.885R on unseen data.

### 6. Market making with inventory and adverse selection
[`Quant_Lesson_6_Market_Making.ipynb`](Quant_Lesson_6_Market_Making.ipynb)

A simulated market maker built in four stages, each fixing the previous version's main failure. Results averaged over 30 simulated markets:

| Version | Avg 20-day P&L | Worst run | Profitable runs |
|---|---|---|---|
| Naive two-sided quoting | +$12,352 | −$96,867 | 80% |
| + Inventory skew and position limits | +$5,060 | +$3,265 | 100% |
| + 5% informed order flow | −$1,619 | −$4,932 | 3% |
| + Volatility-adaptive spreads | +$2,267 | −$131 | 97% |

Companion engine (Python): [`market_maker.py`](market_maker.py) (Avellaneda-Stoikov-style quoting with inventory skew and kill switches), [`venue.py`](venue.py) (order management with queue-position modelling), [`replay.py`](replay.py) (tick replay with markout P&L attribution), plus live data recorders for Coinbase ([`recorder.py`](recorder.py)) and Kalshi ([`kalshi_recorder.py`](kalshi_recorder.py)).

### 7. Pairs trading: correlation is not cointegration
[`Quant_Lesson_7_Pairs_Trading.ipynb`](Quant_Lesson_7_Pairs_Trading.ipynb)

Screens 45 stock pairs, filters for correlation, and applies the Engle-Granger cointegration test on a training window before trading the result on unseen data.

| Pair | Correlation | Cointegration p-value |
|---|---|---|
| V/MA | 0.88 | **0.013** |
| HD/LOW | 0.88 | 0.152 |
| KO/PEP | 0.75 | 0.821 |
| XOM/CVX | 0.85 | 0.846 |
| JPM/BAC | 0.80 | 0.987 |

V/MA out of sample: −0.4% a year, Sharpe −0.12, and the relationship weakened (p = 0.182). An earlier version of this notebook used a plain ADF test and found 20 of 45 pairs "cointegrated" — including Pepsi/Exxon. Switching to the correct Engle-Granger critical values removed the false positives; the correction is documented in the notebook.

### 8. Kalman-filter hedge ratios across a basket of pairs
[`Quant_Lesson_8_Rolling_Pairs_Basket.ipynb`](Quant_Lesson_8_Rolling_Pairs_Basket.ipynb)

Fifteen same-industry pairs chosen on economic grounds before any testing (avoiding the multiple-testing problem), each traded with a hedge ratio re-estimated daily by a Kalman filter, compared against a fixed hedge.

| | Rolling hedge | Fixed hedge | Hold SPY |
|---|---|---|---|
| Annual return | −2.0% | −2.3% | +16.9% |
| Sharpe | −1.05 | −0.74 | +1.03 |
| Volatility | **1.9%** | 3.1% | 16.5% |
| Worst drawdown | **−4.3%** | −5.1% | −18.8% |
| Correlation with SPY | 0.15 | 0.24 | — |

The adaptive hedge mattered most where relationships drifted — Merck/Pfizer went from −23.0% (fixed) to +13.5% (rolling) — but did not produce a portfolio edge. With roughly 18 trades per pair at about 0.2% per round trip, costs (≈2% a year) account for nearly the entire loss. A key practical finding: the filter's adaptation rate determines whether it works at all — set too fast, it absorbs the very spread it is meant to trade.

---

## Tools

| Tool | What it does |
|---|---|
| [`CSP_Scanner.ipynb`](CSP_Scanner.ipynb) | Pulls a live put chain and ranks cash-secured puts that pass a full checklist: assignment probability, annualized return, capital limit, liquidity and earnings date |
| [`Long_Option_Planner.ipynb`](Long_Option_Planner.ipynb) | Takes a directional thesis (target, stop, timeframe), prices every contract at the target and stop, sizes to 1% risk, and flags trades whose underlying reward-to-risk no contract can fix |

---

## What I learned

1. **Out-of-sample testing is the whole game.** Three strategies looked reasonable on the data they were built on. None survived unseen data.
2. **Costs decide thin edges.** The pairs basket was roughly breakeven before costs; the market maker's edge disappeared once fees matched the spread.
3. **Correlation is not a relationship.** Most highly correlated pairs were not cointegrated, and the one that was broke down.
4. **Position size matters more than entry.** Monte Carlo results show sizing determines survival when an estimated edge turns out to be wrong.
5. **Distributions have fat tails.** Probability estimates from Black-Scholes understate large moves, so strikes far from the money carry more risk than the model implies.

## What I'd build next

- Intraday pairs trading on a larger universe with lower-cost execution assumptions
- A volatility risk premium strategy with a regime filter on the VIX futures term structure
- GARCH volatility forecasting compared against implied volatility
- Live paper trading through a broker API to measure real fills against backtested fills

---

## Running the code

All notebooks run in [Google Colab](https://colab.research.google.com) with no local setup. To run locally:

```bash
pip install -r requirements.txt
```

Market data comes from Yahoo Finance via `yfinance` and is roughly 15 minutes delayed.

## Disclaimer

This repository is for research and education. Nothing here is investment advice, and past results — real or simulated — do not predict future performance.
