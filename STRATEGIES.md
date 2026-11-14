# Strategies in the current backtester

The offline backtester has four built-in strategies. A **long** trade buys first and sells later. Each strategy decides using completed candles available at decision time. The backtester fills a market order at the next candle’s open and applies the configured spread, slippage, fees, and price rounding. [Execution rules](BACKTEST_CONTRACT.md#d-03--ausführung)

| Strategy | What it is based on | When it buys | When it sells |
| --- | --- | --- | --- |
| `single_long` | A one-entry rule for checking the simulator | Sends one buy signal at the first available candle close | At its stop, or at the end if the run is configured to close open positions |
| `sma_crossover` | Two moving averages of candle closes | The fast average crosses above the slow one while the strategy is flat | The fast average crosses below the slow one, or its stop triggers when risk-based sizing is configured |
| `two_legged_pullback` | A four-candle pattern with two drops separated by a bounce | The second drop makes a lower low and the pattern passes the configured thresholds | At its stop or profit target |
| `ridge_long` | A trained linear regression model and a fixed trading rule | The predicted return exceeds the entry threshold and the trade fits within the session when a session calendar is configured | At the planned holding horizon, with the market exit filled at the next available open |

Any position still open at the end can be closed by the backtest’s `liquidate_at_end` setting. [Strategy registry](src/market_trader/strategies.py) · [Strategy configuration reference](BACKTEST_CONTRACT.md#registrierte-strategien)

## Ridge

### What the model learns

Ridge learns a weighted sum from historical examples. Each training example pairs four numbers known at the decision time with the return that followed. The fit tries to reduce squared prediction errors and adds a penalty for large weights. That penalty is what makes it **Ridge regression**. This is a linear machine-learning model. The current implementation uses a fixed penalty of `0.1` and learns the weights and input scaling from the training rows. [Ridge regression reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) · [Repository implementation](src/market_trader/baseline.py)

The saved GER30 model uses eight completed 15-minute candles. From them it calculates:

| Input | What it measures |
| --- | --- |
| Last candle return | How much the close changed from the preceding candle’s close |
| Full window return | How much the close changed from the first close in the eight-candle window |
| Latest high–low range | The latest candle’s high minus its low, divided by its close |
| Close vs. window mean | How far the latest close is above or below the average close of the eight candles, as a fraction of that average |

For each input, the model subtracts the training average, divides by the training standard deviation, and multiplies by the learned weight. It adds those four contributions to a starting value called the **intercept**. The contribution bars in the report show how each input moved this particular forecast up or down. [Feature calculation](src/market_trader/training_data.py) · [Saved model and data splits](datasets/ger30-15m-2025-10-02-to-2026-02-13/training/DATASET.md)

### How the forecast becomes a trade

The model’s target is the gross 30-minute price return from the next candle’s open to the second future candle’s close. The trading rule buys one unit in the GER30 comparison if the forecast is **strictly above +1 basis point** (+0.01%), there is no open position, and the planned exit fits inside the session. It uses a timed exit and sends no stop or profit target. The simulated exit uses the following candle’s **open** at the 30-minute boundary. The label’s closing price and the simulated exit’s opening price can differ, and fills also include modeled costs. [Ridge trading rule](BACKTEST_CONTRACT.md#eingefrorenes-ridge-modell) · [GER30 run settings](datasets/ger30-15m-2026-02-17-to-03-06/DATASET.md#settings-fixed-before-the-runs)

`entry_threshold` and `quantity` are trading settings you can change while reusing the same model. Changing the inputs, lookback, target, or fitted weights calls for a new model fit.

### Worked example from the report

In **Ridge trade 7**, on 20 February 2026, the latest close was slightly above the eight-candle average. That input pulled the forecast down after scaling and weighting. The return across the full window made the largest positive contribution.

| Input | Value | Contribution |
| --- | ---: | ---: |
| Model intercept | — | +0.515 bp |
| Last candle return | +0.0416% | +0.155 bp |
| Full window return | +0.4594% | +1.597 bp |
| Latest high–low range | 0.1247% | −0.027 bp |
| Close vs. window mean | +0.1726% | −1.141 bp |
| **Forecast** | | **+1.099 bp** |

The forecast cleared +1 bp, so the strategy bought one unit at **12:30 Berlin time** for **25,254.33** and sold at **13:00** for **25,258.33**. The actual return on the model’s target was **+2.38 bp**. The move between the reference prices was +6.00 points, and modeled spread and slippage reduced that to **+4.00** in simulated P&L. The report shows the same trade at **11:30–12:00 UTC**. For a losing example, trade 3 forecast a larger rise (+1.96 bp) and ended at −17.52. The held-out forecast test found similar errors for Ridge and a forecast of zero. [Interactive report](datasets/ger30-15m-2026-02-17-to-03-06/ridge-vs-sma-report.html) · [Held-out evaluation](datasets/ger30-15m-2025-10-02-to-2026-02-13/training/DATASET.md#february-held-out-evaluation)

## The three rule-based strategies

### Single long

This is a simulator check. It submits a single buy on the first eligible decision, even if the market is falling, and makes no further entry attempts. `stop_distance` sets a percentage below that decision candle’s close. `risk_fraction` gives the simulator a budget for sizing the position from the planned stop loss and modeled costs. In the example, those settings are 2% and 1% respectively. A gap can make the realized loss exceed the planned risk. [Example config](configs/backtest-example.toml)

### SMA crossover

SMA means **simple moving average**. The strategy compares a short average of recent closes with a longer one. A buy requires the fast average to move from below or equal to the slow average to strictly above it. The reverse crossing triggers a sell if there is a long position. The strategy needs one more candle than the slow period so it can compare the current and previous averages.

The GER30 comparison used 1- and 2-candle averages and a fixed one-unit size. With those particular periods, the fast average is the latest close, and a crossover marks a change in the direction of consecutive closes, including transitions from a tie. This run had no stop and could hold overnight. The separate synthetic example uses a stop and risk-based sizing. [GER30 comparison settings](datasets/ger30-15m-2026-02-17-to-03-06/DATASET.md#settings-fixed-before-the-runs) · [Synthetic example config](configs/backtest-sma-example.toml)

### Two-legged pullback

This strategy looks for four consecutive candles. The first establishes a reference. The second drops to a lower low. The third bounces enough, closes higher, and stays between the first trough’s low and the reference high. The fourth makes a new lower low and closes far enough below the first candle’s close. If all conditions pass and there is no open position, it buys at the next open.

The stop is set below the second trough. The profit target uses a multiple of the distance from the signal candle’s close to that stop. The example sets a 0.3% minimum move, 0.2% minimum bounce, 0.2% stop buffer, and a target distance of twice the signal-to-stop distance. The actual reward and risk depend on where the next candle opens and on execution costs. [Detailed pattern rules](BACKTEST_CONTRACT.md#zweibeiniger-pullback) · [Example config](configs/backtest-pullback-example.toml)
