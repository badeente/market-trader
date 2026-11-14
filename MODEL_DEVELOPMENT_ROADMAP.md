# Neural model development roadmap

**Goal.** Train a neural network on tabular market data and a second one on chart images. For each run, be able to see what the model predicted, inspect why the strategy acted, change a specific behavior, and measure what changed against a baseline.

## Where we are

The [strategy guide](STRATEGIES.md) explains the four built-in strategies and walks through a Ridge trade from the report.

The offline backtester, chronological data splits, and Ridge baseline work. On the February forecast test, Ridge and a zero-return forecast had nearly identical errors. Ridge made a small simulated profit on a separate 14-session run. The report command now generates interactive HTML from saved runs. Both neural models are still to build. [Report command](TESTING.md#wiederverwendbarer-html-bericht) · [Forecast evaluation](datasets/ger30-15m-2025-10-02-to-2026-02-13/training/DATASET.md#february-held-out-evaluation) · [Backtest results](datasets/ger30-15m-2026-02-17-to-03-06/DATASET.md#results)

## Milestones

| Stage | What we build | What you can review |
| --- | --- | --- |
| **1. Reusable run report · implemented** | The report command reads saved results. It shows forecasts versus actual returns, baseline errors, after-cost equity and drawdown, and a trade detail view with entry, exit, costs, and P&L. It shows Ridge inputs and forecasts when the model and dataset are supplied. | Open a saved report, inspect a winning trade and a losing trade, and trace each to the prediction, entry rule, and fills. Regenerate the report from the saved inputs. |
| **2. Define the first experiment** | Fix the target, timestamp rules, data splits, baselines, and comparison metrics. Use 30-minute GER30 return as the proposed first target. Export timestamped forecasts from a frozen model so the same backtester and report can read them. | Confirm what the model predicts and when a trade could happen. Check a few labeled candles by hand and replay saved Ridge forecasts through the matching strategy. |
| **3. Train a tabular neural net** | Train a small FastAI `tabular_learner` in a pinned training environment. Start with the same four features as Ridge. Save the weights, preprocessing, settings, and forecasts. Compare both models on the same rows and with the same trading rule. | Compare the first neural run against Ridge and the zero-return forecast. Review the training and validation curves, then reproduce the run from its saved settings. Treat the first run as a pipeline check, regardless of which model wins. |
| **4. Tune one behavior at a time** | Turn your observations into hypotheses. Compare each variant with its parent on the same validation dates. Record the intended change, metric, trade-offs, and the decision to keep, revise, or drop it. | Pick a behavior, see which trades changed, and decide whether the result is closer to what you wanted. |
| **5. Train a chart-image neural net** | Render reproducible candlestick images from the same decision windows. Train a FastAI `vision_learner` with the same target and dates as the tabular model. If the intended input is platform screenshots, also evaluate timestamped screenshots from that platform. | Open the exact image the network saw for an individual prediction, then compare both neural models in the same report and backtester. |
| **6. Evaluate a frozen candidate** | Freeze the chosen model and trading settings, then run once on a fresh, previously unseen period. Compare against the agreed baselines and record the conclusion. | Decide whether the improvement holds up across sessions, after costs, and with enough trades to interpret the result. |

The FastAI trainers can live in this repository with their own dependencies. Start with a small, reproducible run and keep the backtester on its current environment. [FastAI tabular learner](https://docs.fast.ai/tabular.learner.html) · [FastAI vision learner](https://docs.fast.ai/vision.learner.html)

## How you steer the model

| What you observe or want | First change to try | What we inspect |
| --- | --- | --- |
| “It trades too often on tiny predicted moves.” | Raise the entry threshold on the same saved predictions. This changes the trading rule without retraining. | Trade count, forecast bins, average net P&L per trade, and after-cost return. |
| “It seems to miss the context before a reversal.” | Compare the current inputs with one added feature or a longer history window. Retrain with the same target. | Forecast errors on the same eligible timestamps, plus matched examples of reversals and ordinary sessions. |
| “I want it to find moves big enough to cover costs.” | Compare a higher entry threshold first. If the forecast still answers the wrong question, try a target such as the probability that a 30-minute move exceeds a specified cost. | Whether the predicted probabilities match observed frequencies, and which new or skipped trades improve the after-cost result. |
| “It fits the training data but falls apart on later dates.” | Compare a smaller network, stronger regularization, or more training history, one change per comparison. | Training and validation curves and results by session. |
| “The image model seems to react to a chart label or layout.” | Test the same candle windows with controlled chart layouts or crops. | Predictions for paired images and the exact input used for each decision. |

For the first experiment, I’d keep a long-only, fixed-size, 30-minute strategy with the existing modeled costs. Agree on the label before training. The current Ridge label uses the next open to the second candle’s close. The strategy’s planned exit uses the following open. If the goal is to predict the move between simulated fills, define the label using the entry and planned exit opens and rerun Ridge on that label for a fair comparison. [Current training target](TRAINING_DATA_CONTRACT.md#merkmale-und-ziel) · [Ridge exit rule](BACKTEST_CONTRACT.md#eingefrorenes-ridge-modell)

Use chronological validation periods for tuning, fit preprocessing on training rows, and drop examples whose labels extend into the next split. The February forecast and trading periods already shown in reports are available for exploration. Reserve a fresh period for the final check. Compare forecast error against zero return and Ridge. Compare trading outcomes with the same size, entry/exit policy, and costs when testing models against each other. Record net P&L, drawdown, trade count, and results by session. [Existing split contract](TRAINING_DATA_CONTRACT.md#zeitliche-abschnitte)

## How we keep track

Before each run, we write down the hypothesis and what would count as an improvement. You choose the behavior to prioritize. I prepare the test cases, run the comparison, and bring back a report with examples we can inspect together. After review, we record whether to keep the change and what question comes next.

Use one short entry per experiment. The saved run should link the configuration, dataset and split dates, target and feature versions, model weights and preprocessing, timestamped predictions, backtest, and report. Save the code version and seed so we can reproduce the comparison.

```text
Experiment / parent run
Question and one proposed change
What stays fixed and what counts as improvement
Validation result, trade-offs, and examples reviewed
Decision and next question
```

**Next review.** Open the Ridge report and review one winning and one losing trade together. Decide whether the first neural net should predict the 30-minute return or the probability of a move large enough to cover costs. The starting recommendation is the 30-minute return with the target aligned to the planned fills. Write the target test cases before changing the training data. [Repository workflow](ROADMAP.md#1-ziel-und-verbindliche-reihenfolge)
