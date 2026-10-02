"""
Real-world Data Project (Finance): Stock Price Analysis & Next-Day Direction Prediction
-----------------------------------------------------------------------------------------
Steps: 1) Load data  2) Clean  3) EDA + charts  4) Feature engineering
       5) Model (Random Forest, time-based split)  6) Evaluate  7) Conclusions

Install:  pip install yfinance pandas numpy matplotlib seaborn scikit-learn
Run:      python stock_analysis_project.py
Outputs:  charts saved in ./charts/ and a summary printed in the terminal
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TICKER = "RELIANCE.NS"      # change to e.g. "TCS.NS", "INFY.NS", "AAPL"
START, END = "2019-01-01", "2025-12-31"
os.makedirs("charts", exist_ok=True)
sns.set_style("whitegrid")

# ---------------------------------------------------------------- 1. LOAD
df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):          # newer yfinance versions
    df.columns = df.columns.get_level_values(0)
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------- 2. CLEAN
print("\nMissing values:\n", df.isna().sum())
df = df.dropna().sort_index()
df = df[~df.index.duplicated()]

# ---------------------------------------------------------------- 3. EDA
df["Return"] = df["Close"].pct_change()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["Close"], label="Close", lw=1)
plt.plot(df["MA50"], label="50-day MA")
plt.plot(df["MA200"], label="200-day MA")
plt.title(f"{TICKER} Price with Moving Averages"); plt.legend()
plt.savefig("charts/1_price_ma.png", dpi=150, bbox_inches="tight"); plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(df["Return"].dropna(), bins=60, kde=True)
plt.title("Distribution of Daily Returns")
plt.savefig("charts/2_returns_hist.png", dpi=150, bbox_inches="tight"); plt.close()

plt.figure(figsize=(12, 3.5))
plt.bar(df.index, df["Volume"], width=1)
plt.title("Trading Volume")
plt.savefig("charts/3_volume.png", dpi=150, bbox_inches="tight"); plt.close()

ann_return = df["Return"].mean() * 252
ann_vol = df["Return"].std() * np.sqrt(252)
sharpe = ann_return / ann_vol
cum = (1 + df["Return"].fillna(0)).cumprod()
max_dd = (cum / cum.cummax() - 1).min()
print(f"\nAnnual return: {ann_return:.2%} | Volatility: {ann_vol:.2%} | "
      f"Sharpe (rf=0): {sharpe:.2f} | Max drawdown: {max_dd:.2%}")

# ---------------------------------------------------------------- 4. FEATURES
def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)

feat = pd.DataFrame(index=df.index)
feat["ret_1"] = df["Return"]
feat["ret_5"] = df["Close"].pct_change(5)
feat["ret_10"] = df["Close"].pct_change(10)
feat["vol_10"] = df["Return"].rolling(10).std()
feat["ma_ratio"] = df["Close"] / df["MA50"]
feat["rsi"] = rsi(df["Close"])
feat["vol_chg"] = df["Volume"].pct_change()
# Target: does tomorrow's close go UP? (shift(-1) avoids look-ahead leakage in features)
feat["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

plt.figure(figsize=(7, 5))
sns.heatmap(feat.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("charts/4_corr.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 5. MODEL
X, y = feat.drop(columns="target"), feat["target"]
split = int(len(feat) * 0.8)                       # time-based split, NOT random
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestClassifier(n_estimators=300, max_depth=5,
                               min_samples_leaf=20, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ---------------------------------------------------------------- 6. EVALUATE
acc = accuracy_score(y_test, pred)
baseline = max(y_test.mean(), 1 - y_test.mean())   # always guess majority class
print(f"\nModel accuracy: {acc:.3f} | Majority-class baseline: {baseline:.3f}")
print(classification_report(y_test, pred))

plt.figure(figsize=(4.5, 4))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.savefig("charts/5_confusion.png", dpi=150, bbox_inches="tight"); plt.close()

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(6, 4)); imp.plot.barh(); plt.title("Feature Importance")
plt.savefig("charts/6_importance.png", dpi=150, bbox_inches="tight"); plt.close()

# Simple backtest: hold the stock only on days the model predicts UP
test_ret = df["Return"].shift(-1).reindex(X_test.index)
strategy = (pd.Series(pred, index=X_test.index) * test_ret).fillna(0)
plt.figure(figsize=(10, 4))
plt.plot((1 + test_ret.fillna(0)).cumprod(), label="Buy & Hold")
plt.plot((1 + strategy).cumprod(), label="Model Strategy")
plt.title("Backtest on Test Period (no transaction costs)"); plt.legend()
plt.savefig("charts/7_backtest.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 7. CONCLUSIONS
print("""
CONCLUSIONS (edit with your own numbers)
1. Price shows long-term trend; 50/200-day MAs help visualise momentum regimes.
2. Daily returns are roughly symmetric with fat tails -> large moves are more
   common than a normal distribution suggests.
3. The model's accuracy should be compared with the majority-class baseline;
   if it is only marginally better, next-day direction is close to a random
   walk, which is consistent with market-efficiency theory.
4. Limitations: one stock, price-only features, no transaction costs, no
   news/fundamentals. Next steps: add macro/sentiment data, try walk-forward
   validation, test several tickers.
""")
