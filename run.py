import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from data_loader import load_data
from normalize import normalize
from model import train_random_forest

# Load and normalize
d = load_data()
d, scaler = normalize(d)

# Forecast future
pred_days = 30
pred = train_random_forest(d['norm'], pred_days)
pred = scaler.inverse_transform(pred.reshape(-1,1)).flatten()

last_date = d.index[-1]
future_dates = pd.date_range(last_date, periods=pred_days+1, closed='right')

# Stats / Filters
def filter_range(start, end):
    return d.loc[str(start):str(end)]

f2020_2025 = filter_range("2020", "2025")
stats = {
    "period": "2020-2025",
    "min_price": float(f2020_2025['price'].min()),
    "max_price": float(f2020_2025['price'].max()),
    "avg_price": float(f2020_2025['price'].mean()),
    "volatility": float(f2020_2025['price'].std())
}

print("\n=== STATS (2020-2025) ===")
for k,v in stats.items():
    print(k, ":", v)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=d.index, y=d['price'], name="Historical"))
fig.add_trace(go.Scatter(x=future_dates, y=pred, mode='lines+markers', name="Forecast 30d"))
fig.update_layout(title="NVDA Stock Trend + Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
fig.show()
