import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_random_forest(series, days=30, lags=[1,2,3,5,10]):
    df = pd.DataFrame({'y': series})
    for l in lags:
        df[f'lag_{l}'] = df['y'].shift(l)
    df = df.dropna()

    X = df[[f'lag_{l}' for l in lags]].values
    y = df['y'].values

    m = RandomForestRegressor(n_estimators=300, random_state=42)
    m.fit(X, y)

    forecast = []
    hist = series.copy().tolist()

    for _ in range(days):
        feats = [hist[-l] for l in lags]
        p = m.predict([feats])[0]
        forecast.append(p)
        hist.append(p)

    return np.array(forecast)


def iterative_train_random_forest(series, days=30, lags=[1,2,3,5,10], iterations=5):
    all_forecasts = []

    for _ in range(iterations):
        forecast = train_random_forest(series, days, lags)
        all_forecasts.append(forecast)

    return np.mean(all_forecasts, axis=0)


if __name__ == "__main__":
    # Simple test
    data = pd.DataFrame({'norm': np.linspace(0, 1, 100)})
    
    pred = iterative_train_random_forest(data['norm'], days=10, iterations=3)
    # here we are predicting 10 days ahead with 3 iterations to average out to see what the stock might be like in 10 days
    print("Forecasted values:", pred)
    # Semi volatile and a dump at the end to simulate stock prices -> volatility
    sample_stock_numbers_for_a_year = np.array([100, 200, 100, 150, 400, 239, 543, 113, 321, 1000, 543, 10, 3]) # this gives numbers averageing around 200+ showing it doesnt handle volatility well which is why we need to account for market trends too AI can just dip one day and skyrocket the next

    pred = iterative_train_random_forest(pd.Series(sample_stock_numbers_for_a_year), days=13, iterations=7)
    # Given the volatility of these stock prices we want to see what the model predicts in 13 days with 7 iterations to average out the randomness
    print("Forecasted values:", pred)
    # Fairly realistic stock data to show consistency
    # here we have a more consistent set of stock prices so we want to see how it peforms in 5 days with 10 iterations
    sample_stock_numbers_for_a_year = np.array([100, 102, 101, 105, 110, 108, 107, 111, 115, 117, 
                                              120, 125, 130]) # This was consistent
    
    pred = iterative_train_random_forest(pd.Series(sample_stock_numbers_for_a_year), days=5, iterations=10)
    print("Forecasted values:", pred)