from sklearn.preprocessing import MinMaxScaler
import pandas as pd


"""
We Want to Normalize the 'price' column in our dataframe to a range between 0 and 1.

This is important because we want to ensure out model can grasp patterns in the data without being skewed by large price values.

To do this, we will use MinMaxScaler from sklearn which scales the data to a specified range.

Through using the data we grab from data_loader.py and yahoo finance, we create a new column for normalized prices called 'norm'. This value will be between 0 and 1. For example
with out example we wrote in the if __name__ == "_main_: block, 

We see that a price of a stock starts at 100, increase 100 incremently each month up to 500. The normalized values will start at 0.0 for 100 and go up to 1.0 for 500, with values in between representing their relative position in that range.

In month 0 the price is 100 -> norm = 0.0
In month 1 the price is 200 -> norm = 0.25 (This is because 200 is 25% of the way from 100 to 500)
.... So on

The Highest price for NVDA in our dataset will be scaled to 1.0 and the lowest to 0.0 with all other prices scaled proportionally in between.

We want to showcase the regular data alongside trhe normilized data to see how the scaling affects the values.
"""
def normalize(df):
    s = MinMaxScaler()
    df['norm'] = s.fit_transform(df[['price']])
    return df, s

if __name__ == "__main__":
    # Simple test
    data = pd.DataFrame({'price': [100, 200, 300, 400, 500]})
    norm_data, scaler = normalize(data)
    print(norm_data)
    print("Inverse transform of 0.5:", scaler.inverse_transform([[0.5]]))


