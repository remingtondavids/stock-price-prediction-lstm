""" Use long short-term memory (LSTM) neural networks to predict stock prices """
from xml.etree.ElementInclude import include


from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=1000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('TSLA',
                   start=start_date,
                   end=end_date,
                   progress=False)
data["Date"] = data.index

data = data[["Date", "Open", "High", "Low",
             "Close", "Adj Close", "Volume"]]

data.reset_index(drop=True, inplace=True)
data.tail()

figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"],
                                        high=data["High"],
                                        low=data["Low"],
                                        close=data["Close"])])

figure.update_layout(title="GM Stock Price Analysis",
                     xaxis_rangeslider_visible=False)

figure.show()

correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))

x = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)


#features = [Open, High, Low, Adj Close, Volume]
features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
print(model.predict(features))
