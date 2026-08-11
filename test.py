import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

win_width, win_height = 8,5

plt.figure(figsize=(win_width,win_height))

data = yf.Ticker("AAPL").history(start="2023-01-01", end="2024-01-01")
close = data["Close"]#default aleady adjusts for stock splits and inflations etc

close.plot(title="AAPL close")
plt.show(block=False)


plt.figure(figsize=(win_width,win_height))

R = close.diff().var() # this kinda makes this whole thing reduandant for now since it forces a lookahead bias
Q = 0.1 * R

x_est, P = close.iloc[0], R #iloc is  interger location, so we just take first data pint as x_est

#p is prediction uncertaininty
#k is gain, how muhc weight to the difference we move by
#q is uncertainity in estimate/process variances
#r is uncertainty in the new data points

filtered = [x_est]
for price in close.iloc[1:]:#for each data point in the ticker list after the first

    P += Q
    K = P / (P + R)#gain is how much we weight the new price and the remaining is how much we weight the oldprice estimate
    x_est += K * (price - x_est)#using gain, how much do we trust the difference to effect our estimate
    P *= (1 - K)#this keeps p from growing forever as k is always less than 1 so multiplying by a fraction to keep p low


    filtered.append(x_est)

filtered = pd.Series(filtered, index=close.index)


close.plot(label="raw")
filtered.plot(label="filtered")
plt.legend()
plt.show()



signal = np.where(filtered > close, 1, -1) # return 1 if true or -1 if false to each data point in series
signal = pd.Series(signal, index=close.index)
position = signal.shift(1)#so we skip day 1
daily_returns = close.pct_change()#finds change from previous data point to current

trade_cost = 0.0005  #  0.05% per trade, just some arbitrary number i chose but should refine
traded = position.diff().fillna(0) != 0 #fills with 0 and returns true or false if traded since there would always be a diff due to trade cost
strategy_returns = position * daily_returns
strategy_returns[traded] -= trade_cost

plt.figure(figsize=(win_width, win_height))
cumulative = (1 + strategy_returns.fillna(0)).cumprod() # remember this is all vectorised and so is cumulative
cumulative.plot(title="Strategy cumulative returns")
plt.show()

sharpe = strategy_returns.mean() / strategy_returns.std() * (252 ** 0.5) #assumes a 0 risk free rate which is goofy but fine for now
running_max = cumulative.cummax()#just finds the all time peak
drawdown = cumulative / running_max - 1
max_dd = drawdown.min() # to find the maximum risk "pain" from all time peak and trough

print("Sharpe", sharpe)
print("Max drawdown", max_dd)