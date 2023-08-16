import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import statistics
import math

# Read data
file = open('News_Dataset.csv')
dataset = pd.read_csv('News_Dataset.csv')
stock = yf.Ticker("^GSPC")
dayInterval = 1 #days
#valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo


# Get historical market data
start_date = dataset.iloc[0][0]
end_date = dataset.iloc[dataset["Date"].size-1][0]
stockdata = stock.history(start= start_date, end= end_date, interval=(str(dayInterval)+'d'))

# Byte() to String(), remove b' and b" and replace /' with '
for row in range(dataset["Date"].size-1):
    for column in range(2,27):       
        try:
            #Cannot use .decode() becuase the pandas treat the item as a str instead of a byte
            dataset.iloc[row, column] = dataset.iloc[row,column].lstrip("b'")
            dataset.iloc[row, column] = dataset.iloc[row,column].lstrip('b"')
            dataset.iloc[row, column] = dataset.iloc[row,column].rstrip("'")
            dataset.iloc[row, column] = dataset.iloc[row,column].rstrip('"')
            dataset.iloc[row, column] = dataset.iloc[row,column].replace("\\'", "'")
        except:
            i = 1 #do nothing when the element is empty because some days do not contain 25 news

# Merge both datasets
stockdata_time = stockdata.index.strftime('%Y-%m-%d')
stockdata['Date'] = stockdata_time
stockdata.index = range(0, len(stockdata.index))
print(stockdata.info())

datajoinned = pd.merge(dataset, stockdata, how='inner', on='Date')

datajoinned = datajoinned.drop('Label', axis=1)
# Check changes Increasement or No change = 1, Decreasement = 0 and add to dataframe
label = []
openPrices = datajoinned["Open"]
closePrices = datajoinned["Close"]
for i in range(openPrices.size):
    if closePrices[i] >= openPrices[i]:
        label.append(1)
    else:
        label.append(0)

datajoinned["Label"] = label

# Check changes Increasement or No change = 1, Decreasement = 0 and add to dataframe
label, percentageChanges = [], []
openPrice = datajoinned["Open"]
closePrice = datajoinned["Close"]
# Calculate percentage changes
for i in range(openPrice.size): 
    priceChange = closePrice[i] - openPrice[i]
    percentageChange = priceChange / openPrice[i]
    percentageChanges.append((percentageChange*100))
# Find splitting points
acceptable_max = sorted(percentageChanges, reverse=True)[math.ceil(len(percentageChanges) * (1/3))] 
acceptable_min = sorted(percentageChanges, reverse=True)[math.ceil(len(percentageChanges) * (2/3))]
# Labeling 
for i in range(openPrice.size):
    if percentageChanges[i] > acceptable_max:
        label.append(1)
    elif percentageChanges[i] < acceptable_min:
        label.append(-1)
    else:
        label.append(0)
datajoinned["Label_I"] = label


from datetime import datetime, timedelta
change_in_vol_spread = 5
later_end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=14)).strftime('%Y-%m-%d') #14 days buffer
later_stockdata = stock.history(start= start_date, end= later_end_date, interval=(str(dayInterval)+'d'))
later_stockdata.to_csv("later_stock_data.csv")

#change in momentum for labeling
days_period = 5
label_new = []
for i in range(openPrice.size):
    current_chunk = later_stockdata.iloc[i:days_period+i]
    closing_prices = current_chunk["Close"]
    days_of_increase, days_of_decrease = 0, 0
    for j in range(1, closing_prices.size):
        if closing_prices[j] > closing_prices[j-1]:
            days_of_increase += 1
        elif closing_prices[j] < closing_prices[j-1]:
            days_of_decrease += 1
        else:
            days_of_increase += 1
            days_of_decrease += 1
    if days_of_increase > days_of_decrease:
        label_new.append(1)
    elif days_of_increase < days_of_decrease:
        label_new.append(-1)
    else:
        label_new.append(0)
datajoinned["Label_II"] = label_new

#change in volume
change_in_vol = []
for i in range(openPrice.size):
    current_chunk = later_stockdata.iloc[i:change_in_vol_spread+i]
    vol = current_chunk["Volume"]
    change = 0
    for j in range(1, vol.size):
        change += vol[j] - vol[j-1]
    change_in_vol.append(change/vol.size)
datajoinned["Vol_change"] = change_in_vol





# Final
print(datajoinned.info())
datajoinned.to_csv("Processed_Dataset.csv")