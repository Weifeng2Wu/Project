from Keys import ameritrade
import requests
from statistics import mean,stdev
import matplotlib.pyplot as plt

import time

Stock = "SPY"
Time_period = 24
Threshold = .5
Protion = .2 # 其他条件不变下，Protion越高，earning越大
endDate = "1654308000000" #US/Eastern Time: 2022-06-03 22:00:00

position = {
    'amount': 100,
    'average_cost': 400,
    'earning': 0
}

payload = {
    'apikey': ameritrade,
    'periodType': 'day',
    'period': "5",
    'frequencyType': "minute",
    'frequency': "1",
    'endDate': endDate
}

url = 'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(Stock)

result = requests.get(url, params=payload)
price, bup_price, bdown_price = [], [], []
position_amount = []

#calculate bollinger bands
def get_bbands(Price,Time_period):
    price = Price[-Time_period:]
    ma = mean(price)
    std = stdev(price)
    bup = ma + 2*std
    bdown = ma - 2*std
    return bup, bdown

def place_order(bup,bdown,cur):
    if cur+Threshold > bup:
        #sell
        sell_amount = position['amount']*Protion
        if sell_amount < 0.1:
            print('cannot sell the stock')
            return
        position['earning'] += sell_amount*(cur-position['average_cost'])
        position['amount'] -= sell_amount
        #position['amount'] = position['amount']*(1-Protion)
        print('Sell {} amount of {} at ${}'.format(sell_amount,Stock,cur))
        print(position['earning'])
    if cur-Threshold < bdown:
        #buy
        buy_amount = position['amount'] * Protion
        total_value = buy_amount * cur + position["amount"] * position["average_cost"]
        position['amount'] += buy_amount
        position['average_cost'] = total_value/position['amount']
        print('Buy {} amount of {} at ${}'.format(buy_amount,Stock,cur))


def main():
    for candles in result.json()['candles']:
        cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(candles["datetime"])/1000.0))
        cur_price = (candles['low']+candles['high'])/2
        price.append(cur_price)

        if len(price) >= Time_period:
            period_price = price[-Time_period:]
            bup, bdown = get_bbands(period_price,Time_period)
            bup_price.append(bup)
            bdown_price.append(bdown)
            place_order(bup,bdown,cur_price)

        else:
            bup_price.append(cur_price)
            bdown_price.append(cur_price)
        position_amount.append(position['amount'])
    print(position)

    plt.plot(price, '-.')
    plt.plot(bup_price, '-r.')
    plt.plot(bdown_price, '-g.')
    plt.show()

import sys
old_stdout = sys.stdout
log_file = open("message.log","w")
sys.stdout = log_file

main()

sys.stdout = old_stdout
log_file.close()



