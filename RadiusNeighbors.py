import numpy as np
import pandas as pd
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# read csv
data = pd.read_csv("radius_wtx.csv", encoding='latin1')
data['time'] = pd.to_datetime(data['time'])

# corelation
corr = data.corr(method='spearman')
print(corr['tomorrow'])

# 使用 close_price 列作為特徵
#close_prices = data.drop(columns=['K(9,3)', 'D(9,3)', 'J', 'DIF12-26', 'MACD9', 'OSC','financing','financing_difference','tomorrow']).values
#features=data.drop(columns=['K(9,3)', 'D(9,3)', 'J', 'DIF12-26', 'MACD9', 'OSC','financing','financing_difference','tomorrow'])
#features=['close','EMA5','EMA10','EMA60','EMA120','UB2.00','trust','dealer','volume','MA5','MA10']
features=['close','trust']
#print(features)
# 獲取日期

#print("dates : ",dates)
# 使用前20天的數據來預測第21天的數據
x = []
y = []
x_dates=[]
y_dates = []



for i in range(0,len(data) - 80,20):
    x.append(data[features].iloc[i:i+60].values.flatten())  # 將20天的資料平坦化為一個矩陣
    y.append(data['close'].iloc[i+60:i+80])
    x_dates.append(data['time'].iloc[i:i+60])
    y_dates.append(data['time'].iloc[i+60:i+80])


X = np.array(x)
y = np.array(y)
#print(y)
#標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled.shape)
split_index = int(0.8 * len(X_scaled))  # 80% for training
print("split_index : ",split_index)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
dates_train, dates_test = y_dates[:split_index], y_dates[split_index:]
x_dtr,x_dts  = x_dates[:split_index], x_dates[split_index:]

#model
radius = 20
model = RadiusNeighborsRegressor(radius=radius, weights='distance')#weight:uniform=all points equally
model.fit(X_train, y_train)

#預測
y_pred = model.predict(X_test)

# 獲取鄰居數量
neighbors_count = [len(neighbors) for neighbors in model.radius_neighbors(X_test, return_distance=False)]
neighbors_indices = model.radius_neighbors(X_test, return_distance=False)#輸出會是索引值
print(neighbors_count)
#print("NB : ",neighbors_indices)
#計算強弱指標
strength_indicator = y_pred - y_test
# 計算預測值的日差值
#diffs = np.diff(y_pred, axis=1)
# 在日差值矩陣前加一列0
#strength_indicator = np.hstack((np.zeros((diffs.shape[0], 1)), diffs))


# 輸出預測的強弱指標和對應的日期
for idx, date_block in enumerate(dates_test):
    print(idx)
    
    # 輸出該日期區間的每一天的預測date, 
    for date,strength in zip(date_block, strength_indicator[idx]): #strength_indicator[idx]代表是預測完扣掉原本該有的 他的[]中的第idx組一組有n個預測結果

        print(f"Date: {date}, Strength Indicator: {strength}, Neighbors Count: {neighbors_count[idx]}")
    #test
    #print("Neighbors' dates:")
    #neighbor_dates = [x_dtr[n] for n in neighbors_indices[idx]]
    #print(neighbor_dates[1]) #還在看到底鄰居那幾個到底符合不符合
    #for date in neighbor_dates:
        #print(date)
    print("Neighbors' statistics:")
    neighbor_indices = neighbors_indices[idx]
    #print(neighbor_indices)
    
    if neighbor_indices.size == 0:
        print("No neighbors found for this test instance.")
        print("------")
        input("Press Enter to continue...")  # pause execution
        continue

    # 获取这些neighbors的20天后的真实价格
    #neighbors_future_prices = [y_train[n][-1] for n in neighbor_indices]
    
    # 计算涨跌的点数
    #price_changes = [y_train[n][-1] - X_train[n][-len(features):][features.index('close')] for n in neighbor_indices]
    #print("price_changes : ",price_changes)
    price_changes = []
    for n in neighbor_indices:
        last_day_close = X[n][-2]  # 從 -2 處取得 'close' 的值，因為您有2個特徵，'close' 和 'DIF12-26'#raw為特別保留
        future_price = y_train[n][-1]
    
        print("Last Day Close:", last_day_close)
        print("Future Price (20 days later):", future_price)
    
        price_change = future_price - last_day_close
        price_changes.append(price_change)

    print(price_changes)
    # 计算上涨的概率
    rise_probability = sum(1 for change in price_changes if change > 0) / len(price_changes)
    print(f"Rise Probability: {rise_probability:.2f}")
    
    # 计算涨跌的最大点数
    print(f"Maximum Rise Points: {max(price_changes):.2f}")
    print(f"Maximum Drop Points: {min(price_changes):.2f}")

    # 獲得 X_train 中的 'close' 值，也就是某一天的價格
    close_values = [X[n][-2] for n in range(len(X_train))]

    # 使用 y_train 來計算 20 天後的價格變動
    price_changes = [y[-1] - close for y, close in zip(y_train, close_values)]

    # 根據價格變動決定顏色
    colors = ['red' if change > 0 else 'blue' for change in price_changes]


    #每次預測的neibor點陣圖
    '''
    plt.scatter(close_values, price_changes, c=colors)
    plt.xlabel('Price at a Specific Day')
    plt.ylabel('Price Change after 20 Days')
    plt.title('Scatter Plot of Price Changes')
    plt.show()

    print("------")  # 分隔符號
    # 每顯示一組測試數據的預測結果後暫停
    input("Press Enter to continue...")  # pause execution
    '''

print(len(dates_test))
# 繪製 y_test 和 y_pred
plt.figure(figsize=(14, 6))

# 因為 y_test 和 y_pred 是多維數組, 我們需要將它們平坦化以進行繪圖
#plt.plot(np.array(dates_test[:13]).flatten(), np.array(y_test[:13]).flatten(), label='Actual', color='blue')
plt.plot(np.array(dates_test).flatten(), np.array(y_pred).flatten(), label='Predicted', color='red', linestyle='--')

plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()