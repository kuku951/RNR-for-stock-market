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
dates = data['time'].values  # 從第21天開始前20天的數據來預測
print(dates.shape)

# 使用前20天的數據來預測第21天的數據
x = []
y = []
y_dates = []  # 用來保存每組數據的開始日期
'''
for i in range(20, len(close_prices)):
    X.append(close_prices[i-20:i])
    y.append(close_prices[i])
'''
#print(data[features].iloc[0:20])
#print(data['close'].iloc[20])
#len(len(data))

for i in range(len(data) - 80):
    x.append(data[features].iloc[i:i+60].values.flatten())  # 將20天的資料平坦化為一個矩陣
    y.append(data['close'].iloc[i+60:i+80])
    y_dates.append(data['time'].iloc[i+60:i+80])

#print(x)
X = np.array(x)
y = np.array(y)
#print(y)
#標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #len(x)=1890 1970組數據扣掉80組所以會有1890天 因為是每隔一日做一次

#print(len(X))
split_index = int(0.8 * len(X_scaled))  # 80% for training
#print("split_index : ",split_index)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
dates_train, dates_test = y_dates[:split_index], y_dates[split_index:]

#StopAsyncIteration
#model
radius = 8
model = RadiusNeighborsRegressor(radius=radius)
model.fit(X_train, y_train)

#預測
y_pred = model.predict(X_test)

# 獲取鄰居數量
neighbors_count = [len(neighbors) for neighbors in model.radius_neighbors(X_test, return_distance=False)]

#計算強弱指標
strength_indicator = y_pred - y_test


'''
# sort
sorted_indices = np.argsort(dates_test)
sorted_dates = dates_test[sorted_indices]
sorted_strength_indicator = strength_indicator[sorted_indices]
sorted_neighbors_count = np.array(neighbors_count)[sorted_indices]

# 輸出預測的強弱指標和對應的日期，每20天一個分隔
count = 0
for date, indicator, neighbor in zip(sorted_dates, sorted_strength_indicator, sorted_neighbors_count):
    if count != 0 and count % 20 == 0:
        print("------")  # 分隔符號
    print(f"Date: {date}, Strength Indicator: {indicator}, Neighbors Count: {neighbor}")
    count += 1
'''
#strength_indicator neighbors_count dates_test
# sort
'''
sorted_indices = np.argsort(dates_test)
sorted_dates = dates_test[sorted_indices]
sorted_strength_indicator = strength_indicator[sorted_indices]
sorted_neighbors_count = np.array(neighbors_count)[sorted_indices]
'''
'''
batch_size = 20  # 每一批顯示的預測天數

all_dates = data['time'].values  # 從原始數據中獲取所有日期

# 輸出預測的強弱指標和對應的日期
for idx, date in enumerate(dates_test):
    
    # 找到當前date在all_dates中的索引
    current_index = np.where(all_dates == date)[0][0]

    # 輸出該日期的每一天的預測
    for i in range(20):
        if current_index + i < len(all_dates):  # 確保索引不超過all_dates的長度
            future_date = all_dates[current_index + i]
            print(f"Date: {future_date}, Strength Indicator: {sorted_strength_indicator[idx][i]}, Neighbors Count: {sorted_neighbors_count[idx]}")
    print("------")  # 分隔符號
    # 每顯示20天的預測結果後暫停
    if (idx + 1) % batch_size == 0:
        input("Press Enter to continue...")  # pause execution
'''

# 輸出預測的強弱指標和對應的日期
for idx, date_block in enumerate(dates_test):
    print(idx)
    
    # 輸出該日期區間的每一天的預測date, 
    for date,strength in zip(date_block, strength_indicator[idx]): #strength_indicator[idx]代表是預測完扣掉原本該有的 他的[]中的第idx組一組有n個預測結果
       # print(date_block)
       # print(strength_indicator[idx])
        print(f"Date: {date}, Strength Indicator: {strength}, Neighbors Count: {neighbors_count[idx]}")

    
    print("------")  # 分隔符號
    # 每顯示一組測試數據的預測結果後暫停
    input("Press Enter to continue...")  # pause execution

'''
# 为了简化, 我们只使用前20个预测值进行绘图（您可以修改这个数字）
n_points = 50
dates_to_plot = sorted_dates[:n_points]
strengths_to_plot = [si[0] for si in sorted_strength_indicator[:n_points]]
neighbors_to_plot = sorted_neighbors_count[:n_points]

plt.figure(figsize=(12, 6))

# 绘制强弱指标
plt.subplot(2, 1, 1)
plt.scatter(dates_to_plot, strengths_to_plot, color='b', label="Strength Indicator")
plt.title('Strength Indicator over Time')
plt.xlabel('Date')
plt.ylabel('Strength Indicator')
plt.legend()
plt.grid(True)

# 绘制邻居数量
plt.subplot(2, 1, 2)
plt.scatter(dates_to_plot, neighbors_to_plot, color='r', label="Neighbors Count")
plt.title('Neighbors Count over Time')
plt.xlabel('Date')
plt.ylabel('Neighbors Count')
plt.legend()
plt.grid(True)

# 调整布局并显示图
plt.tight_layout()
plt.show()
'''
