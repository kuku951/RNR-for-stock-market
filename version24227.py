import numpy as np
import pandas as pd
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.dates as mdates

# read csv
data = pd.read_csv("ra_new2013_2023_copy2.csv", encoding='latin1')
data['time'] = pd.to_datetime(data['time'])
# corelation
corr = data.corr(method='spearman')
print(corr['percentage'])

# 使用 close_price 列作為特徵
#close_prices = data.drop(columns=['K(9,3)', 'D(9,3)', 'J', 'DIF12-26', 'MACD9', 'OSC','financing','financing_difference','tomorrow']).values
#features=data.drop(columns=['K(9,3)', 'D(9,3)', 'J', 'DIF12-26', 'MACD9', 'OSC','financing','financing_difference','tomorrow'])
#features=['close','EMA5','EMA10','EMA60','EMA120','UB2.00','trust','dealer','volume','MA5','MA10']
features=['trust_buy','close','financing_error']
#print(features)
# 獲取日期

#print("dates : ",dates)
# 使用前20天的數據來預測第21天的數據
x = []
y = []
x_dates=[]
y_dates = []

#print("len",len(data))

for i in range(0,len(data) - 80,20):
    x.append(data[features].iloc[i:i+60].values.flatten())  # 將20天的資料平坦化為一個矩陣
    y.append(data['percentage'].iloc[i+60:i+80])
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
neighbors_indices = model.radius_neighbors(X_test, return_distance=False)#輸出會是索引值(training data的)

#print("NB : ",neighbors_indices)
#計算強弱指標
strength_indicator = y_pred - y_test
# 計算預測值的日差值
#diffs = np.diff(y_pred, axis=1)
# 在日差值矩陣前加一列0
#strength_indicator = np.hstack((np.zeros((diffs.shape[0], 1)), diffs))


# 輸出預測的強弱指標和對應的日期 idx代表第幾組預測
for idx, date_block in enumerate(dates_test):
    limit = 0
    print("enu : ",enumerate(dates_test))
    # 輸出該日期區間的每一天的預測date, 
    for date,strength in zip(date_block, strength_indicator[idx]): #strength_indicator[idx]代表是預測完扣掉原本該有的 他的[]中的第idx組一組有n個預測結果

        print(f"Date: {date}, Strength Indicator: {strength}, Neighbors Count: {neighbors_count[idx]}")

    #輸出相似的日期
    neighbor_indices = neighbors_indices[idx]
    
    #輸出數據點相似的日期
    for idx, date_block in enumerate(dates_test):
        # 獲取當前測試點的鄰居索引
        neighbor_indices = neighbors_indices[idx]
        print(f"Test Point {neighbor_indices} Neighbors' Dates:")

        if len(neighbor_indices) == 0:
            print("No neighbors found for this test instance.")
        else:
            # 輸出每個鄰居的日期
            for n_idx in neighbor_indices:
                neighbor_dates = x_dtr[n_idx]  # 獲取鄰居的日期
                first_date = neighbor_dates.iloc[0]  # 第一天
                last_date = neighbor_dates.iloc[-1]  # 最後一天
                print(f"Neighbor {n_idx}: {first_date} to {last_date}")

        # 輸出日期區間中每天的預測
        for date, strength in zip(date_block, strength_indicator[idx]):
            print(f"Date: {date}, Strength Indicator: {strength}, Neighbors Count: {neighbors_count[idx]}")
        

        # 確保 limit 不超過 x_dts 的索引範圍
        if limit < len(x_dts):
            group_dates = x_dts[limit]  # 取出當前的日期組
            mask = data['time'].isin(group_dates)
            selected_data = data[mask]

            # 創建一個 figure 和兩個 subplots: (2 rows, 1 column)
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # 調整 figsize 以適應垂直排列的兩個圖表

            # 第一個 subplot: 繪製收盤價
            ax[0].plot(selected_data['time'], selected_data['close'], marker='o', linestyle='-', color='red')
            # 設定日期格式
            locator = mdates.AutoDateLocator()
            formatter = mdates.DateFormatter('%Y-%m-%d')
            ax[0].xaxis.set_major_locator(locator)
            ax[0].xaxis.set_major_formatter(formatter)
            ax[0].set_title(f'Group {limit} Close Prices Over Time')
            ax[0].set_xlabel('Date')
            ax[0].set_ylabel('Close Price')
            ax[0].tick_params(axis='x', rotation=45)  # 旋轉日期標籤

            # 第二個 subplot: 繪製 trust_buy 的長條圖
            ax[1].bar(selected_data['time'], selected_data['trust_buy'], color='blue')
            # 設定日期格式
            ax[1].xaxis.set_major_locator(locator)
            ax[1].xaxis.set_major_formatter(formatter)
            ax[1].set_title(f'Group {limit} Trust Buy Over Time')
            ax[1].set_xlabel('Date')
            ax[1].set_ylabel('Trust Buy')
            ax[1].tick_params(axis='x', rotation=45)  # 旋轉日期標籤

            plt.tight_layout()  # 自動調整子圖參數，以給定的填充參數
            plt.show()

            limit += 1  # 準備處理下一組日期
        else:
            print("No more groups to display.")
        
        for n_idx in neighbor_indices:
             # 獲取鄰居的日期
            neighbor_dates = x_dtr[n_idx]
            
            # 獲取對應的收盤價和 trust_buy
            neighbor_data = data.loc[data['time'].isin(neighbor_dates)]
            
            # 創建一個 figure 和兩個 subplots: (2 rows, 1 column)
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # 調整 figsize 以適應垂直排列的兩個圖表

            # 第一個 subplot: 繪製收盤價線圖
            ax[0].plot(neighbor_data['time'], neighbor_data['close'], marker='o', linestyle='-', color='blue')
            # 設定日期格式
            locator = mdates.AutoDateLocator()
            formatter = mdates.AutoDateFormatter(locator)
            ax[0].xaxis.set_major_locator(locator)
            ax[0].xaxis.set_major_formatter(formatter)
            ax[0].set_title(f'Neighbor {n_idx} Close Prices Over Time')
            ax[0].set_xlabel('Date')
            ax[0].set_ylabel('Close Price')
            ax[0].tick_params(axis='x', rotation=45)

            # 第二個 subplot: 繪製 trust_buy 長條圖
            ax[1].bar(neighbor_data['time'], neighbor_data['trust_buy'], color='green')
            # 設定日期格式
            ax[1].xaxis.set_major_locator(locator)
            ax[1].xaxis.set_major_formatter(formatter)
            ax[1].set_title(f'Neighbor {n_idx} Trust Buy Over Time')
            ax[1].set_xlabel('Date')
            ax[1].set_ylabel('Trust Buy')
            ax[1].tick_params(axis='x', rotation=45)

            plt.tight_layout()  # 自動調整子圖參數，以給定的填充參數
            plt.show()

        print("------")  # 分隔符
        input("Press Enter to continue...")  # 暫停執行
    # 獲得 X_train 中的 'close' 值，也就是某一天的價格
    close_values = [X[n][-2] for n in range(len(X_train))]

    # 使用 y_train 來計算 20 天後的價格變動
    price_changes = [y[-1] - close for y, close in zip(y_train, close_values)]

    # 根據價格變動決定顏色
    colors = ['red' if change > 0 else 'blue' for change in price_changes]
    break #直接結束不然會重複組數
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