利用Radius Neighbors Regressor模型預測台灣股市加權指數並賦予強弱指標
本專案使用機器學習方法來預測股市的漲跌趨勢且賦予強弱指標。利用 Python 的資料處理與機器學習庫，例如 Pandas、NumPy 和 scikit-learn，結合財經數據建立預測模型。

項目依賴
NumPy
Pandas
scikit-learn
Matplotlib
Plotly
程式碼說明
數據讀取與前處理：

使用 Pandas 載入 ra_new2013_2023_copy2.csv 文件，並進行數據預處裡。
計算數據的相關係數，並選取重要的特徵。
特徵選擇與標準化：

根據相關性分析選擇特徵。
利用 StandardScaler 進行數據標準化。
建立模型：

使用 RadiusNeighborsRegressor 建立回歸模型。
訓練模型並進行預測。
結果評估：

計算鄰居的數量並分析預測結果。
使用 Matplotlib 和 Plotly 繪製實際與預測數據的比較圖。
漲跌幅度計算：

根據鄰居數據計算漲跌幅度。
輸出每個測試實例的漲跌幅度與相對應的鄰居日期。
如何運行
確保所有依賴已正確安裝。
將 CSV 數據文件放置於專案根目錄下。
執行 version24227.py 檔案。
預測結果
此模型提供了股市未來價格變化的預測，可視化結果顯示預測與實際情況的比較，幫助投資者做出更精確的投資決策。
