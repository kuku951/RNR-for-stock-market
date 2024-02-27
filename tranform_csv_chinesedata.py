import pandas as pd

def read_csv_with_encoding(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='utf-16')
# 讀取CSV文件
file_path = 'ra_new2013_2023_copy.csv'  # 你的CSV文件路徑
df = read_csv_with_encoding(file_path)

# 定義轉換函數
def convert_chinese_numerals_to_numbers(chinese_numeral):
    conversion_factors = {'窾': 10**4, '货': 10**8}
    for chinese_char, factor in conversion_factors.items():
        if chinese_char in chinese_numeral:
            number_part, _ = chinese_numeral.split(chinese_char)
            return float(number_part) * factor
    return float(chinese_numeral)

# 確保 'a' 欄位是字符串類型
df['trust_buy'] = df['trust_buy'].astype(str)

# 應用轉換函數
df['trust_buy'] = df['trust_buy'].apply(convert_chinese_numerals_to_numbers)

# 將轉換後的數據寫回CSV文件
df.to_csv(file_path, index=False)
