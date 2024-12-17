# parsing_keystats.py
import pandas as pd

def fetch_financial_ratios_from_file(file_path='keystats.csv'):
    # Đọc dữ liệu từ file Excel đã xuất
    df = pd.read_excel(file_path, engine='openpyxl', skiprows=7)
    return df

def parse_financial_ratios(df):
    # Đặt cột 'year' làm chỉ mục (nếu có)
    if 'year' in df.columns:
        df.set_index('year', inplace=True)
    return df
