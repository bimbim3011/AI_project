from vnstock3 import Vnstock, Trading
import pandas as pd
import logging

def fetch_stock_data(ticker):
    try:
        stock = Trading()
        # Giả sử `price_board` lấy dữ liệu mới nhất mà không cần lọc ngày
        data = stock.price_board(symbols_list=[ticker])
        df = pd.DataFrame(data)
        df['symbol'] = ticker 
        logging.info(f"Fetched data for {ticker}: {data[:5]}")  # Ghi lại 5 mục đầu tiên để kiểm tra
        logging.info(f"Tên các cột trong DataFrame trả về: {df.columns}")
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def select_columns(df, ticker):
    try:
        # Làm phẳng các cột chỉ số đa cấp
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
        logging.info(f"Các cột sau khi làm phẳng: {df.columns}")

        # Thêm cột symbol
        df['symbol'] = ticker

        # Chọn các cột cần thiết
        selected_columns = ['symbol', 'listing_last_trading_date', 'listing_ref_price', 'listing_ceiling', 'listing_floor', 'match_match_price', 'bid_ask_bid_1_price', 'bid_ask_ask_1_price']
        df_selected = df[selected_columns]
        logging.info(f"Selected columns: {selected_columns}")
        logging.info(f"Data after selecting columns: {df_selected.head()}")
        return df_selected
    except KeyError as e:
        logging.error(f"Error in selecting columns: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error in selecting columns: {e}")
        return pd.DataFrame()


def fetch_and_select_data(ticker):
    df = fetch_stock_data(ticker)
    if df.empty:
        return df
    df_selected = select_columns(df, ticker)
    return df_selected