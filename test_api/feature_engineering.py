import pandas as pd
import logging

def feature_engineering(df):
    try:
        # Kiểm tra xem cột match_match_price có tồn tại không
        if 'match_match_price' not in df.columns:
            raise KeyError("'match_match_price' not found in DataFrame columns")
        
        # Ví dụ về kỹ thuật tính năng: tính toán biến động giá và giá trung bình
        df['price_movement'] = df['match_match_price'] - df['listing_ref_price']
        df['average_price'] = (df['bid_ask_bid_1_price'] + df['bid_ask_ask_1_price']) / 2

        # Thêm các logic kỹ thuật tính năng khác nếu cần

        logging.info(f"Data after feature engineering columns: {df.columns}")
        logging.info(f"Data after feature engineering head: {df.head()}")
        return df
    except KeyError as e:
        logging.error(f"Error in feature engineering data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error in feature engineering data: {e}")
        return pd.DataFrame()
