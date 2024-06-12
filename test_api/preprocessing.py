import pandas as pd
import logging

def preprocess_data(df):
    try:
        # Làm phẳng các cột chỉ số đa cấp nếu cần
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

        # Ghi lại tên các cột sau khi làm phẳng
        logging.info(f"Các cột sau khi làm phẳng: {df.columns}")

        # Bảo toàn cột symbol và chuyển đổi các cột khác sang kiểu số
        symbol_column = df['symbol'].copy()  # Sao chép cột symbol để bảo toàn
        for col in df.columns:
            if col != 'symbol' and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logging.info(f"Giá trị cột listing_last_trading_date trước khi xử lý: {df['listing_last_trading_date']}")


        # Điền các giá trị NaN bằng giá trị ngày tháng hợp lệ (ví dụ: ngày hiện tại)
        df['listing_last_trading_date'] = df['listing_last_trading_date'].ffill().bfill().fillna(0)
        
        df = df.fillna(0)

        # Khôi phục cột symbol ban đầu
        df['symbol'] = symbol_column

        logging.info(f"Dữ liệu sau khi tiền xử lý các cột: {df.columns}")
        logging.info(f"Dữ liệu sau khi tiền xử lý head: {df.head()}")
        return df
    except KeyError as e:
        logging.error(f"Error in preprocessing data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error in preprocessing data: {e}")
        return pd.DataFrame()

