import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.title("Dự đoán giá cổ phiếu Việt Nam")

symbol = st.text_input("Nhập mã cổ phiếu:")
start_date = st.date_input("Ngày bắt đầu:")
end_date = st.date_input("Ngày kết thúc:")
model_type = st.selectbox("Chọn mô hình:", ["lstm", "random_forest"])

if st.button("Dự đoán"):
    try:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        response = requests.get("http://127.0.0.1:8000/predict", params={
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "model_type": model_type
        })
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        
        if "predictions" in data:
            predictions = data['predictions']
            df = pd.DataFrame(predictions, columns=['Dự đoán'])
            st.line_chart(df)
        else:
            st.error("Không có dự đoán được trả về. Thông báo lỗi: {}".format(data.get("error", "Unknown error")))
    except requests.exceptions.RequestException as e:
        st.error("Lỗi khi kết nối đến server: {}".format(e))
    except ValueError as e:
        st.error("Lỗi khi giải mã dữ liệu JSON: {}".format(e))
    except Exception as e:
        st.error("Đã xảy ra lỗi: {}".format(e))
