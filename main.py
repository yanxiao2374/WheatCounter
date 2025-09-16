import streamlit as st
from PIL import Image
import torch
from predict import predict_count

st.title("🌾 Wheat Counter")

uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上传的图片", use_container_width=True)

    # 预测按钮
    if st.button("开始预测"):
        count = predict_count(image)
        st.success(f"预测结果：{count:.2f}")
