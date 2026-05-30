import streamlit as st
import os
import numpy as np

st.title("RescueVision Lab Demo")

if st.button("Run pipeline"):
    os.system("python detect.py")
    st.image("data/detect_out.jpg")

    os.system("python motion.py")
    st.line_chart(np.load("data/motion_trace.npy"))

    os.system("python pose.py")

    os.system("python depth.py")
    st.image("data/depth.png")

    os.system("python fuse.py")
