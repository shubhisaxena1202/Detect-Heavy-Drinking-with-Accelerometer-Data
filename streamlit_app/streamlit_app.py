# Use: streamlit run streamlit_app.py to execute it
# test file is pushed as well.

import streamlit as st 
import pandas as pd
import pickle
from utils import process_data

class_label = {0: "Sober", 1: "Intoxicated"}
menu = ["SVM","Random Forest"]
choice = st.sidebar.selectbox("Models",menu)


# @st.cache()
def load_model_predict(X_test):
    loaded_model = pickle.load(open('../svm_feature27.pkl', 'rb'))
    result = loaded_model.predict(X_test)
    return result[0]
      
st.markdown("<h1 style='text-align: center; color: blue;'>Detect Heavy Drinking Episodes</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: Black;'>Upload the csv file having x,y,z accelerometer readings recorded at 40Hz (miliseconds level) for 5 seconds</h1>", unsafe_allow_html=True)

data_file = st.file_uploader("Upload CSV",type=["csv"])

generate_btn = st.button("Get Prediction")
if generate_btn:
    if data_file is None:
            st.markdown("<h4 style='text-align: center; color: black;'>Please upload data file</h1>".format(pred_class), unsafe_allow_html=True)
    else:
        df = pd.read_csv(data_file)
        st.dataframe(df)        
        X = process_data(df)
        pred_class = load_model_predict(X)
        st.write(' ')
        st.markdown("<h4 style='text-align: center; color: black;'>Intoxication Status at the provided accelerometer data timestamp:</h1>".format(pred_class),            unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: green;'>{}</h1>".format(class_label[pred_class]), unsafe_allow_html=True)
