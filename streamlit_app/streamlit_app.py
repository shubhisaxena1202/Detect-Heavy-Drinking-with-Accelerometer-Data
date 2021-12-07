# Use: streamlit run streamlit_app.py to execute it
# test file is pushed as well.

import streamlit as st 
import pandas as pd
import pickle
from utils import process_data, get_metrics
  
class_label = {0: "Sober", 1: "Intoxicated"}
expt_menu = ["Single Window","Test Bulk"]
expt_choice = st.sidebar.selectbox("Experiments",expt_menu)

model_menu = ["SVM","Random Forest"]
model_choice = st.sidebar.selectbox("Models",model_menu)


# @st.cache()
def load_model_predict(X_test):
    if model_choice == "SVM":
        loaded_model = pickle.load(open('./saved_models/model_svm_5_27.pkl', 'rb'))
    if model_choice == "Random Forest":
        loaded_model = pickle.load(open('./saved_models/model_rf_5_27.pkl', 'rb'))
    scaler = pickle.load(open('./saved_models/scaler.pkl', 'rb'))
    X_test_scaled = scaler.transform(X_test)
    result = loaded_model.predict(X_test_scaled)
    return result
      
st.markdown("<h1 style='text-align: center; color: blue;'>Detect Heavy Drinking Episodes</h1>", unsafe_allow_html=True)

if expt_choice == "Single Window":  
    st.markdown("<h3 style='text-align: center; color: Black;'>Upload the csv file having x,y,z accelerometer readings recorded at 40Hz (miliseconds level) for 5 seconds</h1>", unsafe_allow_html=True)
    data_file = st.file_uploader("Upload CSV",type=["csv"])
    generate_btn = st.button("Get Prediction")
    if generate_btn:
        if not data_file:
                st.markdown("<h4 style='text-align: center; color: black;'>Please upload data file</h1>", unsafe_allow_html=True)
        else:
            df = pd.read_csv(data_file)
            st.dataframe(df)        
            X, _, imp_features = process_data(df, is_bulk=False)
            pred_class = load_model_predict(X[imp_features])[0]
            st.write(' ')
            st.markdown("<h4 style='text-align: center; color: black;'>Intoxication Status at the provided accelerometer data timestamp: using {}</h1>".format(model_choice), unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: green;'>{}</h1>".format(class_label[pred_class]), unsafe_allow_html=True)

if expt_choice == "Test Bulk":      
    st.write(' ')
    st.markdown("<h3 style='text-align: center; color: Black;'>BULK TEST PREDICTIONS using {}</h1>".format(model_choice), unsafe_allow_html=True)
    test_file = st.file_uploader("Upload TEST CSV",type=["csv"])
    test_btn = st.button("Get all predictions")
    if test_btn:
        if not test_file:
                st.markdown("<h4 style='text-align: center; color: black;'>Please upload test file</h1>", unsafe_allow_html=True)
        else:
            test_df = pd.read_csv(test_file)
            X, gold_labels, imp_features = process_data(test_df, is_bulk=True)
            pred_class = pd.DataFrame(load_model_predict(X[imp_features]), columns=["pred"])
            df = pd.concat([X['timestamp'], pred_class, gold_labels], axis=1)
            st.markdown("<h4 style='text-align: center; color: black;'>The Predictions are given below: </h1>", unsafe_allow_html=True)
            st.write("\n")
            st.dataframe(df) 
             
            eval_metrics = get_metrics(pred_class, gold_labels)
            st.markdown("<h4 style='text-align: center; color: black;'>The evaluation metrics are such:</h1>", unsafe_allow_html=True)
            st.write("\n")
            st.dataframe(eval_metrics) 

