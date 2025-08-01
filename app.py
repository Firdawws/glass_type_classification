import pandas as pd
import numpy as np
import streamlit as st
import joblib

#load out saved components
model =joblib.load('gradient_boosting_model.pkl')
label_encoder=joblib.load('label_encoder.pkl')
scaler=joblib.load('scaler (1).pkl')

st.title('Glass Type Prediction App')
st.write('This model predicts the glass type according to the input features below. Please enter your vaues:')


RI=st.slider('Reractive Index:', 1.5,1.8)
NI=st.slider('Sodium:',10.0,18.0)
Mg=st.slider('Magnesium:',0.0,4.0)
AI=st.slider('Aluminium:',0.0,4.0)
Si=st.slider('Silicon:',70.0,88.0)
K=st.slider('Potassium:',0.0,0.5)
Ca=st.slider('Calcium:',5.0,10.0)
Ba=st.slider('Barium:',0.0,5.0)
Fe=st.slider('Iron:',0.0,5.0)


#preparing input features for model
features = np.array([[RI,NI,Mg,AI,Si,K,Ca,Ba,Fe]])
scaled_features=scaler.transform(features)

#predictions
if st.button('Predict Glass Type'):
    Predictions_encoded=model.predict(scaled_features)
    Predictions_label=label_encoder.inverse_transform(Predictions_encoded)[0]


    st.success(f'Predicted glass type: {Predictions_label}')

    