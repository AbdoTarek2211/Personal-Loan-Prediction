import streamlit as st
import requests
import joblib
import numpy as np
from streamlit_lottie import st_lottie
from PIL import Image
st.set_page_config(page_title='Loan Prediction', page_icon = "random")
st.title('Personal Loan Prediction')
def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
def prepare_input_data(age,exp,Income,Zipc,Family,ccavg,education,mortgage,online,CreditCard):
    if online == 'Yes':
        on = 1
    else:
        on = 0
    if CreditCard == 'Yes':
        CC = 1
    else:
        CC = 0
    A = [age,exp,Income,Zipc,Family,ccavg,education,mortgage,on,CC]
    sample = np.array(A).reshape(-1,len(A))
    return sample

loaded_model = joblib.load(open("random_forest_file", 'rb'))



st.write('# Loan Prediction Deployment')

lottie_link = "https://assets8.lottiefiles.com/packages/lf20_ax5yuc0o.json"
animation = load_lottie(lottie_link)

st.write('---')
st.subheader('Enter your details to predict your Loan Status')

with st.container():
    
    right_column, left_column = st.columns(2)
    
    with right_column:
        name = st.text_input('Name: ')

        age = st.number_input('Age: ')
        
        exp = st.number_input('Experience: ', min_value=0, max_value=100)
        
        Income = st.number_input('Income : ', min_value=0, max_value=100000)

        Zipc = st.number_input('ZIP Code : ')

        Family = st.number_input('Family : ', min_value=0, max_value=10)
        
        ccavg = st.number_input('CCAvg : ',min_value = 0.0, max_value = 10.0)
        
        education = st.number_input('Education : ',min_value = 0,max_value = 10)
        
        mortgage = st.number_input('Mortgage : ',min_value = 0,max_value = 1000)
        
        online = st.radio('Online : ',['Yes','No'])
        
        CreditCard = st.radio('CreditCard : ',['Yes','No'])
        
        sample = prepare_input_data(age,exp,Income,Zipc,Family,ccavg,education,mortgage,online,CreditCard)

    with left_column:
        st_lottie(animation, speed=1, height=400, key="initial")
if st.button('Predict'):
    pred_Y = loaded_model.predict(sample)
    
    if pred_Y == 1:
        #st.write("## Predicted Status : ", result)
        st.write('### Congratulations ', name)
        st.balloons()
    else:
        st.write('### Sorry', name)