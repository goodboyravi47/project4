import streamlit as st
import os
import pandas as pd
import joblib as jb
import time

heading_style = '''
<div style="color:red;" align='center'>
<h1>stroke detection</h1>
</div>
'''
def return_df(gender,
    age,
    hypertension,
    heart_disease,
    ever_married,
	work_type,
	Residence_type,
	avg_glucose_level,
    bmi,
    smoking_status):
    kbn={
    'gender':[gender],
    'age':[age],
    'hypertension':[hypertension],
    'heart_disease':[heart_disease],
    'ever_married':[ever_married],
	'work_type':[work_type],
	'Residence_type':[Residence_type],
    'avg_glucose_level':[avg_glucose_level],
    'bmi':[bmi],
    'smoking_status':[smoking_status]
    }   
    final_df=pd.DataFrame(kbn)
    return final_df


def base_model():
    bmodel=jb.load(os.path.join('finalize_rf_model2.pkl'))
    return bmodel

st.markdown(heading_style, unsafe_allow_html=True)
gender=st.selectbox('Select your gender',['Male','Female'])
age=st.number_input('Enter your Age:',min_value=0)
hypertension=st.slider('Select Hypetension level:',0,1,0)
heart_disease=st.slider('Select Heart-Disases level:',0,1,0)
ever_married=st.selectbox('Marital Status:',['Yes','No'])
work_type=st.selectbox('Selec your work-type:',['Private','Self-employed','Govt_job'])
Residence_type=st.selectbox('Select your residence type:',['Urban','Rural'])
avg_glucose_level=st.number_input('Enter your glucose-level',min_value=0)
bmi=st.number_input('Enter your bmi',min_value=0)
smoking_status=st.selectbox('Select your smoking status:',['formerly smoked','smokes','never smoked'])
df=return_df(gender,
    age,
    hypertension,
    heart_disease,
    ever_married,
	work_type,
	Residence_type,
	avg_glucose_level,
    bmi,
    smoking_status)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	if predictions==1:
		st.write('Approved')
	elif predictions==0:
		st.write('Not Approved')
