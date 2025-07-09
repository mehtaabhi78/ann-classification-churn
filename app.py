import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

#load the model
#model = tf.keras.models.load_model('langchiainHF/pythonOnlyLearning/nn/ann/ann.h5')
model = tf.keras.models.load_model('ann.h5')

#scaler = pickle.load(open("langchiainHF/pythonOnlyLearning/nn/ann/scaler.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

#with open('langchiainHF/pythonOnlyLearning/nn/ann/label_encoder_gender.pkl','rb') as file:
with open('label_encoder_gender.pkl','rb') as file:
    le_gender=pickle.load(file)

#with open('langchiainHF/pythonOnlyLearning/nn/ann/one_hot_encoder_geo.pkl', 'rb') as file:
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo=pickle.load(file)


#stream lit app
st.title('Customer Churn Prediction')
geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',le_gender.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
number_of_products=st.slider("Number of Products",0,4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#prep input data
input_data={
 'CreditScore':credit_score,
 'Gender':le_gender.transform([gender])[0],
 'Age':age,
 'Tenure':tenure,
 'Balance':balance,
 'NumOfProducts':number_of_products,
 'HasCrCard':has_cr_card,
 'IsActiveMember':is_active_member,
 'EstimatedSalary':estimated_salary
}

#geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df
#geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_df=pd.DataFrame([input_data])
input_df=pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)
#input_df
#input_df['CreditScore'] = input_df['CreditScore'].astype(int)

#input_df.dtypes

input_scaled=scaler.transform(input_df)
##input_scaled
#predict churn

prediction = model.predict(input_scaled)
#print(prediction)
prediction_prob=prediction[0][0]
prediction_prob

if prediction_prob>.5:
    st.write('Churn')
else:
    st.write('Not Churn')