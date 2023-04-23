
import pickle
import streamlit as st
import numpy as np

# Load the trained model and feature scaler
model = pickle.load(open('trained_srav.sav','rb'))

st.title('Melbourne Housing Price Predictor')


Regionname = st.slider('Region Name:',1,8,3)
rooms = st.slider('Rooms', 1, 10, 3)
distance = st.slider('Distance', 0, 50, 10)
bathroom = st.slider('Bathroom', 1, 5, 2)
car = st.slider('Car', 0, 11, 1)
Type = st.slider('Type:',0,3,1)
SellerG = st.slider('SellerG',0,111,25)
Propertycount = st.slider('Propertycount:', 0, 1000, 100)
landsize = st.slider('Landsize', 0, 1000, 500)
building_area = st.slider('Building Area', 0, 1000, 500)
year_built = st.slider('Year Built', 1800, 2022, 2000)

def predict():
    float_features = [float(x) for x in [Regionname,rooms,distance,bathroom,car,Type,SellerG,Propertycount,landsize,building_area,year_built]]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    label = prediction[0]
    
    print(type(label))
    print(label)

    st.success('Recommended crop to grow is : ' + str(label) + ' :thumbsup:')
    
trigger = st.button('Predict', on_click=predict)
