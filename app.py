import pandas as pd
import joblib
import streamlit as st

# Load the trained model and feature scaler
model = joblib.load('C:/Users/hello/OneDrive/Desktop/ML Project/price_prediction.pkl')
scaler = joblib.load('C:/Users/hello/OneDrive/Desktop/ML Project/melbourne_housing_scaler.pkl')

# Define a function to preprocess the input data
def preprocess_input(data):
    # Select the relevant features
    selected_features = ['Regionname','Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt','Type','SellerG','Propertycount']
    data = data[selected_features]
    
    # Apply feature scaling using the pre-trained scaler
    data = scaler.transform(data)
    
    return data

# Define the Streamlit app
def app():
    # Set the app title
    st.title('Melbourne Housing Price Predictor')
    
    # Define the input form
    st.sidebar.title('Input Features')
    Regionname = st.text_input('Region Name:')
    rooms = st.sidebar.slider('Rooms', 1, 10, 3)
    distance = st.sidebar.slider('Distance', 0, 50, 10)
    bathroom = st.sidebar.slider('Bathroom', 1, 5, 2)
    car = st.sidebar.slider('Car', 0, 4, 1)
    Type = st.selectbox('Type:',('H','t','u'))
    SellerG = st.selectbox('SellerG',('Agency','Agent'))
    Propertycount = st.sidebar.slider('Propertycount:', 0, 1000, 100)
    landsize = st.sidebar.slider('Landsize', 0, 1000, 500)
    building_area = st.sidebar.slider('Building Area', 0, 1000, 500)
    year_built = st.sidebar.slider('Year Built', 1800, 2022, 2000)
    
    # Create a pandas DataFrame with the input data
    input_data = pd.DataFrame({
        'Rooms': [rooms],
        'Distance': [distance],
        'Bathroom': [bathroom],
        'Car': [car],
        'Landsize': [landsize],
        'BuildingArea': [building_area],
        'YearBuilt': [year_built],
        'Regionname':[Regionname],
        'Type':[Type],
        'SellerG':[SellerG],
        'Propertycount':[Propertycount]
       
    })
    
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data)
    
    # Make predictions using the pre-trained model
    predictions = model.predict(preprocessed_data)[0]
    
    # Display the predicted price
    st.subheader('Predicted Price')
    st.write('${:,.2f}'.format(predictions))

if __name__ == '__main__':
    app()
