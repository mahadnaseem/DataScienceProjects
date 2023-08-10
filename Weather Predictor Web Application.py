import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image

st.set_page_config(page_title='Weather Prediction',
    layout='wide')

cloudy = Image.open('C:\\Users\\mahad\\Desktop\\ml-app-main\\cloudy.png') 
clear = Image.open('C:\\Users\\mahad\\Desktop\\ml-app-main\\sunny.png') 
foggy = Image.open('C:\\Users\\mahad\\Desktop\\ml-app-main\\foggy.png') 


st.set_option('deprecation.showPyplotGlobalUse', False)

navbar = st.container()

with navbar:
    col1, col2, col3 = st.columns(3)
    col2.image("logo.png")

# Load the dataset
df = pd.read_csv("C:\\Users\\mahad\\Desktop\\ml-app-main\\weather_clean.csv")  # Replace "dummy_dataset.csv" with your dataset file

# Split the dataset into features and target
X = df.drop(['Summary'],axis=1)
Y = df.Summary

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, Y_train)

# Sidebar
st.sidebar.markdown("<p style='font-size: 24px;'><b>Model Information:</p></b>", unsafe_allow_html=True)
st.sidebar.markdown("**Machine Learning Model:** RandomForestClassifier")
st.sidebar.markdown("**Dataset:** Weather Dataset")

# Model statistics
st.sidebar.subheader("Model Statistics:")
st.sidebar.markdown("**Training Accuracy**: {:.2f}%".format(accuracy_score(Y_train, model.predict(X_train)) * 100))
st.sidebar.markdown("**Testing Accuracy**: {:.2f}%".format(accuracy_score(Y_test, model.predict(X_test)) * 100))

# Confusion matrix
st.sidebar.subheader("Confusion Matrix:")
cm = confusion_matrix(Y_test, model.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.sidebar.pyplot()


# Streamlit web application
def main():

    # Set the title and description of the app
    # st.write('Enter the following features to predict the weather:')
    st.markdown("<p style='font-size: 18px;'><b>Enter the following features to predict the weather:<b></p>", unsafe_allow_html=True)


    # Input features
    temperature = st.number_input('Temperature (C)', min_value=-20.0, max_value=40.0, format="%.1f", help="Enter a value between -20.0 and 40.0")
    humidity = st.number_input('Humidity', min_value=0.1, max_value=1.0, format="%.1f", help="Enter a value between 0.1 and 1.0")
    wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0, max_value=80.0, format="%.1f", help="Enter a value between 0.0 and 80.0")
    visibility = st.number_input('Visibility (km)', min_value=0.0, max_value=100.0, format="%.1f", help="Enter a value between 0.0 and 100.0")
    pressure = st.number_input('Pressure', min_value=1000.0, max_value=1050.0, format="%.1f", help="Enter a value between 1000.0 and 1050.0")

    # Make predictions
    if st.button('Predict'):
        if not temperature or not humidity or not wind_speed or not visibility or not pressure:
            st.error("All fields are required")
        else:    
            input_data = [[temperature, humidity, wind_speed, visibility, pressure]]
            prediction = model.predict(input_data)
            pred = ' '.join(prediction)
            # st.write('The predicted weather is:', prediction)
            if pred == 'Cloudy':
                st.image(cloudy)
            elif pred == 'Clear':
                st.image(clear)
            elif pred == 'Foggy':
                st.image(foggy)


# Run the application
if __name__ == '__main__':
    main()
