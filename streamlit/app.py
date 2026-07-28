import streamlit as st 
import numpy as np
import pickle

@st.cache_resource
def load_pickle_model():
    with open('iris_model.pkl', 'rb') as f:
        model, target_names = pickle.load(f)
    return model, target_names

# Load once, use everywhere
model, species_names = load_pickle_model()


# 2. USER INTERFACE

st.title("🌸 Iris Flower Predictor (Pickle Version)")
st.markdown("This app loads a pre-trained `.pkl` model – no retraining happens here!")

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 3.8)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# 3. PREDICT USING THE LOADED MODEL

if st.button("Predict Species"):
    prediction = model.predict(input_data)
    predicted_species = species_names[prediction[0]]
    st.success(f" The model predicts: **{predicted_species.title()}**")