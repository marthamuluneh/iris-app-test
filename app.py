import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model - UPDATED with your actual filename
model = joblib.load('iris_lr_model.joblib')

st.title("🌸 Iris Flower Species Prediction")
st.write("Enter the flower measurements to predict the species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

if st.button("🔮 Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = species_map[prediction]
    
    st.success(f"**Predicted Species: {predicted_species}**")
    
    st.write("### Prediction Confidence:")
    prob_df = pd.DataFrame({
        "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "Probability (%)": [round(p * 100, 2) for p in probabilities]
    })
    st.bar_chart(prob_df.set_index("Species"))dex("Species"))
