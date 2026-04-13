import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('iris_lr_model.joblib')

st.title("🌸 Iris Flower Species Prediction")
st.write("Enter the flower measurements to predict the species.")

# Using number_input instead of slider
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.5, step=0.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

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
    st.bar_chart(prob_df.set_index("Species"))
