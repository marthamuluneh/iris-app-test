import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('iris_logistic_regression_model.joblib')

st.title("🌸 Iris Flower Species Prediction")
st.markdown("### Enter the flower measurements below:")

# Sliders for input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

if st.button("Predict Species"):
    # Prepare input for the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = species_map[prediction]
    
    st.success(f"**Predicted Species: {predicted_species}**")
    
    # Show probabilities as bar chart
    st.write("### Prediction Confidence:")
    prob_df = pd.DataFrame({
        "Species": list(species_map.values()),
        "Probability (%)": [round(p * 100, 2) for p in probabilities]
    })
    st.bar_chart(prob_df.set_index("Species"))
