import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    img {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load('iris_lr_model.joblib')

model = load_model()

# Title
st.title("🌸 Iris Flower Classifier")
st.markdown("Predict the species of an iris flower based on its measurements.")

# Input layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
    sepal_width  = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width  = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

st.markdown("---")

# Prediction
if st.button("🔮 Predict Species"):
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    with st.spinner("Analyzing flower..."):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

    # Species + image URLs
    species_map = {
        0: ("🌱 Iris-setosa", "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg"),
        1: ("🌿 Iris-versicolor", "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"),
        2: ("🌼 Iris-virginica", "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg")
    }

    predicted_species, image_url = species_map[prediction]

    # Layout: image + result
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image_url, caption="Predicted Flower", use_container_width=True)

    with col2:
        st.markdown(f"""
            <div style="
                background-color:#ffffff;
                padding:20px;
                border-radius:10px;
                box-shadow:0 2px 10px rgba(0,0,0,0.1);
                text-align:center;">
                <h3>Prediction Result</h3>
                <h2 style="color:#6c63ff;">{predicted_species}</h2>
            </div>
        """, unsafe_allow_html=True)

    # Confidence chart
    st.markdown("### 📊 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "Probability (%)": [round(p * 100, 2) for p in probabilities]
    }).set_index("Species")

    st.bar_chart(prob_df)

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Demo")
