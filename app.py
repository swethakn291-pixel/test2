import pickle

import pandas as pd
import streamlit as st

MODEL_PATH = "sales_prediction_model.PKL"
FEATURES = ["youtube", "facebook", "newspaper"]

st.set_page_config(page_title="Sales Predictor", page_icon="📈", layout="centered")


@st.cache_resource
def load_model(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Fallback in case the model was saved with joblib
        import joblib
        return joblib.load(path)


st.title("📈 Sales Prediction App")
st.write("Enter your advertising budgets to predict expected sales.")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Place it in the same folder as this app.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    youtube = st.number_input("YouTube budget", min_value=0.0, value=100.0, step=10.0)
with col2:
    facebook = st.number_input("Facebook budget", min_value=0.0, value=50.0, step=10.0)
with col3:
    newspaper = st.number_input("Newspaper budget", min_value=0.0, value=30.0, step=10.0)

if st.button("Predict Sales", type="primary"):
    input_df = pd.DataFrame([[youtube, facebook, newspaper]], columns=FEATURES)
    try:
        prediction = model.predict(input_df)
        result = float(prediction[0])
        st.success(f"Predicted Sales: **{result:,.2f}**")
        st.subheader("Input summary")
        st.dataframe(input_df, hide_index=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
