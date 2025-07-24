import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model_and_encoders():
    """
    Loads the trained model and encoders from disk.
    Uses st.cache_resource to prevent reloading on every interaction.
    """
    try:
        model = tf.keras.models.load_model('model.h5')
        with open('label_encoder_gender.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        with open('onehot_encoder_geo.pkl', 'rb') as f:
            onehot_encoder_geo = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, label_encoder_gender, onehot_encoder_geo, scaler
    except FileNotFoundError:
        st.error("Model or encoder files not found. Please ensure 'model.h5', 'label_encoder_gender.pkl', 'onehot_encoder_geo.pkl', and 'scaler.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        st.stop()

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

def load_css():
    """
    Injects custom CSS for a modern, clean UI.
    """
    st.markdown("""
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #000000;
        }

        [data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #E0E0E0;
        }

        h1 {
            color: #1E3A8A; /* Deep blue for main title */
            font-weight: 700;
        }

        h2, h3 {
            color: #374151; 
        }
        
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #000000;
            border-radius: 12px;
            padding: 2rem 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            # border: 1px solid #E0E0E0;
        }
                
        .stButton>button {
            border: none;
            border-radius: 8px;
            color: white;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
            width: 100%; /* Make button take full column width */
            box-shadow: 0 2px 10px rgba(59, 130, 246, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%);
        }

        /* --- Prediction Result Styling --- */
        .result-safe {
            color: #059669; 
            font-size: 22px;
            font-weight: 700;
            text-align: center;
        }

        .result-churn {
            color: #D946EF; /* Red for 'Churn' */
            font-size: 22px;
            font-weight: 700;
            text-align: center;
        }
        
        /* Style the metric value and delta */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
        }
        [data-testid="stMetricDelta"] > div {
            font-size: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()


if 'prediction_prob' not in st.session_state:
    st.session_state.prediction_prob = None
    st.session_state.churn_prediction = None


with st.sidebar:
    st.header("Project Information")
    st.markdown("""
    This application predicts whether a bank customer is likely to churn (leave the bank) based on their account information.

    **Instructions:**
    1.  Enter the customer's details on the main page.
    2.  Click the **"Predict Churn"** button.
    3.  The model's prediction and probability will be displayed.
    
    The prediction is made by a deep learning model (ANN) trained on historical customer data.
    """)
    st.markdown("---")
    st.info("Developed with ❤️ using Streamlit")


# --- Main Application ---
st.title("👤 Customer Churn Prediction")
st.markdown("Provide the customer's details below to get a churn prediction.")

# --- Input Section in a bordered container ---
with st.container(border=True):
    st.subheader("Customer Details")
    
    # Create columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        geography = st.selectbox("🌍 **Geography**", onehot_encoder_geo.categories_[0], help="Customer's country of residence.")
        gender = st.selectbox("🧑 **Gender**", label_encoder_gender.classes_, help="Customer's gender.")
        age = st.slider("🎂 **Age**", 18, 92, 38, help="Customer's age.")

    with col2:
        credit_score = st.number_input("💳 **Credit Score**", min_value=300, max_value=850, value=650, help="Customer's credit score (300-850).")
        balance = st.number_input("💰 **Balance**", value=75000.0, format="%.2f", help="Customer's account balance.")
        estimated_salary = st.number_input("💵 **Estimated Salary**", value=100000.0, format="%.2f", help="Customer's estimated salary.")

    with col3:
        tenure = st.slider("📅 **Tenure (Years)**", 0, 10, 5, help="Number of years the customer has been with the bank.")
        num_of_products = st.slider("📦 **Number of Products**", 1, 4, 1, help="Number of bank products the customer uses.")
        has_cr_card = st.selectbox("✅ **Has Credit Card?**", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the customer have a credit card?")
        is_active_member = st.selectbox("⚡ **Is Active Member?**", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Is the customer an active member?")

    # Centered predict button
    st.markdown("<br>", unsafe_allow_html=True)
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col2:
        if st.button("Predict Churn", key="predict_button"):
            # Prepare the input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary],
            })

            # One-hot encode the geography
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_cols)

            # Concatenate the one-hot encoded geography with the input data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            
            # Reorder columns to match the training data order
            try:
                expected_columns = list(scaler.get_feature_names_out())
                input_data = input_data[expected_columns]
            except Exception as e:
                st.error(f"An error occurred while preparing data for the model: {e}. Please check if the input features match the model's expectations.")
                st.stop()

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)
            st.session_state.prediction_prob = prediction[0][0]
            st.session_state.churn_prediction = st.session_state.prediction_prob > 0.5


# --- Prediction Logic and Display ---
if st.session_state.prediction_prob is not None:
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.session_state.churn_prediction:
                st.metric(label="Churn Probability", value=f"{st.session_state.prediction_prob:.2%}", delta="High Risk", delta_color="inverse")
                st.markdown('<p class="result-churn">🚨 Customer is likely to CHURN</p>', unsafe_allow_html=True)
            else:
                st.metric(label="Retention Probability", value=f"{1 - st.session_state.prediction_prob:.2%}", delta="Low Risk", delta_color="normal")
                st.markdown('<p class="result-safe">✅ Customer is likely to STAY</p>', unsafe_allow_html=True)

        with col2:
            st.write("**Analysis:**")
            if st.session_state.churn_prediction:
                st.warning(f"The model predicts a high churn probability of **{st.session_state.prediction_prob:.1%}**. This indicates a significant risk of the customer leaving. Consider proactive retention strategies.")
            else:
                st.success(f"The model predicts a low churn probability. The likelihood of the customer staying is **{1 - st.session_state.prediction_prob:.1%}**. The customer appears to be satisfied.")
            st.info("This prediction is based on the provided data and the patterns learned by the model. It should be used as a guide for decision-making.")
