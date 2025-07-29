import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title='Personal Loan Prediction', 
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 2px solid #17a2b8;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_lottie(url):
    """Load Lottie animation from URL with error handling"""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except requests.exceptions.RequestException:
        st.warning("Could not load animation. Continuing without it.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    model_files = [
        "best_decision_tree_model.pkl",
        "random_forest_file", 
        "best_model.pkl",
        "loan_prediction_model.pkl"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                st.success(f"✅ Model loaded successfully from {model_file}")
                return model, model_file
            except Exception as e:
                st.error(f"Error loading {model_file}: {str(e)}")
                continue
    
    st.error("❌ No valid model file found. Please ensure the model is saved properly.")
    return None, None

def prepare_input_data(age, exp, income, zipcode, family, ccavg, education, mortgage, online, creditcard):
    """Prepare input data for prediction with proper feature alignment"""
    
    # Convert categorical variables to numerical
    online_num = 1 if online == 'Yes' else 0
    creditcard_num = 1 if creditcard == 'Yes' else 0
    
    # Create feature array based on your model's expected features
    # Adjust this order based on your actual model's feature order
    features = [age, exp, income, zipcode, family, ccavg, education, mortgage, online_num, creditcard_num]
    
    # Convert to numpy array and reshape
    sample = np.array(features).reshape(1, -1)
    
    return sample

def validate_inputs(age, exp, income, zipcode, family, ccavg, education, mortgage):
    """Validate user inputs"""
    errors = []
    
    if age <= 0 or age > 100:
        errors.append("Age must be between 1 and 100")
    
    if exp < 0 or exp > age:
        errors.append("Experience cannot be negative or greater than age")
    
    if income <= 0:
        errors.append("Income must be greater than 0")
    
    if zipcode <= 0:
        errors.append("ZIP Code must be a positive number")
    
    if family <= 0:
        errors.append("Family size must be at least 1")
    
    if ccavg < 0:
        errors.append("Credit Card Average cannot be negative")
    
    if education < 1 or education > 3:
        errors.append("Education level must be between 1 and 3")
    
    if mortgage < 0:
        errors.append("Mortgage cannot be negative")
    
    return errors

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Personal Loan Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, model_name = load_model()
    
    if model is None:
        st.stop()
    
    # Information section
    with st.expander("ℹ️ About This Application", expanded=False):
        st.markdown("""
        <div class="info-box">
        <h4>How it works:</h4>
        <ul>
            <li>This application uses machine learning to predict loan approval probability</li>
            <li>Enter your personal and financial information in the form below</li>
            <li>The model will analyze your data and provide a prediction</li>
            <li>Results are based on patterns learned from historical banking data</li>
        </ul>
        
        <h4>Model Information:</h4>
        <ul>
            <li><strong>Algorithm:</strong> Decision Tree (Best performing model)</li>
            <li><strong>Accuracy:</strong> 98.87%</li>
            <li><strong>Precision:</strong> 93.20%</li>
            <li><strong>Current Model:</strong> {}</li>
        </ul>
        </div>
        """.format(model_name), unsafe_allow_html=True)
    
    # Load animation
    lottie_link = "https://assets8.lottiefiles.com/packages/lf20_ax5yuc0o.json"
    animation = load_lottie(lottie_link)
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Your Information")
        
        # Personal Information
        st.markdown("**Personal Details**")
        name = st.text_input('Full Name:', placeholder="Enter your full name")
        
        col_age, col_exp = st.columns(2)
        with col_age:
            age = st.number_input('Age:', min_value=18, max_value=80, value=30, step=1)
        with col_exp:
            exp = st.number_input('Years of Experience:', min_value=0, max_value=50, value=5, step=1)
        
        # Financial Information
        st.markdown("**Financial Details**")
        col_income, col_mortgage = st.columns(2)
        with col_income:
            income = st.number_input('Annual Income (in thousands):', min_value=1, max_value=500, value=50, step=1)
        with col_mortgage:
            mortgage = st.number_input('Mortgage Amount (in thousands):', min_value=0, max_value=1000, value=0, step=1)
        
        col_ccavg, col_family = st.columns(2)
        with col_ccavg:
            ccavg = st.number_input('Credit Card Average Spending (in thousands):', min_value=0.0, max_value=20.0, value=1.0, step=0.1)
        with col_family:
            family = st.number_input('Family Size:', min_value=1, max_value=10, value=2, step=1)
        
        # Other Details
        st.markdown("**Additional Information**")
        col_zip, col_edu = st.columns(2)
        with col_zip:
            zipcode = st.number_input('ZIP Code:', min_value=10000, max_value=99999, value=12345, step=1)
        with col_edu:
            education = st.selectbox('Education Level:', 
                                   options=[1, 2, 3], 
                                   format_func=lambda x: {1: "Undergraduate", 2: "Graduate", 3: "Advanced/Professional"}[x],
                                   index=1)
        
        col_online, col_cc = st.columns(2)
        with col_online:
            online = st.radio('Online Banking User:', ['Yes', 'No'], index=0)
        with col_cc:
            creditcard = st.radio('Credit Card Holder:', ['Yes', 'No'], index=0)
    
    with col2:
        st.subheader("🎭 Loan Prediction")
        
        # Display animation if available
        if animation:
            st_lottie(animation, speed=1, height=300, key="loan_animation")
        else:
            # Fallback image or text
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h2>🏦</h2>
                <p>Loan Prediction System</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction button
        if st.button('🔮 Predict Loan Approval', type="primary", use_container_width=True):
            if not name.strip():
                st.error("Please enter your name before predicting!")
            else:
                # Validate inputs
                errors = validate_inputs(age, exp, income, zipcode, family, ccavg, education, mortgage)
                
                if errors:
                    st.error("Please fix the following errors:")
                    for error in errors:
                        st.error(f"• {error}")
                else:
                    # Prepare data and make prediction
                    try:
                        with st.spinner("Analyzing your data..."):
                            sample = prepare_input_data(age, exp, income, zipcode, family, ccavg, education, mortgage, online, creditcard)
                            prediction = model.predict(sample)[0]
                            
                            # Get prediction probability if available
                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(sample)[0]
                                confidence = max(probabilities) * 100
                            else:
                                confidence = 85  # Default confidence for models without probability
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-box success-box">
                                <h3>✅ Congratulations, {name}!</h3>
                                <p><strong>Loan Approval: LIKELY</strong></p>
                                <p>Confidence: {confidence:.1f}%</p>
                                <p>Based on your profile, you have a high probability of loan approval!</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                            
                        else:
                            st.markdown(f"""
                            <div class="prediction-box error-box">
                                <h3>❌ Sorry, {name}</h3>
                                <p><strong>Loan Approval: UNLIKELY</strong></p>
                                <p>Confidence: {confidence:.1f}%</p>
                                <p>Based on your current profile, loan approval probability is low. Consider improving your financial metrics.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display input summary
                        with st.expander("📋 Your Input Summary", expanded=False):
                            st.write(f"**Name:** {name}")
                            st.write(f"**Age:** {age} years")
                            st.write(f"**Experience:** {exp} years")
                            st.write(f"**Income:** ${income}k annually")
                            st.write(f"**Family Size:** {family}")
                            st.write(f"**Education:** {['Undergraduate', 'Graduate', 'Advanced/Professional'][education-1]}")
                            st.write(f"**Credit Card Avg:** ${ccavg}k")
                            st.write(f"**Mortgage:** ${mortgage}k")
                            st.write(f"**Online Banking:** {online}")
                            st.write(f"**Credit Card:** {creditcard}")
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.error("Please check your inputs and try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8rem;">
        <p>🔒 Your data is processed securely and not stored permanently.</p>
        <p>This prediction is for informational purposes only and should not be considered as financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
