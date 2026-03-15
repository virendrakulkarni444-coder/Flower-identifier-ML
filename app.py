import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Flower Perfume Predictor", page_icon="🌸", layout="centered")
# ---- MODEL TRAINING FUNCTION ----
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 500
    
    flowers = {
        'Rose': {'label': 0, 'perfume_suitable': True, 'pl_mean': 5.0, 'pw_mean': 3.0, 'si_mean': 8.5, 'color': [1, 2]},
        'Jasmine': {'label': 1, 'perfume_suitable': True, 'pl_mean': 2.0, 'pw_mean': 1.0, 'si_mean': 9.0, 'color': [2]},
        'Sunflower': {'label': 2, 'perfume_suitable': False, 'pl_mean': 8.0, 'pw_mean': 4.0, 'si_mean': 2.0, 'color': [3]},
        'Lavender': {'label': 3, 'perfume_suitable': True, 'pl_mean': 1.0, 'pw_mean': 0.5, 'si_mean': 8.0, 'color': [4]},
        'Daisy': {'label': 4, 'perfume_suitable': False, 'pl_mean': 3.0, 'pw_mean': 1.5, 'si_mean': 3.0, 'color': [2, 3]}
    }
    
    data = []
    labels = []
    for name, props in flowers.items():
        for _ in range(n_samples // len(flowers)):
            pl = np.random.normal(props['pl_mean'], 0.5)
            pw = np.random.normal(props['pw_mean'], 0.3)
            si = np.clip(np.random.normal(props['si_mean'], 1.0), 1, 10)
            color = np.random.choice(props['color'])
            data.append([max(0.1, pl), max(0.1, pw), si, color])
            labels.append(props['label'])
            
    label_to_name_mapping = {v['label']: k for k, v in flowers.items()}
    df = pd.DataFrame(data, columns=['PetalLength', 'PetalWidth', 'ScentIntensity', 'ColorCode'])
    df['Flower'] = [label_to_name_mapping[l] for l in labels]
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Flower', axis=1), y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    return model, flowers, df, accuracy
# ---- LOAD DATA AND MODEL ----
model, flowers, df, accuracy = train_model()
label_to_name = {v['label']: k for k, v in flowers.items()}
perfume_map = {k: v['perfume_suitable'] for k, v in flowers.items()}
# ---- UI LAYOUT ----
st.title("🌸 Flower Perfume Predictor")
st.markdown("""
Welcome to the Flower Perfume Project Presentation!
This application uses **Machine Learning** to classify flowers based on their physical attributes and scent, and then determines if they are highly suitable for making perfumes.
""")
tab1, tab2, tab3 = st.tabs(["🧪 Predictor", "📊 Data Insights", "⚙️ Model Details"])
# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("Predict Flower & Perfume Suitability")
    st.write("Enter the characteristics of the flower below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pl = st.slider("Petal Length (cm)", min_value=0.1, max_value=12.0, value=3.0, step=0.1)
        pw = st.slider("Petal Width (cm)", min_value=0.1, max_value=6.0, value=1.5, step=0.1)
        
    with col2:
        si = st.slider("Scent Intensity (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        color_choice = st.selectbox("Flower Color", options=["Red", "White", "Yellow", "Purple"])
        
    color_map_inv = {"Red": 1, "White": 2, "Yellow": 3, "Purple": 4}
    color = color_map_inv[color_choice]
    
    st.markdown("---")
    
    if st.button("🔮 Analyze Flower", use_container_width=True):
        # Prediction
        features = pd.DataFrame([[pl, pw, si, color]], columns=['PetalLength', 'PetalWidth', 'ScentIntensity', 'ColorCode'])
        pred_label = model.predict(features)[0]
        flower_name = label_to_name[pred_label]
        is_suitable = perfume_map[flower_name]
        
        st.subheader("Results:")
        
        rcol1, rcol2 = st.columns(2)
        rcol1.metric("Predicted Species", flower_name)
        
        if is_suitable:
            rcol2.success("✅ HIGHLY suitable for perfumes!")
        else:
            rcol2.error("❌ Not recommended for perfumes.")
# --- TAB 2: DATA INSIGHTS ---
with tab2:
    st.header("Dataset Overview")
    st.write("We synthetically generated 500 samples across 5 flower species based on realistic physical traits.")
    
    st.dataframe(df.sample(5)) # show 5 random rows
    
    st.subheader("Feature Distributions")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='PetalLength', y='ScentIntensity', hue='Flower', palette='Set2')
    plt.title("Petal Length vs Scent Intensity")
    st.pyplot(fig)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Flower', y='ScentIntensity', palette='pastel')
    plt.title("Scent Intensity distribution by Flower")
    st.pyplot(fig2)
# --- TAB 3: MODEL DETAILS ---
with tab3:
    st.header("Model Performance")
    
    st.write(f"**Algorithm:** Random Forest Classifier")
    st.write(f"**Test Set Accuracy:** {accuracy*100:.1f}%")
    
    st.metric(label="Model Accuracy", value=f"{accuracy*100:.1f}%")
    
    st.write("### How It Works:")
    st.markdown("""
    1. **Data Generation:** Synthetic dataset representing the flowers.
    2. **Training Phase:** Random Forest finds patterns linking features (e.g. Lavender has high scent but small petals).
    3. **Prediction Phase:** New inputs are passed into the trained decision trees.
    4. **Perfume Mapping:** After classifying the plant, we use a predefined business-logic map to check if that species is heavily used in the perfume industry.
    """)
