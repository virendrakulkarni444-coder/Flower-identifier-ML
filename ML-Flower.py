import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# 1. Create a synthetic dataset
# We'll create a dataset of flowers with hypothetical features:
# - PetalLength (cm)
# - PetalWidth (cm)
# - ScentIntensity (1-10 scale)
# - ColorCode (1: Red, 2: White, 3: Yellow, 4: Purple)
# Labels: 0: Rose, 1: Jasmine, 2: Sunflower, 3: Lavender, 4: Daisy
np.random.seed(42)
n_samples = 500
# Generating somewhat realistic feature distributions for each flower
# Features: [PetalLength, PetalWidth, ScentIntensity, ColorCode]
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
df = pd.DataFrame(data, columns=['PetalLength', 'PetalWidth', 'ScentIntensity', 'ColorCode'])
y = np.array(labels)
# 2. Train a Machine Learning Model
print("Training the ML model on the flower dataset...")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
# Evaluate the model to see how well it learned
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully! Accuracy on test data: {accuracy*100:.2f}%\n")
# 3. Create a prediction function
label_to_name = {v['label']: k for k, v in flowers.items()}
perfume_map = {k: v['perfume_suitable'] for k, v in flowers.items()}
def predict_flower_and_perfume(features):
    """
    Given features [PetalLength, PetalWidth, ScentIntensity, ColorCode],
    predicts the flower species and whether it's suitable for perfume.
    """
    df_features = pd.DataFrame([features], columns=['PetalLength', 'PetalWidth', 'ScentIntensity', 'ColorCode'])
    pred_label = model.predict(df_features)[0]
    flower_name = label_to_name[pred_label]
    is_suitable = perfume_map[flower_name]
    
    return flower_name, is_suitable
# 4. Demonstrate with some new test examples
test_samples = [
    [4.8, 3.2, 9.0, 1],  # Looks like a Rose (Red, high scent)
    [2.1, 0.9, 9.5, 2],  # Looks like a Jasmine (White, very high scent)
    [8.5, 4.2, 1.5, 3],  # Looks like a Sunflower (Yellow, low scent)
    [1.2, 0.6, 7.5, 4]   # Looks like a Lavender (Purple, high scent)
]
print("--- Flower Perfume Predictor Demo ---")
for i, sample in enumerate(test_samples, 1):
    name, suitable = predict_flower_and_perfume(sample)
    suitability_text = "YES (Highly Recommended)" if suitable else "NO (Not recommended)"
    
    color_map = {1: "Red", 2: "White", 3: "Yellow", 4: "Purple"}
    color_str = color_map[sample[3]]
    
    print(f"Sample {i}: Length={sample[0]:.1f}cm, Width={sample[1]:.1f}cm, Scent={sample[2]:.1f}/10, Color={color_str}")
    print(f"-> Predicted Flower: {name}")
    print(f"-> Suitable for Perfumes: {suitability_text}\n")
# 5. Interactive Mode
print("--- Interactive Flower Perfume Predictor ---")
while True:
    print("\nEnter flower details (or type 'quit' to exit):")
    try:
        user_input = input("Petal Length (cm): ")
        if user_input.lower() == 'quit': 
            print("Exiting...")
            break
        pl = float(user_input)
        
        pw = float(input("Petal Width (cm): "))
        si = float(input("Scent Intensity (1-10): "))
        print("Color Codes: 1: Red, 2: White, 3: Yellow, 4: Purple")
        color = int(input("Color Code (1-4): "))
        
        if color not in [1, 2, 3, 4]:
            print("Warning: Unknown color code, but proceeding anyway.")
            
        name, suitable = predict_flower_and_perfume([pl, pw, si, color])
        suitability_text = "YES (Highly Recommended)" if suitable else "NO (Not recommended)"
        
        print(f"\n======================================")
        print(f"-> Predicted Flower: {name}")
        print(f"-> Suitable for Perfumes: {suitability_text}")
        print(f"======================================\n")
        
    except ValueError:
        print("Invalid input. Please enter valid numerical values.")
    except KeyboardInterrupt:
        print("\nExiting...")
        break
