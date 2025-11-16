"""
Train Credit Card Fraud Detection Model
Generates: model.pkl, scaler.pkl
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
print("=" * 70)

# Load dataset
print("\n[1/5] Loading creditcard.csv...")
df = pd.read_csv('creditcard.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Separate features and target
y = df['Class']
X = df.drop('Class', axis=1)

# Handle categorical columns (encode text to numbers)
print("\n[2/5] Encoding categorical features...")
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"   Encoded {len(categorical_cols)} categorical columns")

# Get feature names
FEATURES = list(X.columns)
print(f"   Total features: {len(FEATURES)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Fraud rate: {y.mean()*100:.2f}%")

# Scale features
print("\n[3/5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("   ✅ Saved scaler.pkl")

# Train model
print("\n[4/5] Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("   ✅ Model training complete")

# Evaluate
print("\n[5/5] Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"   True Negatives:  {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives:  {cm[1][1]}")

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n   ROC-AUC Score: {auc:.4f}")

# Save model
joblib.dump(model, 'model.pkl')
print("\n   ✅ Saved model.pkl")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  ✓ model.pkl")
print("  ✓ scaler.pkl")
print("\nRun: python app.py")
print("=" * 70)