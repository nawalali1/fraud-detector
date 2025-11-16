"""
Convert German Credit ARFF dataset to creditcard.csv format
Run this ONCE to generate creditcard.csv
"""
import pandas as pd
from scipy.io import arff

print("Converting dataset_31_credit-g.arff to creditcard.csv...")

# Load ARFF file
data, meta = arff.loadarff('data/dataset_31_credit-g.arff')

# Convert to DataFrame
df = pd.DataFrame(data)

# Decode bytes to strings for all object columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.decode('utf-8')

# Rename columns to match credit card fraud convention (V1, V2,..)
column_mapping = {
    'checking_status': 'Time',
    'duration': 'V1',
    'credit_history': 'V2', 
    'purpose': 'V3',
    'credit_amount': 'Amount',
    'savings_status': 'V4',
    'employment': 'V5',
    'installment_commitment': 'V6',
    'personal_status': 'V7',
    'other_parties': 'V8',
    'residence_since': 'V9',
    'property_magnitude': 'V10',
    'age': 'V11',
    'other_payment_plans': 'V12',
    'housing': 'V13',
    'existing_credits': 'V14',
    'job': 'V15',
    'num_dependents': 'V16',
    'own_telephone': 'V17',
    'foreign_worker': 'V18',
    'class': 'Class'
}

df = df.rename(columns=column_mapping)

# Convert 'good' -> 0 (legitimate), 'bad' -> 1 (fraud)
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})

# Save as creditcard.csv
df.to_csv('data/creditcard.csv', index=False)

print("SUCCESS! Created creditcard.csv")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"\nTarget distribution:")
print(f"   0 (Legitimate): {(df['Class'] == 0).sum()}")
print(f"   1 (Fraud):      {(df['Class'] == 1).sum()}")
print("\n Now run: python train_model.py")