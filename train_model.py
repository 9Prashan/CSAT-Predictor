# ==========================
# train_model.py or train_model.ipynb
# ==========================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load cleaned dataset
df = pd.read_csv("Customer_support_data.csv")

# Label encode categorical features
le_channel = LabelEncoder()
le_shift = LabelEncoder()

df['channel_name'] = le_channel.fit_transform(df['channel_name'])
df['Agent Shift'] = le_shift.fit_transform(df['Agent Shift'])

# Save the encoders for later use
joblib.dump(le_channel, "le_channel.pkl")
joblib.dump(le_shift, "le_shift.pkl")

# Define X and y
X = df[['channel_name', 'Agent Shift', 'Item_price']]
y = df['csat_score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, "final_csat_model.pkl")
print("✅ Model and encoders saved successfully.")
