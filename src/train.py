import pandas as pd
import joblib
import os
from preprocessing import preprocess, build_pipeline, encode_target

os.makedirs("../models", exist_ok=True)

# Load dataset
df = pd.read_csv("../notebook/data/coin_gecko_2022-03-16.csv")

# Preprocess
df = preprocess(df)

# Features / Target
X = df.drop('volatility', axis=1)
y = df["volatility"]

# Encode target
y_encoded, label_encoder = encode_target(y)

# Build pipeline (scaling + passthrough + model)
pipeline = build_pipeline()

# Train on all data
pipeline.fit(X, y_encoded)

# Save model + label encoder
joblib.dump(pipeline, "../models/final_model.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")

print("Final model + label encoder saved!")
