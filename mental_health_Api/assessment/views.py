from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pickle
import joblib
import numpy as np
import pandas as pd
import os

# Load the trained model
# Get base dir safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR,'C:\\Users\\HP\\Desktop\\Project AI-Powered Mental Health.pkl')

# Debug print
print(f"📁 Loading model from: {model_path}")

# Check file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

# Load model
model = joblib.load(model_path)             


# Define features in same order as training
FEATURE_ORDER = [
    'Age', 'Gender', 'self_employed', 'family_history',
    'work_interfere', 'no_employees', 'remote_work',
    'tech_company', 'benefits'
]

class MentalHealthPrediction(APIView):
    def post(self, request):
        try:
            user_input = request.data
            df = pd.DataFrame([user_input])  # Convert to DataFrame
            df.fillna('Unknown', inplace=True)

            # One-hot encode input like training
            df = pd.get_dummies(df)
            model_features = model.feature_names_in_
            for col in model_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[model_features]  # Reorder columns

            prediction = model.predict(df)[0]

            # Return recommendation
            recommendation = (
                "We recommend professional therapy and support apps like Calm or Headspace."
                if prediction == 1 else
                "You're doing well. Keep practicing mindfulness and self-care regularly."
            )

            return Response({
                "prediction": int(prediction),
                "recommendation": recommendation
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
