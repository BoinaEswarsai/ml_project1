from flask import Flask, request, render_template
from joblib import load
from utils import verify_company_and_get_sentiment, prepare_features
import pandas as pd

app = Flask(__name__)
model = load("model.joblib")  # Load your trained ML model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        title = request.form.get('title', '')
        location = request.form.get('location', '')
        profile = request.form.get('profile', '')
        description = request.form.get('description', '')
        requirements = request.form.get('requirements', '')
        industry = request.form.get('industry', '')

        # Combine all fields into a single string for text input
        combined_text = f"{title} {location} {profile} {description} {requirements} {industry}"

        # Check for empty input
        if not combined_text.strip():
            return render_template('index.html', result="⚠️ Please enter job details.")

        # Get sentiment and website_flag using verify_company_and_get_sentiment
        sentiment, website_flag = verify_company_and_get_sentiment(profile)

        # Prepare features for the model
        features = prepare_features(title, location, profile, description, requirements, industry, sentiment, website_flag)

        # Convert features list to a pandas DataFrame with correct column names
        features_df = pd.DataFrame([features], columns=['text', 'sentiment', 'website_flag'])

        # Predict using model
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]

        # Interpret result
        if prediction == 1:
            result = f"❌ This job looks FAKE with {round(probability[1]*100, 2)}% confidence."
        else:
            result = f"✅ This job looks REAL with {round(probability[0]*100, 2)}% confidence."

        return render_template('index.html', result=result)

    except Exception as e:
        print(f"Error in predict(): {str(e)}")
        return render_template('index.html', result=f"❌ Something went wrong during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)