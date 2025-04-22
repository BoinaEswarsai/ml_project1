import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
from utils import verify_company_and_get_sentiment

# Load dataset
df = pd.read_csv('reduced_balanced_job_postings.csv')

# Combine text fields
df['text'] = df[['title', 'location', 'company_profile', 'description', 'requirements', 'industry']].fillna('').agg(' '.join, axis=1)

# Compute sentiment and website_flag
df['sentiment'], df['website_flag'] = zip(*df['company_profile'].apply(verify_company_and_get_sentiment))

# Features and target
X = df[['text', 'sentiment', 'website_flag']]
y = df['fraudulent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, stop_words='english'), 'text'),
        ('num', StandardScaler(), ['sentiment', 'website_flag'])
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

# Save model
dump(model, 'model.joblib')