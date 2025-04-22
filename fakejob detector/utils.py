from googlesearch import search
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

def verify_company_and_get_sentiment(profile_text):
    try:
        query = f"{profile_text} company website"
        website = next(search(query, num_results=1), None)
        if not website:
            return 0.0, 0
        response = requests.get(website, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        sentiment = TextBlob(text).sentiment.polarity
        # Check for scam indicators in website text
        scam_keywords = ['work from home', 'no experience', 'urgent hiring', 'registration fee', 'instant approval']
        if any(keyword in text.lower() for keyword in scam_keywords):
            sentiment = min(sentiment, -0.1)  # Lower sentiment for suspicious sites
            return sentiment, 0  # No website_flag for scam-like sites
        return sentiment, 1
    except Exception as e:
        print(f"Error in verify_company_and_get_sentiment: {str(e)}")
        return 0.0, 0

def prepare_features(title, location, profile, description, requirements, industry, sentiment, website_flag):
    combined_text = f"{title} {location} {profile} {description} {requirements} {industry}"
    # Add fraud indicators to text
    fraud_indicators = ''
    if 'fee' in description.lower() or 'urgent' in description.lower() or 'no experience' in description.lower():
        fraud_indicators = ' FRAUD_INDICATOR'
    return [combined_text + fraud_indicators, sentiment, website_flag]