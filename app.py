import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load data
df = pd.read_csv("smart_kitchen_realistic_data.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sentiment classification
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Feedback Text"].apply(classify_sentiment)

# Recommendation engine
def recommend_action(row):
    if row['Sentiment'] == 'Negative':
        return f"Improve {row['Feature Used']} experience"
    return ""

df["Recommendation"] = df.apply(recommend_action, axis=1)

# Streamlit UI
st.title("🍽️ Smart Kitchen Product Analytics Dashboard")
st.markdown("Analyze user feedback, sentiment trends, and actionable insights.")

# Filters
features = df["Feature Used"].unique().tolist()
sentiments = df["Sentiment"].unique().tolist()

selected_feature = st.selectbox("🔎 Filter by Feature", ["All"] + features)
selected_sentiment = st.selectbox("😊 Filter by Sentiment", ["All"] + sentiments)

filtered_df = df.copy()
if selected_feature != "All":
    filtered_df = filtered_df[filtered_df["Feature Used"] == selected_feature]
if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["Sentiment"] == selected_sentiment]

# Sentiment pie chart
st.subheader("📊 Sentiment Distribution")
sentiment_counts = filtered_df["Sentiment"].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
st.pyplot(fig1)
plt.close(fig1)

# Feature usage bar chart
st.subheader("📈 Feature Usage Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(data=filtered_df, x="Feature Used", order=df["Feature Used"].value_counts().index, ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)
plt.close(fig2)

# Top 5 Recommendations
st.subheader("🚨 Top 5 Improvement Recommendations")
top_recs = df[df["Recommendation"] != ""].groupby("Recommendation").size().sort_values(ascending=False).head(5)
st.table(top_recs.reset_index(name="Count"))
