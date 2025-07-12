
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import altair as alt
import numpy as np

# Streamlit UI - Title and Upload
st.title("📊 User Feedback Analytics Dashboard for Product Improvement")
st.markdown("Upload your feedback CSV to get data-driven insights, sentiment trends, and feature recommendations.")
uploaded_file = st.file_uploader("📤 Upload your feedback CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
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

    # TF-IDF Keyword Extraction
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = vectorizer.fit_transform(df["Feedback Text"])
    keywords = vectorizer.get_feature_names_out()

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

    # 📊 Sentiment Pie Chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = filtered_df["Sentiment"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)
    plt.close(fig1)

    # 📈 Feature Usage
    st.subheader("📈 Feature Usage Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=filtered_df, x="Feature Used", order=df["Feature Used"].value_counts().index, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    plt.close(fig2)

    # 📅 Sentiment Trend Over Time
    st.subheader("📅 Sentiment Trend Over Time")
    sentiment_over_time = df.groupby([df["Timestamp"].dt.date, "Sentiment"]).size().reset_index(name="Count")
    sentiment_chart = alt.Chart(sentiment_over_time).mark_line(point=True).encode(
        x='Timestamp:T',
        y='Count:Q',
        color='Sentiment:N',
        tooltip=['Timestamp:T', 'Sentiment:N', 'Count:Q']
    ).properties(width=700)
    st.altair_chart(sentiment_chart, use_container_width=True)

    # 🚨 Top 5 Recommendations with %
    st.subheader("🚨 Top 5 Recommendations (by % of Negative Feedback)")
    negative_df = df[df["Sentiment"] == "Negative"]
    top5 = negative_df["Recommendation"].value_counts().head(5)
    total_neg = len(negative_df)
    top5_percent = (top5 / total_neg * 100).round(1).astype(str) + '%'
    rec_df = pd.DataFrame({"Recommendation": top5.index, "Count": top5.values, "Percentage": top5_percent.values})
    st.dataframe(rec_df)

    st.subheader("📊 Top 5 Recommendations Bar Chart")
    chart = alt.Chart(rec_df).mark_bar().encode(
        x=alt.X('Count:Q', title='Count'),
        y=alt.Y('Recommendation:N', sort='-x'),
        tooltip=['Recommendation', 'Count', 'Percentage']
    ).properties(width=700, height=300)
    st.altair_chart(chart)

    # 🌡️ Pain Point Heatmap
    st.subheader("🌡️ Pain Point Heatmap (Negative % by Week & Feature)")
    df['Week'] = df['Timestamp'].dt.to_period('W').astype(str)
    heatmap_data = df[df['Sentiment'] == 'Negative'].groupby(['Week', 'Feature Used']).size().unstack().fillna(0)
    total_per_week = df.groupby('Week').size()
    heatmap_data_percent = heatmap_data.div(total_per_week, axis=0) * 100
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data_percent.T, cmap="Reds", annot=True, fmt=".1f", linewidths=.5, ax=ax3)
    st.pyplot(fig3)

    # 🚀 Post-launch Comparison
    st.subheader("🚀 Sentiment Before vs After Product Launch")
    launch_date = st.date_input("📆 Select Product Launch Date", value=pd.to_datetime(df["Timestamp"].min()).date())
    before = df[df["Timestamp"].dt.date < launch_date]["Sentiment"].value_counts(normalize=True) * 100
    after = df[df["Timestamp"].dt.date >= launch_date]["Sentiment"].value_counts(normalize=True) * 100
    compare_df = pd.DataFrame({"Before Launch": before, "After Launch": after}).fillna(0).round(1)
    st.dataframe(compare_df)

    # 🧠 User Feedback Personas (Clustering)
    st.subheader("🧠 User Feedback Personas")
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    df["Persona Cluster"] = cluster_labels
    persona_summary = df.groupby("Persona Cluster")["Feedback Text"].apply(lambda x: " | ".join(x.head(2)))
    for cluster_id, example in persona_summary.items():
        st.markdown(f"**Persona {cluster_id}**: {example[:300]}...")

    # ⚖️ Feature Prioritization Matrix (Impact vs Effort)
    st.subheader("⚖️ Feature Prioritization Matrix (Impact vs Effort)")
    feature_counts = df[df['Sentiment'] == 'Negative']["Feature Used"].value_counts()
    effort_lookup = {feat: np.random.randint(1, 10) for feat in feature_counts.index}
    prioritization_df = pd.DataFrame({
        "Feature": feature_counts.index,
        "Impact": feature_counts.values,
        "Effort": [effort_lookup[f] for f in feature_counts.index]
    })
    scatter_chart = alt.Chart(prioritization_df).mark_circle(size=150).encode(
        x=alt.X('Effort:Q', scale=alt.Scale(zero=False)),
        y='Impact:Q',
        color='Feature:N',
        tooltip=['Feature', 'Impact', 'Effort']
    ).properties(width=700, height=400)
    st.altair_chart(scatter_chart)

    # 📉 Weekly Sentiment Tracker
    st.subheader("📉 Weekly Sentiment Tracker")
    sentiment_weekly = df.groupby([df["Timestamp"].dt.to_period("W").astype(str), "Sentiment"]).size().reset_index(name="Count")
    sentiment_weekly.columns = ['Week', 'Sentiment', 'Count']
    weekly_chart = alt.Chart(sentiment_weekly).mark_line(point=True).encode(
        x='Week:T',
        y='Count:Q',
        color='Sentiment:N',
        tooltip=['Week:T', 'Sentiment:N', 'Count:Q']
    ).properties(width=700)
    st.altair_chart(weekly_chart)

    # 🔑 Keywords
    st.subheader("🔑 Top Keywords in Feedback (TF-IDF)")
    st.write(", ".join(keywords))

    # 📥 Downloadable Recommendations CSV
    st.subheader("📥 Download Recommendations")
    recs_df = df[df["Recommendation"] != ""][["Feature Used", "Feedback Text", "Recommendation"]]
    csv = recs_df.to_csv(index=False).encode('utf-8')
    st.download_button("📤 Download CSV", data=csv, file_name='recommendations_export.csv', mime='text/csv')
else:
    st.info("👆 Please upload a CSV file to begin analysis.")
