
# User Feedback Analytics Dashboard for Product Improvement

This is a data-driven dashboard built using **Streamlit** to analyze and visualize user feedback for any digital product. The dashboard provides actionable insights into feature usage, user sentiment trends, and improvement recommendations using natural language processing and clustering techniques.

---

##  Features

- Upload your own CSV feedback data for instant analysis  
- Sentiment classification using VADER  
- Keyword extraction using TF-IDF  
- Persona clustering using KMeans  
- Feature prioritization via impact-effort matrix  
- Trend charts and heatmaps for post-launch feedback tracking  
- Downloadable CSV for top actionable recommendations  

---

##  File Structure

```
product-feedback-dashboard/
├── app.py                         # Main Streamlit dashboard
├── smart_kitchen_realistic_data.csv  # Sample product feedback dataset
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation
```

---
##  How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/product-feedback-dashboard.git
cd product-feedback-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

> The dashboard will open in your browser at `http://localhost:8501`.

---

##  Deployment

Deploy your app on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to your GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub and deploy the app using `app.py`.

---

##  Technologies Used

- Python  
- Streamlit  
- Pandas & NumPy  
- VADER (Sentiment Analysis)  
- TF-IDF Vectorization  
- KMeans Clustering  
- Altair, Matplotlib & Seaborn (Visualization)

---

## Feedback

This dashboard is built to support product teams in data-driven decision-making. For feedback, feature requests, or suggestions, feel free to open an issue or fork the repository.

---

##  License

MIT License — free to use with proper attribution.
