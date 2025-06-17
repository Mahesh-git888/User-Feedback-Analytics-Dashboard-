
#  Smart Kitchen Product Analytics Dashboard

This is a data-driven dashboard built using **Streamlit** to analyze and visualize product feedback for a smart kitchen platform like **Kookar AI**. The dashboard provides valuable insights into feature usage, user feedback sentiment, and improvement recommendations based on natural language processing.

---

##  Features

-  Interactive analytics of user feedback data  
-  Sentiment analysis using VADER  
-  Actionable recommendations based on negative feedback  
-  Visualizations of feature usage and sentiment trends  
-  Built with Streamlit for quick deployment and demo  

---

##  File Structure

```

smart-kitchen-product-analytics/
├── app.py                          # Main dashboard app
├── smart\_kitchen\_realistic\_data.csv  # Dummy product feedback data
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation

````

---

##  How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-kitchen-product-analytics.git
cd smart-kitchen-product-analytics
````

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

##  Deployment (Optional)

You can deploy this app easily using [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this project to your GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Connect your GitHub account and select the repo.
4. Set `app.py` as the main file and deploy!

---

##  Technologies Used

* Python 
* Streamlit 
* Pandas 
* VADER Sentiment Analysis 
* Matplotlib & Seaborn for Visualization 

---

##  Feedback

This dashboard is inspired by real-world product feedback workflows for AI startups like **Kookar AI**. For suggestions, feel free to open an issue or fork the repo.

---

##  License

MIT License. Use freely with attribution.

```
.
