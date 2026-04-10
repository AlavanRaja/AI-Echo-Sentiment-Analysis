# 🤖 AI Echo: Your Smartest Conversational Partner

Sentiment Analysis on ChatGPT User Reviews using NLP and Machine Learning

## 📌 Project Overview

**AI Echo** is a Natural Language Processing (NLP) project that analyzes user reviews of a ChatGPT application and classifies them into three sentiment categories:

- ✅ Positive
- ⚠️ Neutral
- ❌ Negative

This project uses **text preprocessing**, **TF-IDF vectorization**, and **Machine Learning models** to understand customer feedback and present insights through a **Streamlit web application**.

---

## 🎯 Problem Statement

Sentiment analysis is an NLP technique used to determine the emotional tone of text. In this project, user reviews of a ChatGPT application are analyzed and classified as positive, neutral, or negative.

The main objective is to:
- Understand customer satisfaction
- Identify user concerns
- Discover improvement areas
- Build an interactive dashboard for prediction and insights

---

## 💼 Business Use Cases

- **Customer Feedback Analysis:** Understand what users like and dislike.
- **Brand Reputation Management:** Track how users feel about the application.
- **Feature Enhancement:** Identify recurring complaints and improvement needs.
- **Automated Support Prioritization:** Detect negative reviews quickly.
- **Marketing Strategy Optimization:** Use sentiment trends for better decisions.

---

## 🧠 Project Objectives

- Perform text preprocessing on user reviews
- Convert text into machine-readable features
- Train sentiment classification models
- Compare model performance
- Generate visual insights from the dataset
- Build a Streamlit app for real-time sentiment prediction
- Answer key business questions from the review data

---

## 📂 Project Structure

```bash
AI_Echo_Project/
│
├── app.py
├── sentiment_analysis.ipynb
├── chatgpt_style_reviews_dataset.xlsx
├── cleaned_reviews.csv
├── model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

The dataset contains **500 reviews** and **12 columns**.

| Column | Description |
|---|---|
| `date` | Date when the review was submitted |
| `title` | Short title of the review |
| `review` | Full review text |
| `rating` | Rating given by the user (1 to 5) |
| `username` | Reviewer's username |
| `helpful_votes` | Number of helpful votes |
| `review_length` | Length of the review |
| `platform` | Platform where review was posted |
| `language` | Review language |
| `location` | User location |
| `version` | App/version number |
| `verified_purchase` | Whether the user is verified or not |

---

## 🧹 Data Preprocessing

The preprocessing steps used in this project are:

- Convert text to lowercase
- Remove URLs
- Remove special characters and numbers
- Tokenize text
- Remove stopwords
- **Keep negation words** like `not`, `no`, and `never` because they affect sentiment
- Create cleaned review text for modeling

### Example

| Original Review | Cleaned Review |
|---|---|
| Not satisfied, many bugs and issues. | not satisfied many bugs issues |
| Amazing quality and user-friendly interface. | amazing quality userfriendly interface |

---

## 🏷️ Sentiment Labeling

The `rating` column is converted into sentiment classes:

- **Positive** → Ratings **4 and 5**
- **Neutral** → Rating **3**
- **Negative** → Ratings **1 and 2**

### Labeling Logic

```python
def label_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'
```

---

## 📈 Exploratory Data Analysis (EDA)

The following visualizations were created:

1. Sentiment Distribution
2. Rating Distribution
3. Average Rating Over Time
4. Platform vs Average Rating
5. Review Length by Sentiment
6. Verified Purchase vs Average Rating
7. Top Locations by Average Rating
8. Top Keywords in Positive vs Negative Reviews

These help understand the structure of the dataset and user behavior patterns.

---

## 🔤 Feature Engineering

The cleaned review text is converted into numerical features using **TF-IDF Vectorization**.

### TF-IDF Settings

- `max_features = 5000`
- `ngram_range = (1, 2)`

This means the model uses:
- Single words
- Two-word combinations

---

## 🤖 Machine Learning Models Used

Three models were trained and compared:

1. **Naive Bayes**
2. **Logistic Regression**
3. **Random Forest Classifier**

### Why These Models?

- **Naive Bayes** works well for text classification
- **Logistic Regression** is simple and effective for NLP tasks
- **Random Forest** helps compare performance with an ensemble model

---

## 📊 Model Evaluation

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- AUC-ROC Score

### Results

| Model | Accuracy |
|---|---|
| Naive Bayes | 1.0000 |
| Logistic Regression | 1.0000 |
| Random Forest | 1.0000 |

### Best Model

**Naive Bayes** was selected as the final model.

### Note on Accuracy

The dataset contains highly structured review patterns, so the model achieved very high accuracy. In real-world data, the accuracy may be lower due to more complex and noisy reviews.

---

## 📌 Key Insights Answered

This project answers the following 10 business questions:

1. What is the overall sentiment of user reviews?
2. How does sentiment vary by rating?
3. How has sentiment changed over time?
4. Do verified users leave more positive or negative reviews?
5. Are longer reviews more likely to be negative or positive?
6. Which platform gets better reviews?
7. Which locations show the most positive or negative sentiment?
8. Is there a difference in sentiment across platforms?
9. Which ChatGPT versions are associated with higher or lower sentiment?
10. What are the most common negative feedback themes?

---

## 💻 Streamlit Application Features

The Streamlit app includes 5 sections:

### 1. Home
- Project title
- Summary
- KPI cards
- Dataset preview

### 2. EDA Dashboard
- Visual charts for review analysis

### 3. Predict Sentiment
- User can type a new review
- App predicts Positive / Neutral / Negative

### 4. Bulk Analysis
- Upload CSV or Excel file
- Predict sentiment for multiple reviews

### 5. Key Insights
- Business insight charts and findings
- Answers to all 10 questions

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Streamlit**
- **Openpyxl**

---

## ⚙️ Installation

### Step 1: Clone the Repository

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Notebook

Open `sentiment_analysis.ipynb` in VS Code or Jupyter Notebook and run all cells.

This will generate:
- `cleaned_reviews.csv`
- `model.pkl`
- `tfidf_vectorizer.pkl`

### Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📦 requirements.txt

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
openpyxl
```

---

## ▶️ How It Works

1. Load the dataset
2. Clean the review text
3. Create sentiment labels from ratings
4. Perform EDA
5. Convert text using TF-IDF
6. Train ML models
7. Compare results
8. Save the best model
9. Build and run the Streamlit app

---

## 🧪 Sample Prediction Examples

| Review | Expected Sentiment |
|---|---|
| This app is amazing and very useful. | Positive |
| The app is okay, but needs improvement. | Neutral |
| Very bad experience, too many bugs. | Negative |

---

## 🎓 Viva Preparation Points

You can explain the following during viva:

- Why sentiment analysis is useful
- Why preprocessing is important
- Why negation words should not be removed
- Why TF-IDF is used
- Why Naive Bayes performs well for text classification
- Why multiple models were compared
- Why Streamlit was chosen for deployment

---

## 🚀 Future Improvements

Possible future enhancements:

- Use WordCloud for better text visualization
- Add deep learning models like LSTM or BERT
- Deploy online using Streamlit Cloud or AWS
- Add multilingual sentiment analysis
- Perform topic modeling on negative reviews

---

## 👨‍💻 Author

**Abishek Raja**  
Data Science Student  

---

## 📄 License

This project is created for **educational purposes only** as part of a Data Science academic submission.

---

## 🙌 Acknowledgement

This project was developed as part of a data science assignment on NLP, sentiment analysis, machine learning, and dashboard deployment.

If you find this project useful, consider giving it a ⭐ on GitHub.