# ============================================================
# AI Echo: Your Smartest Conversational Partner
# Sentiment Analysis Dashboard - Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Echo: Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .positive { color: #2ecc71; font-weight: bold; font-size: 26px; }
    .negative { color: #e74c3c; font-weight: bold; font-size: 26px; }
    .neutral  { color: #f39c12; font-weight: bold; font-size: 26px; }
    </style>
""", unsafe_allow_html=True)

# ── STOPWORDS & CLEAN FUNCTION ───────────────────────────────
STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your',
    'yours','yourself','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs',
    'themselves','what','which','who','whom','this','that','these',
    'those','am','is','are','was','were','be','been','being','have',
    'has','had','having','do','does','did','doing','a','an','the',
    'and','but','if','or','because','as','until','while','of','at',
    'by','for','with','about','between','into','through',
    'during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','both',
    'each','few','more','most','other','some','such',
    'only','own','same','s','t','can','will',
    'just','don','should','now','ve','ll','re', "very"
])

def clean_text(text):
    if pd.isnull(text) or text == "":
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ── LOAD MODEL & DATA ────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open("C:/Users/abish/OneDrive/Desktop/AI_Echo_Project/notebooks/model.pkl", "rb"))
    tfidf = pickle.load(open("C:/Users/abish/OneDrive/Desktop/AI_Echo_Project/notebooks/tfidf_vectorizer.pkl", "rb"))
    return model, tfidf

@st.cache_data
def load_data():
    df = pd.read_excel("C:/Users/abish/OneDrive/Desktop/AI_Echo_Project/data/chatgpt_style_reviews_dataset.xlsx")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['cleaned_review'] = df['review'].apply(clean_text)
    def label_sentiment(r):
        if r >= 4:   return 'Positive'
        elif r == 3: return 'Neutral'
        else:        return 'Negative'
    df['sentiment'] = df['rating'].apply(label_sentiment)
    return df

model, tfidf = load_model()
df = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.markdown("## 🤖 AI Echo")
st.sidebar.markdown("**Your Smartest Conversational Partner**")
st.sidebar.markdown("---")
page = st.sidebar.radio("📌 Navigate", [
    "🏠 Home",
    "📊 EDA Dashboard",
    "🔍 Predict Sentiment",
    "📁 Bulk Analysis",
    "📈 Key Insights"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**📦 Dataset Info**")
st.sidebar.write(f"Total Reviews  : `{len(df)}`")
st.sidebar.write(f"Total Columns  : `{df.shape[1]}`")
st.sidebar.write(f"Platforms      : `{df['platform'].nunique()}`")
st.sidebar.write(f"Locations      : `{df['location'].nunique()}`")

# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🤖 AI Echo: Your Smartest Conversational Partner")
    st.markdown("### Sentiment Analysis on ChatGPT User Reviews")
    st.markdown("---")

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Total Reviews",  len(df))
    col2.metric("⭐ Avg Rating",     round(df['rating'].mean(), 2))
    col3.metric("✅ Positive Reviews", len(df[df['sentiment']=='Positive']))
    col4.metric("❌ Negative Reviews", len(df[df['sentiment']=='Negative']))

    st.markdown("---")

    # About section
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 Project Objective")
        st.info("""
        This project analyzes **ChatGPT user reviews** and classifies
        them as **Positive**, **Neutral**, or **Negative** using
        Natural Language Processing (NLP) and Machine Learning.
        """)
        st.markdown("### 🛠️ Tech Stack")
        st.success("""
        - **Language:** Python
        - **NLP:** TF-IDF Vectorization
        - **Models:** Naive Bayes, Logistic Regression, Random Forest
        - **Deployment:** Streamlit
        - **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn
        """)
    with col_b:
        st.markdown("### 📊 Sentiment Summary")
        colors = ['#2ecc71','#f39c12','#e74c3c']
        sent_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            sent_counts.values,
            labels=sent_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2}
        )
        for t in autotexts:
            t.set_fontsize(12)
            t.set_fontweight('bold')
        ax.set_title("Sentiment Distribution", fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown("### 📋 Sample Data")
    st.dataframe(df[['date','title','review','rating','sentiment','platform','location']].head(10),
                 use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("---")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Sentiment Distribution")
        sent_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(sent_counts.index, sent_counts.values,
                      color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, sent_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+2,
                    str(val), ha='center', fontweight='bold')
        ax.set_ylabel("Count"); ax.set_title("Sentiment Distribution", fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("2️⃣ Rating Distribution (1–5 Stars)")
        rc = df['rating'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(rc.index.astype(str), rc.values,
                      color=['#e74c3c','#e67e22','#f1c40f','#27ae60','#2980b9'],
                      edgecolor='black', width=0.6)
        for bar, val in zip(bars, rc.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+1,
                    str(val), ha='center', fontweight='bold')
        ax.set_xlabel("Star Rating"); ax.set_ylabel("Count")
        ax.set_title("Rating Distribution", fontweight='bold')
        st.pyplot(fig); plt.close()

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3️⃣ Average Rating Over Time")
        df_time = df.groupby(df['date'].dt.to_period('M'))['rating'].mean().reset_index()
        df_time['date'] = df_time['date'].astype(str)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df_time['date'], df_time['rating'],
                marker='o', color='#3498db', linewidth=2)
        ax.fill_between(range(len(df_time)), df_time['rating'],
                        alpha=0.1, color='#3498db')
        ax.set_xticks(range(len(df_time)))
        ax.set_xticklabels(df_time['date'], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Avg Rating"); ax.set_ylim(1, 5)
        ax.set_title("Avg Rating Over Time", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("4️⃣ Platform vs Average Rating")
        plat = df.groupby('platform')['rating'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(plat.index, plat.values,
                      color='#9b59b6', edgecolor='black', width=0.5)
        for bar, val in zip(bars, plat.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.05,
                    f"{val:.2f}", ha='center', fontweight='bold')
        ax.set_ylabel("Avg Rating"); ax.set_ylim(0, 5)
        ax.set_title("Platform vs Avg Rating", fontweight='bold')
        plt.xticks(rotation=15)
        st.pyplot(fig); plt.close()

    # Row 3
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("5️⃣ Review Length by Sentiment")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='sentiment', y='review_length',
                    order=['Positive','Neutral','Negative'],
                    palette={'Positive':'#2ecc71','Neutral':'#f39c12','Negative':'#e74c3c'},
                    ax=ax)
        ax.set_title("Review Length by Sentiment", fontweight='bold')
        ax.set_xlabel("Sentiment"); ax.set_ylabel("Review Length")
        st.pyplot(fig); plt.close()

    with col6:
        st.subheader("6️⃣ Verified Purchase vs Avg Rating")
        vp = df.groupby('verified_purchase')['rating'].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(vp.index, vp.values,
                      color=['#1abc9c','#e74c3c'], edgecolor='black', width=0.4)
        for bar, val in zip(bars, vp.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.05,
                    f"{val:.2f}", ha='center', fontweight='bold')
        ax.set_ylim(0, 5); ax.set_ylabel("Avg Rating")
        ax.set_title("Verified Purchase vs Avg Rating", fontweight='bold')
        st.pyplot(fig); plt.close()

    # Row 4
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("7️⃣ Top 10 Locations by Avg Rating")
        loc_r = df.groupby('location')['rating'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(loc_r.index, loc_r.values, color='#e67e22', edgecolor='black')
        ax.set_ylabel("Avg Rating"); ax.set_ylim(0, 5)
        ax.set_title("Top 10 Locations", fontweight='bold')
        plt.xticks(rotation=30, ha='right', fontsize=8)
        st.pyplot(fig); plt.close()

    with col8:
        st.subheader("8️⃣ ChatGPT Version vs Avg Rating")
        ver = df.groupby('version')['rating'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(ver.index.astype(str), ver.values,
               color='#2980b9', edgecolor='black', width=0.5)
        for bar, val in zip(ax.patches, ver.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.05,
                    f"{val:.2f}", ha='center', fontsize=9)
        ax.set_ylabel("Avg Rating"); ax.set_ylim(0, 5)
        ax.set_title("Version vs Avg Rating", fontweight='bold')
        plt.xticks(rotation=15)
        st.pyplot(fig); plt.close()

    # Row 5 — Keywords
    st.markdown("---")
    st.subheader("9️⃣ Top Keywords: Positive vs Negative Reviews")
    def top_words(texts, n=15):
        words = " ".join(texts).split()
        return Counter(words).most_common(n)

    pos_w = top_words(df[df['sentiment']=='Positive']['cleaned_review'])
    neg_w = top_words(df[df['sentiment']=='Negative']['cleaned_review'])
    w1, c1 = zip(*pos_w)
    w2, c2 = zip(*neg_w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(list(w1)[::-1], list(c1)[::-1], color='#2ecc71', edgecolor='black')
    axes[0].set_title("✅ Positive Reviews — Top Words", fontweight='bold')
    axes[0].set_xlabel("Frequency")
    axes[1].barh(list(w2)[::-1], list(c2)[::-1], color='#e74c3c', edgecolor='black')
    axes[1].set_title("❌ Negative Reviews — Top Words", fontweight='bold')
    axes[1].set_xlabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT SENTIMENT
# ════════════════════════════════════════════════════════════
elif page == "🔍 Predict Sentiment":
    st.title("🔍 Predict Sentiment for a New Review")
    st.markdown("---")

    st.markdown("### ✍️ Enter a Review Below")
    user_input = st.text_area(
        "Type or paste any ChatGPT review here:",
        placeholder="e.g. Amazing app, very helpful and easy to use!",
        height=150
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        predict_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review before analyzing!")
        else:
            cleaned  = clean_text(user_input)
            vector   = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]
            proba    = model.predict_proba(vector)[0]
            classes  = model.classes_

            st.markdown("---")
            col_r, col_p = st.columns(2)

            with col_r:
                st.markdown("### 🎯 Prediction Result")
                if prediction == "Positive":
                    st.markdown('<p class="positive">✅ POSITIVE</p>', unsafe_allow_html=True)
                    st.success("This review expresses a **positive** experience!")
                elif prediction == "Negative":
                    st.markdown('<p class="negative">❌ NEGATIVE</p>', unsafe_allow_html=True)
                    st.error("This review expresses a **negative** experience!")
                else:
                    st.markdown('<p class="neutral">⚠️ NEUTRAL</p>', unsafe_allow_html=True)
                    st.warning("This review expresses a **neutral** experience!")

            with col_p:
                st.markdown("### 📊 Confidence Scores")
                fig, ax = plt.subplots(figsize=(5, 3))
                bar_colors = []
                for c in classes:
                    if c == 'Positive':   bar_colors.append('#2ecc71')
                    elif c == 'Negative': bar_colors.append('#e74c3c')
                    else:                 bar_colors.append('#f39c12')

                bars = ax.barh(classes, proba, color=bar_colors, edgecolor='black')
                for bar, val in zip(bars, proba):
                    ax.text(val + 0.01, bar.get_y()+bar.get_height()/2,
                            f"{val*100:.1f}%", va='center', fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Confidence")
                ax.set_title("Model Confidence", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            st.markdown("---")
            st.markdown("**🧹 Cleaned version of your input:**")
            st.code(cleaned)

    # Quick test examples
    st.markdown("---")
    st.markdown("### 💡 Quick Test Examples")
    examples = {
        "✅ Positive Example": "This app is absolutely amazing, very helpful and easy to use!",
        "❌ Negative Example": "Terrible experience, too many bugs and it crashes constantly.",
        "⚠️ Neutral Example":  "The app is okay, nothing special but does the basic job."
    }
    for label, example in examples.items():
        with st.expander(label):
            st.write(f"**Review:** {example}")
            c = clean_text(example)
            p = model.predict(tfidf.transform([c]))[0]
            emoji = "✅" if p=="Positive" else ("❌" if p=="Negative" else "⚠️")
            st.write(f"**Predicted:** {emoji} {p}")

# ════════════════════════════════════════════════════════════
# PAGE 4 — BULK ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📁 Bulk Analysis":
    st.title("📁 Bulk Review Analysis")
    st.markdown("Upload your own CSV or Excel file with a `review` column.")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload File", type=["csv", "xlsx"])

    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                udf = pd.read_csv(uploaded)
            else:
                udf = pd.read_excel(uploaded)

            if 'review' not in udf.columns:
                st.error("❌ Your file must have a column named `review`!")
            else:
                st.success(f"✅ File uploaded! {len(udf)} reviews found.")
                udf['cleaned']            = udf['review'].apply(clean_text)
                udf['predicted_sentiment'] = model.predict(tfidf.transform(udf['cleaned']))

                col1, col2, col3 = st.columns(3)
                col1.metric("Total",    len(udf))
                col2.metric("Positive", len(udf[udf['predicted_sentiment']=='Positive']))
                col3.metric("Negative", len(udf[udf['predicted_sentiment']=='Negative']))

                st.markdown("### 📊 Sentiment Breakdown")
                fig, ax = plt.subplots(figsize=(5, 3))
                counts = udf['predicted_sentiment'].value_counts()
                ax.bar(counts.index, counts.values,
                       color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='black')
                ax.set_ylabel("Count"); ax.set_title("Predicted Sentiment", fontweight='bold')
                st.pyplot(fig); plt.close()

                st.markdown("### 📋 Results Table")
                st.dataframe(
                    udf[['review','predicted_sentiment']].rename(
                        columns={'review':'Review','predicted_sentiment':'Predicted Sentiment'}
                    ),
                    use_container_width=True
                )

                csv = udf[['review','predicted_sentiment']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="predicted_sentiments.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Upload a CSV or Excel file to get started.")
        st.markdown("**Expected format:**")
        st.dataframe(pd.DataFrame({
            "review": [
                "This app is amazing!",
                "Terrible experience, very buggy.",
                "It's okay, nothing special."
            ]
        }))

# ════════════════════════════════════════════════════════════
# PAGE 5 — KEY INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📈 Key Insights":
    st.title("📈 Key Sentiment Insights")
    st.markdown("---")

    # Q1 Overall sentiment
    st.markdown("### ❓ Q1: What is the overall sentiment of user reviews?")
    sent_pct = df['sentiment'].value_counts(normalize=True).round(3) * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Positive", f"{sent_pct.get('Positive', 0):.1f}%")
    col2.metric("⚠️ Neutral",  f"{sent_pct.get('Neutral', 0):.1f}%")
    col3.metric("❌ Negative", f"{sent_pct.get('Negative', 0):.1f}%")

    # Q2 Sentiment vs Rating
    st.markdown("---")
    st.markdown("### ❓ Q2: How does sentiment vary by rating?")
    cross = pd.crosstab(df['rating'], df['sentiment'])
    fig, ax = plt.subplots(figsize=(8, 4))
    cross.plot(kind='bar', ax=ax,
               color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='black', rot=0)
    ax.set_title("Rating vs Sentiment", fontweight='bold')
    ax.set_xlabel("Star Rating"); ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Q3 Sentiment over time
    st.markdown("---")
    st.markdown("### ❓ Q3: How has sentiment changed over time?")
    df_sent_time = df.groupby([df['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
    df_sent_time.index = df_sent_time.index.astype(str)
    fig, ax = plt.subplots(figsize=(12, 4))
    for col, color in zip(['Positive','Neutral','Negative'],['#2ecc71','#f39c12','#e74c3c']):
        if col in df_sent_time.columns:
            ax.plot(df_sent_time.index, df_sent_time[col],
                    marker='o', label=col, color=color, linewidth=2)
    ax.set_title("Sentiment Trend Over Time", fontweight='bold')
    ax.set_xlabel("Month"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Q4 Verified vs Unverified
    st.markdown("---")
    st.markdown("### ❓ Q4: Are verified users more satisfied?")
    vp_sent = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index').round(3) * 100
    st.dataframe(vp_sent.style.format("{:.1f}%"), use_container_width=True)

    # Q5 Platform sentiment
    st.markdown("---")
    st.markdown("### ❓ Q5: Sentiment difference across platforms?")
    plat_sent = pd.crosstab(df['platform'], df['sentiment'])
    fig, ax = plt.subplots(figsize=(9, 4))
    plat_sent.plot(kind='bar', ax=ax,
                   color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='black', rot=15)
    ax.set_title("Platform vs Sentiment", fontweight='bold')
    ax.set_xlabel("Platform"); ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Q6 Version sentiment
    st.markdown("---")
    st.markdown("### ❓ Q6: Which ChatGPT version got best sentiment?")
    ver_sent = pd.crosstab(df['version'], df['sentiment'])
    fig, ax = plt.subplots(figsize=(9, 4))
    ver_sent.plot(kind='bar', ax=ax,
                  color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='black', rot=0)
    ax.set_title("ChatGPT Version vs Sentiment", fontweight='bold')
    ax.set_xlabel("Version"); ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Q7: Location vs Sentiment
    st.markdown("---")
    st.markdown("### ❓ Q7: Which locations show the most positive or negative sentiment?")

    loc_sent = df.groupby(['location', 'sentiment']).size().unstack(fill_value=0)
    loc_sent['Total'] = loc_sent.sum(axis=1)
    loc_sent = loc_sent.sort_values('Total', ascending=False).head(10).drop(columns='Total')

    fig, ax = plt.subplots(figsize=(12, 5))
    loc_sent.plot(kind='bar', ax=ax,
                  color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black', rot=30)
    ax.set_title("Top 10 Locations — Sentiment Breakdown", fontweight='bold', fontsize=13)
    ax.set_xlabel("Location")
    ax.set_ylabel("Review Count")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**📌 Insight:** Helps identify region-based user experience issues or appreciation.")

    # Q8: Platform vs Sentiment (percentage)
    st.markdown("---")
    st.markdown("### ❓ Q8: Is there a difference in sentiment across platforms (Web vs Mobile)?")

    plat_sent_pct = pd.crosstab(df['platform'], df['sentiment'], normalize='index').round(3) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    plat_sent_pct.plot(kind='bar', ax=ax,
                       color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black', rot=15)
    ax.set_title("Platform vs Sentiment (% breakdown)", fontweight='bold', fontsize=13)
    ax.set_xlabel("Platform")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Sentiment")
    ax.set_ylim(0, 100)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**📌 Insight:** Identifies where user experience needs improvement — Web or Mobile.")

    # Also show table
    st.markdown("**Percentage Table:**")
    st.dataframe(plat_sent_pct.style.format("{:.1f}%"), use_container_width=True)

    # Q9: Version vs Sentiment (detailed)
    st.markdown("---")
    st.markdown("### ❓ Q9: Which ChatGPT versions are associated with higher or lower sentiment?")

    ver_sent_pct = pd.crosstab(df['version'], df['sentiment'], normalize='index').round(3) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ver_sent_pct.plot(kind='bar', ax=ax,
                      color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black', rot=0)
    ax.set_title("ChatGPT Version vs Sentiment (%)", fontweight='bold', fontsize=13)
    ax.set_xlabel("Version")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Sentiment")
    ax.set_ylim(0, 100)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Best version
    best_ver = ver_sent_pct['Positive'].idxmax() if 'Positive' in ver_sent_pct.columns else "N/A"
    worst_ver = ver_sent_pct['Negative'].idxmax() if 'Negative' in ver_sent_pct.columns else "N/A"

    col_v1, col_v2 = st.columns(2)
    col_v1.success(f"🏆 Highest Positive Sentiment Version: **{best_ver}**")
    col_v2.error(f"⚠️ Highest Negative Sentiment Version: **{worst_ver}**")

    st.markdown("**📌 Insight:** Determines if a version release positively or negatively impacted users.")

    # Q10: Most Common Negative Feedback Themes
    st.markdown("---")
    st.markdown("### ❓ Q10: What are the most common negative feedback themes?")

    neg_reviews = df[df['sentiment'] == 'Negative']['cleaned_review']

    # Top 20 words in negative reviews
    all_neg_words = " ".join(neg_reviews).split()
    neg_word_freq = Counter(all_neg_words).most_common(20)
    words, freqs = zip(*neg_word_freq)

    col_n1, col_n2 = st.columns(2)

    with col_n1:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(list(words)[::-1], list(freqs)[::-1],
                color='#e74c3c', edgecolor='black')
        ax.set_title("Top 20 Words in Negative Reviews", fontweight='bold', fontsize=12)
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_n2:
        st.markdown("#### 🔍 Negative Review Themes")
        st.error("""
        Common pain points found in negative reviews:
        - 🐛 **Bugs & Issues** — app crashes, errors
        - 🐌 **Performance** — slow, laggy responses
        - 😤 **User Experience** — not user-friendly
        - ❌ **Unmet Expectations** — does not work as promised
        - 🔄 **Improvements Needed** — missing features
        """)

        st.markdown("#### 📊 Negative Review Stats")
        neg_df = df[df['sentiment'] == 'Negative']
        st.metric("Total Negative Reviews", len(neg_df))
        st.metric("Avg Rating (Negative)", round(neg_df['rating'].mean(), 2))
        st.metric("Avg Review Length", round(neg_df['review_length'].mean(), 2))

    st.markdown("**📌 Insight:** Use topic grouping to identify recurring pain points and fix them.")




# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>🤖 <b>AI Echo</b> — Built with Python & Streamlit | "
    "Data Science Project 2026</center>",
    unsafe_allow_html=True
)
