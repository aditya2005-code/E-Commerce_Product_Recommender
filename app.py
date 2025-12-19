import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# App Configuration
# ---------------------------
st.set_page_config(page_title="E-Commerce Recommender System", layout="wide")
st.title("🛒 E-Commerce Product Recommendation System")

# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data
def load_data():
    # Update path if required
    df = pd.read_csv("D:\Machine-Learning-Projects\E-Commerce_Product_Recommeder\DataSet\clean_data.csv")
    return df

train = load_data()

# Basic cleaning
train['Tags'] = train['Tags'].fillna('')

# ---------------------------
# Precompute TF-IDF & Cosine Similarity
# ---------------------------
@st.cache_data
def compute_tfidf(tags):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(tags)
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

cosine_similarities = compute_tfidf(train['Tags'])

# ---------------------------
# Recommendation Functions
# ---------------------------

def content_based_recommendation(item_name, top_k=10):
    matches = train.index[train['Name'] == item_name]
    if len(matches) == 0:
        return pd.DataFrame()

    idx = int(matches[0])
    scores = list(enumerate(cosine_similarities[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    return train.iloc[[i[0] for i in scores]]


def rating_based_recommendation(top_k=10):
    return (
        train.groupby('Name', as_index=False)
        .agg({'Rating': 'mean'})
        .sort_values(by='Rating', ascending=False)
        .head(top_k)
    )


def collaborative_filtering_stub():
    # Placeholder for collaborative filtering
    # Can be replaced with matrix factorization / Surprise / implicit models
    return train.sample(10)


def hybrid_recommendation(item_name, top_k=10):
    content_rec = content_based_recommendation(item_name, top_k=top_k*2)
    if content_rec.empty:
        return pd.DataFrame()

    hybrid = (
        content_rec
        .groupby('Name', as_index=False)
        .agg({'Rating': 'mean'})
        .sort_values(by='Rating', ascending=False)
        .head(top_k)
    )
    return hybrid

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Recommendation Settings")
rec_type = st.sidebar.selectbox(
    "Choose Recommendation Type",
    [
        "Content-Based Recommendation",
        "Rating-Based Recommendation",
        "Collaborative Filtering",
        "Hybrid Recommendation"
    ]
)

product_name = None
if rec_type in ["Content-Based Recommendation", "Hybrid Recommendation"]:
    product_name = st.sidebar.selectbox("Select a Product", train['Name'].unique())

# ---------------------------
# Main Recommendation Section
# ---------------------------
st.subheader(rec_type)

if st.button("Get Recommendations"):
    if rec_type == "Content-Based Recommendation":
        result = content_based_recommendation(product_name)
        st.dataframe(result)

    elif rec_type == "Rating-Based Recommendation":
        result = rating_based_recommendation()
        st.dataframe(result)

    elif rec_type == "Collaborative Filtering":
        result = collaborative_filtering_stub()
        st.dataframe(result)

    elif rec_type == "Hybrid Recommendation":
        result = hybrid_recommendation(product_name)
        st.dataframe(result)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built with Streamlit | Content, Rating, Collaborative & Hybrid Recommendations")