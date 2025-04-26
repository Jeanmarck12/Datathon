import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Data and Model
# -------------------------

@st.cache_resource
def load_data():
    df = pd.read_csv("/Users/jeanmarckceant/Desktop/Datathon/Datathon2025_RAW_/places_with_embeddings.csv")
    embeddings = df[[col for col in df.columns if col.startswith('embed_')]].values
    return df, embeddings

@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

places_full_with_embeddings, embeddings = load_data()
sent_trans_model = load_model()

# -------------------------
# Vibe Search Function
# -------------------------

def vibe_search(query, top_k=5):
    query_embedding = sent_trans_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    results = places_full_with_embeddings.iloc[top_k_indices]
    return results[['name', 'neighborhood', 'short_description', 'emoji']]

# -------------------------
# Streamlit UI
# -------------------------

st.title("Vibe Search")

st.write("Find the perfect place based on the vibe you're feeling — not just keywords.")

query = st.text_input("What are you looking for? (e.g., 'best rooftops for sunset', 'where to find hot guys', 'cozy cafes to study')")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please type something to search!")
    else:
        results = vibe_search(query, top_k=5)
        st.subheader("Top Places Matching Your Vibe:")
        for idx, row in results.iterrows():
            st.write(f"### {row['emoji']} {row['name']} ({row['neighborhood']})")
            st.write(f"_{row['short_description']}_")
            st.markdown("---")