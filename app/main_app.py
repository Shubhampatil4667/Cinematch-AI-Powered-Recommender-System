import streamlit as st
import requests
import json

# Connection to our FastAPI backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cinematch AI", page_icon="🍿", layout="wide")

# Custom CSS for Netflix-like dark theme experience
st.markdown("""
    <style>
    .stApp {
        background-color: #141414;
        color: #e5e5e5;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #e50914; /* Netflix Red */
        margin-bottom: 0px;
    }
    .sub-header {
        color: #808080;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #1f1f1f;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        margin-bottom: 20px;
        height: 100%;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.02);
    }
    .genre-tag {
        background-color: #333;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #ddd;
        margin-right: 5px;
    }
    .reason-text {
        color: #46d369; /* Netflix Match Green */
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Cinematch AI 🍿</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Personalized streaming recommendations built with Matrix Factorization & Deep CF.</div>', unsafe_allow_html=True)

st.sidebar.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=2025&auto=format&fit=crop", use_container_width=True)
st.sidebar.markdown("### Profile Settings")

# Dataset (MovieLens latest-small) has users mapped 1-610
user_id = st.sidebar.number_input("User ID (1 to 610)", min_value=1, max_value=610, value=15)
num_recs = st.sidebar.slider("Recommended Count", min_value=3, max_value=24, value=9)
algorithm = st.sidebar.radio("Underlying Algorithm", ["Hybrid AI Approach", "Pure Matrix Factorization (SVD)"])

method_param = "hybrid" if "Hybrid" in algorithm else "svd"

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Hybrid mode provides 'Explainable AI' context like Netflix's \"Because you liked...\"")

# Fetch logic
st.subheader("For You")

if st.sidebar.button("Refresh Recommendations", type="primary"):
    with st.spinner("Connecting to Recommendation Engine..."):
        try:
            res = requests.get(f"{API_URL}/recommend/{user_id}", params={"num_recs": num_recs, "method": method_param})
            
            if res.status_code == 200:
                data = res.json()
                recommendations = data.get("recommendations", [])
                
                if recommendations:
                    # Dynamically generate columns for UI layout
                    cols = st.columns(3)
                    for index, movie in enumerate(recommendations):
                        col = cols[index % 3]
                        with col:
                            # Parse tags
                            genres = movie['genres'].split('|')
                            genre_html = "".join([f'<span class="genre-tag">{g}</span>' for g in genres])
                            
                            # Construct beautiful card
                            card_html = f"""
                            <div class="movie-card">
                                <h3>{movie['title']}</h3>
                                <div style="margin-bottom:10px;">{genre_html}</div>
                            """
                            
                            if movie.get('predicted_rating') is not None:
                                card_html += f'<div style="color: #ffb400;">★ {movie["predicted_rating"]}/5 Predicted Match</div>'
                                
                            if movie.get('reason'):
                                card_html += f'<div class="reason-text">{movie["reason"]}</div>'
                                
                            card_html += "</div>"
                            st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations available for this profile.")
            elif res.status_code == 404:
                st.error("User ID doesn't exist in our latent space. Try a different user from the dataset.")
            else:
                st.error(f"Backend Server Error: {res.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Could not reach the API Backend. Did you start the FastAPI server?")

# Secondary Feature: Item-Item Search
st.markdown("---")
st.subheader("Explore Connected Universe (Item-to-Item Similarity)")

col_a, col_b = st.columns([1, 2])
with col_a:
    search_movie_id = st.number_input("Seed Movie ID (e.g., 1 for Toy Story)", min_value=1, value=1)
    if st.button("Find Similar"):
        st.session_state.search_clicked = True

with col_b:
    if getattr(st.session_state, 'search_clicked', False):
        try:
            res = requests.get(f"{API_URL}/similar/{search_movie_id}", params={"num_recs": 5})
            if res.status_code == 200:
                similars = res.json().get('similar_movies', [])
                for sim in similars:
                    st.markdown(f"- **{sim['title']}** [{round(sim['similarity_score']*100)}% Match]")
            else:
                st.error("Movie not found in the latent database.")
        except:
             st.error("API Connection dropped.")
