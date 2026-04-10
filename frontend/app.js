const API_URL = "http://127.0.0.1:8001";

const movieGrid = document.getElementById('movieGrid');
const refreshBtn = document.getElementById('refreshBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMsg = document.getElementById('errorMsg');

const similarGrid = document.getElementById('similarGrid');
const similarBtn = document.getElementById('similarBtn');
const loadingSpinnerSimilar = document.getElementById('loadingSpinnerSimilar');
const errorMsgSimilar = document.getElementById('errorMsgSimilar');

document.addEventListener('DOMContentLoaded', () => {
    fetchRecommendations();
});

refreshBtn.addEventListener('click', fetchRecommendations);
similarBtn.addEventListener('click', fetchSimilar);

async function fetchRecommendations() {
    const userId = document.getElementById('userId').value || 15;
    const num_recs = document.getElementById('recommendCount').value || 10;
    const method = document.getElementById('algoChoice').value;

    movieGrid.innerHTML = '';
    errorMsg.innerHTML = '';
    loadingSpinner.classList.add('active');

    try {
        const response = await fetch(`${API_URL}/recommend/${userId}?num_recs=${num_recs}&method=${method}`);
        const data = await response.json();
        
        loadingSpinner.classList.remove('active');

        if (!response.ok) {
            errorMsg.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${data.detail || 'Failed to fetch'}`;
            return;
        }

        const recs = data.recommendations || [];
        if (recs.length === 0) {
            errorMsg.innerHTML = "No recommendations found for this user.";
            return;
        }

        recs.forEach((movie, idx) => {
            const delay = idx * 0.05;
            const card = createMovieCard(movie, delay);
            movieGrid.appendChild(card);
        });

    } catch (err) {
        loadingSpinner.classList.remove('active');
        errorMsg.innerHTML = `<i class="fas fa-plug"></i> Failed to connect to backend server.`;
    }
}

async function fetchSimilar() {
    const movieId = document.getElementById('similarMovieId').value || 1;
    
    similarGrid.innerHTML = '';
    errorMsgSimilar.innerHTML = '';
    loadingSpinnerSimilar.classList.add('active');

    try {
        const response = await fetch(`${API_URL}/similar/${movieId}?num_recs=5`);
        const data = await response.json();
        
        loadingSpinnerSimilar.classList.remove('active');

        if (!response.ok) {
            errorMsgSimilar.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${data.detail || 'Failed to fetch'}`;
            return;
        }

        const recs = data.similar_movies || [];
        if (recs.length === 0) {
            errorMsgSimilar.innerHTML = "No similar movies found for this ID.";
            return;
        }

        recs.forEach((movie, idx) => {
            const delay = idx * 0.05;
            const card = createSimilarCard(movie, delay);
            similarGrid.appendChild(card);
        });

    } catch (err) {
        loadingSpinnerSimilar.classList.remove('active');
        errorMsgSimilar.innerHTML = `<i class="fas fa-plug"></i> Connection Error.`;
    }
}

function createMovieCard(movie, delay) {
    const div = document.createElement('div');
    div.className = 'movie-card';
    div.style.animationDelay = `${delay}s`;

    const genresHtml = movie.genres.split('|').map(g => `<span class="genre-badge">${g}</span>`).join('');
    let ratingHtml = '';
    if (movie.predicted_rating) {
         ratingHtml = `<div class="rating"><i class="fas fa-star"></i> ${movie.predicted_rating.toFixed(2)} Match</div>`;
    }
    
    let reasonHtml = '';
    if (movie.reason) {
        reasonHtml = `<div class="reason"><i class="fas fa-lightbulb"></i> ${movie.reason}</div>`;
    }

    div.innerHTML = `
        <div>
            <h3>${movie.title}</h3>
            <div class="genres">${genresHtml}</div>
            ${ratingHtml}
        </div>
        ${reasonHtml}
    `;
    return div;
}

function createSimilarCard(movie, delay) {
    const div = document.createElement('div');
    div.className = 'movie-card';
    div.style.animationDelay = `${delay}s`;

    const genresHtml = movie.genres.split('|').map(g => `<span class="genre-badge">${g}</span>`).join('');
    const similarityHtml = `<div class="rating" style="color:var(--accent);"><i class="fas fa-link"></i> ${(movie.similarity_score * 100).toFixed(1)}% Match</div>`;
    
    div.innerHTML = `
        <div>
            <h3>${movie.title}</h3>
            <div class="genres">${genresHtml}</div>
            ${similarityHtml}
        </div>
    `;
    return div;
}
