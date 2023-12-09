from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the pre-trained model and vectorizer
knn_model = joblib.load("knn_model.joblib")
vectorizer = joblib.load('tfidf_vectorizer.joblib')

df = pd.read_csv(r'D:\recido.csv')


# Endpoint for getting recipe recommendations
@app.route('/')
def index():
    return 'Hello World!'

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json(force=True)
        user_ingredients = data.get('ingredients', '').split(', ')

        # Handle input yang kosong
        if not user_ingredients:
            raise ValueError("Input ingredients tidak boleh kosong.")

        # Transform input pengguna ke dalam format yang sama dengan dataset
        user_input = vectorizer.transform([" ".join(user_ingredients)])

        # Prediksi resep terdekat berdasarkan input pengguna
        distances, indices = knn_model.kneighbors(user_input)

        # Dapatkan rekomendasi berdasarkan indeks yang diberikan oleh model
        recommendations = get_recommendations(indices)

        output = {'recommendations': recommendations}
        status_code = 200

    except Exception as e:
        output = {'error': str(e)}
        status_code = 400

    return jsonify(output), status_code

def get_recommendations(indices):
    # Ambil indeks rekomendasi dari model
    recommended_indices = indices[0]

    # Ambil data rekomendasi dari DataFrame df
    recommended_data = df.iloc[recommended_indices]

    # Ubah struktur data sesuai kebutuhan
    recommendations = [{'Title': row['Title'], 'Ingredients': row['Ingredients'], 'Steps': row['Steps']} for _, row in recommended_data.iterrows()]

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
