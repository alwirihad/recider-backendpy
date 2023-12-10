import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and vectorizer
knn_model = joblib.load("./models/knn_model.joblib")
vectorizer = joblib.load('./models/tfidf_vectorizer.joblib')

csv_path = os.environ.get('CSV_PATH', './data/recido.csv')
df = pd.read_csv(csv_path)


# Endpoint for getting recipe recommendations
@app.route('/')
def index():
    return 'Hello World!'

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json(force=True)
        user_ingredients = data.get('ingredients', '').split(', ')

        if not user_ingredients:
            raise ValueError("Input ingredients tidak boleh kosong.")

        user_input = vectorizer.transform([" ".join(user_ingredients)])

        distances, indices = knn_model.kneighbors(user_input)

        recommendations = get_recommendations(indices)

        output = {'recommendations': recommendations}
        status_code = 200

    except Exception as e:
        output = {'error': str(e)}
        status_code = 400

    return jsonify(output), status_code

def get_recommendations(indices):
    recommended_indices = indices[0]

    recommended_data = df.iloc[recommended_indices]

    recommendations = [{'Title': row['Title'], 'Ingredients': row['Ingredients'], 'Steps': row['Steps']} for _, row in recommended_data.iterrows()]

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
