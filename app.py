import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np


app = Flask(__name__)

# Load the model and feature names
model_data = joblib.load('recipe_knn_model.joblib')
knn_model = model_data['model']
feature_names = model_data['feature_names']

folder_path = 'data'
recipes_df = pd.read_csv(os.path.join(folder_path, 'recido.csv'))

@app.route('/recommend', methods=['POST'])
def recommend_recipe():
    try:
        if not request.json or 'ingredients' not in request.json:
            return jsonify({"error": "Invalid input. 'ingredients' key is required in the JSON request."}), 400
        user_input_ingredients = [ingredient.lower() for ingredient in request.json.get('ingredients', [])]

        filtered_df = recipes_df

        for ingredient in user_input_ingredients:
            lower_ingredient = ingredient.lower()
            filtered_df = filtered_df[filtered_df['Ingredients'].str.lower().str.contains(lower_ingredient)]

        if filtered_df.empty:
            return jsonify({"message": "Resep makanan tidak ditemukan."}), 404

        X_features = pd.get_dummies(filtered_df[['Title', 'Ingredients', 'Steps']], drop_first=True)

        user_input_df = pd.DataFrame(0, columns=feature_names, index=[0])

        for ingredient in user_input_ingredients:
            if ingredient.lower() in feature_names:
               user_input_df[ingredient.lower()] = 1

        # Make predictions
        distances, indices = knn_model.kneighbors(user_input_df)
        filtered_indices = filtered_df.index.to_numpy()

        indices = indices.flatten()
        indices = indices[indices < len(filtered_df)]

        if len(indices) == 0:
            return jsonify({"message": "Resep makanan tidak ditemukan."}), 404

        recommended_recipes = filtered_df.iloc[indices]

        response_data = {
            "message": "Resep Direkomendasikan",
            "recipes": recommended_recipes.to_dict(orient='records')
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
