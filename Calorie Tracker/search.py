from typing import Dict, List, Any
import csv
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer, util


class IngredientNutritionSearch:
    """
    A class used to search for nutrition information of ingredients using both fuzzy and semantic search methods.
    """

    def __init__(self, dataset_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the IngredientNutritionSearch with the dataset path and model name.

        Raises: RuntimeError: If the sentence transformer model fails to load.

        Note: Ingredient names in the dataset and during the search are converted to lowercase for consistent matching.
        """
        self.dataset_path = dataset_path
        self.data = self._load_data()
        self._ingredients = list(self.data.keys())

        try:
            self.encoder = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load `{model_name}`. Error: {e}") from e

        self._embeddings = self.encoder.encode(
            self._ingredients, convert_to_tensor=True
        )

    def _load_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the ingredient nutrition data from the CSV file.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are ingredient names (in lowercase)
                                       and values are their nutrition data.

            Example:
            {
                'apple': {
                    'calories': 52.0,
                    'fats': 0.2,
                    'proteins': 0.3,
                    'carbohydrates': 13.8
                },
                'banana': {
                    'calories': 89.0,
                    'fats': 0.3,
                    'proteins': 1.1,
                    'carbohydrates': 22.8
                }
            }

        Raises:
            FileNotFoundError: If the dataset file is not found.
            KeyError: If the expected columns are missing from the CSV.
            Exception: For any other errors that occur during file reading.
        """
        if self.dataset_path is None:
            raise FileNotFoundError(
                "The dataset path is not provided. Please provide a valid file path."
            )

        required_columns = ['name', 'calories', 'total_fat', 'protein', 'carbohydrate']

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for column in required_columns:
                if column not in reader.fieldnames:
                    raise KeyError(f"The column {column} is not present in the dataset.")
            try:
                dictionary = {row['name'].lower():{
                    'calories': float(row['calories']),
                    'fats': float(row['total_fat'].replace("g", "")),
                    'proteins': float(row['protein'].replace("g", "")),
                    'carbohydrates': float(row['carbohydrate'].replace("g", ""))
                }for row in reader}

                return dictionary

            except Exception as e:
                raise Exception(f"An error occurred while reading the dataset. Error: {e}") from e

    def _fuzzy_search(self, ingredient_name: str, threshold: float) -> str:
        """
        Searches for the closest matching ingredient name using fuzzy matching based on string similarity.

        The fuzzy search compares the input ingredient name (converted to lowercase) with the list of available
        ingredient names (also in lowercase) and returns the best match based on a similarity score. It uses the
        `get_close_matches` method from the `difflib` module.

        Args:
            ingredient_name (str): The name of the ingredient to search for. The input is converted to lowercase.
            threshold (float): The similarity threshold for fuzzy matching.
                               This value determines how similar the input string must be to return a match.

        Returns:
            str: The closest matching ingredient name if found, otherwise None.

            Example:
                Input: "aple" (misspelled)
                Match found: "apple"
                Result: 'apple'
        """
        ingredient_name = ingredient_name.lower()
        result = get_close_matches(ingredient_name, self._ingredients, cutoff=threshold)
        return result[0] if result else None

    def _semantic_search(
            self, ingredient_name: str, threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Searches for the closest matching ingredient name using semantic similarity search.

        The semantic search compares the input ingredient name (converted to lowercase) with the list of
        available ingredient names (also in lowercase) based on semantic meaning rather than string similarity.
        It computes embeddings for both the input and the dataset's ingredient names using a pre-trained sentence
        transformer model, and then calculates cosine similarity between them.

        Args:
            ingredient_name (str): The name of the ingredient to search for. The input is converted to lowercase.
            threshold (float): The similarity threshold for semantic search.
                               This value determines how similar the meaning of the input name must be to return a match.

        Returns:
            str: The closest matching ingredient name if found, otherwise None.

            Example:
                Input: "grn apple" (misspelled or ambiguous)
                Semantic match found: "green apple"
                Result: 'green apple'
        """
        ingredient_name = ingredient_name.lower()
        ingredient_embeddings = self.encoder.encode(ingredient_name, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(ingredient_embeddings,self._embeddings)
        max_similarity, similarity_index = cosine_similarity.max(), cosine_similarity.argmax()
        if max_similarity >= threshold:
            return self._ingredients[similarity_index]
        return None

    def search(
            self,
            img_ingredients: Dict[str, Any],
            search_type: str = "fuzzy",
            threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Searches for the nutritional information of a list of ingredients using the specified search type.

        The method supports two search types:
            - "fuzzy": Uses string similarity to find the closest matching ingredient.
            - "semantic": Uses semantic similarity (via embeddings) to find the closest matching ingredient.

        For each ingredient, the method calculates the nutritional values (calories, fats, proteins, carbohydrates)
        based on the provided weight and adds them to the result. If no match is found, an empty dictionary is returned
        for that ingredient.
        search

        Args:
            img_ingredients (Dict[str, Any]): A dict of ingredients and their weights in the format:
                {'ingredient_name': weight_in_grams}.
            search_type (str): The type of search to use, either "fuzzy" or "semantic" (default is "fuzzy").
            threshold (float): The similarity threshold used for matching (default is 0.6).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing ingredient names, their matched values,
                                  and scaled nutritional information based on the provided weight.

            Example:
                Input: {'aple': 150, 'banana': 120}
                Result:
                [
                    {'apple': {'match': 'apple', 'weight': 150, 'calories': 78.0, 'fats': 0.0, 'proteins': 0.0, 'carbohydrates': 21.0}},
                    {'banana': {'match': 'banana', 'weight': 120, 'calories': 107.0, 'fats': 0.4, 'proteins': 1.3, 'carbohydrates': 27.4}}
                ]

        Raises: ValueError: If `search_type` is neither "fuzzy" nor "semantic".

        Note:
        - The nutritional values are scaled proportionally based on the provided weight
            (e.g., 100 grams of a food gives full nutritional values, 50 grams will return half).
        - Rounded values: All nutritional values are rounded to the nearest whole number (0 decimal places).
        """
        results = []
        for ingredient_name, weight in img_ingredients.items():
            if search_type == "fuzzy":
                match = self._fuzzy_search(ingredient_name, threshold=threshold)
            elif search_type == "semantic":
                match = self._semantic_search(ingredient_name, threshold=threshold)
            else:
                raise ValueError("Search type must be either 'fuzzy' or 'semantic'.")

            if match is not None:
                nutritional_value = self.data.get(match, {})
                scaled_data = {key: round(value * (weight/100), 0) for key, value in nutritional_value.items()}
                results.append({ingredient_name: {'match': match,
                                                  'weight': weight,
                                                  'calories': scaled_data['calories'],
                                                  'fats': scaled_data['fats'],
                                                  'proteins': scaled_data['proteins'],
                                                  'carbohydrates': scaled_data['carbohydrates']}})
            else:
                results.append({ingredient_name: {}})

        return results

if __name__ == "__main__":
    engine = IngredientNutritionSearch("nutrition.csv")

    test_ingredients = {
        "tomatoes": 144,
        "cheese": 125,
        "beef": 133,
        "sausages": 195,
        "pepper": 171,
        "onions": 178,
        "mushrooms": 134,
        "garlic": 133,
        "basil": 191,
        "olives": 166,
    }

    print("Fuzzy search results:")
    fuzzy_results = engine.search(test_ingredients, search_type="fuzzy")
    print(fuzzy_results[:1])

    for r in fuzzy_results:
        for key, values in r.items():
            print(f"search:{key} // dataset: {values.get('match', 'None')}")

    print("Semantic search results:")
    semantic_results = engine.search(test_ingredients, search_type="semantic")
    print(semantic_results[:1])

    for r in semantic_results:
        for key, values in r.items():
            print(f'search:{key} // dataset: {values["match"]}')
