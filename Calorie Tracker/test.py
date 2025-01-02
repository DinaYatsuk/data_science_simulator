from assistant import LLMAssistant
from utils import image_url_to_base64

model_id = "80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb"

system_prompt = "Identify the food shown in the photo and write its nutritional value. In response, return a JSON object with the following fields: calories, proteins, fats, carbohydrates.If the image contains no food, return an empty JSON object.Example of expected response for image with no food: {}. Just return the JSON object, don't write anything else."

assistant = LLMAssistant(system_prompt, model_id, temperature=0.01)

food_image_url = "https://images.unsplash.com/photo-1568901346375-23c9450c58cd"
food_image_Base64 = image_url_to_base64(food_image_url)

non_food_image_url = "https://images.unsplash.com/photo-1508356889337-11a080a10d06"
non_food_image_Base64 = image_url_to_base64(non_food_image_url)

food_response = assistant.generate_response(food_image_Base64, timeout=20)
print(f"Response for food image: {food_response}")

non_food_response = assistant.generate_response(non_food_image_Base64, timeout=20)
print(f"Response for non-food image: {non_food_response}")

### Output:
# Response for food image: {'status': 'success', 'result': {'calories': 500, 'proteins': 30, 'fats': 40, 'carbohydrates': 50}, 'error': ''}
# Response for non-food image: {'status': 'success', 'result': {}, 'error': ''}
