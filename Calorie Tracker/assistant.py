from typing import Dict, Any
import time
import os
import json

import replicate
from dotenv import load_dotenv


class LLMAssistant:
    """
    A class to interact with a large language model (LLM) using the Replicate API.
    """

    def __init__(
        self, system_prompt: str, model_id: str, temperature=0.01, max_tokens=1024
    ):
        """
        Initializes the LLMAssistant instance with a system prompt, model ID, temperature, and max tokens.

        Raises:
            ValueError: If the Replicate API token is not set in the environment variables.
        """
        load_dotenv(dotenv_path=".venv/token.env")

        self.token = os.getenv("REPLICATE_API_TOKEN")

        if not self.token:
            raise ValueError(
                "Environment variable REPLICATE_API_TOKEN is missing or empty."
            )

        self.system_prompt = system_prompt
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize replicate client
        self.client = replicate.Client(api_token=self.token)

    def _generate_input(self, image_Base64: str) -> Dict[str, Any]:
        """
        Generates the input payload for the model using the Base64-encoded image.
        """
        input = {
            "top_p": 1,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "image": f"data:image/jpeg;base64,{image_Base64}",
            "prompt": self.system_prompt,
        }

        return input

    def _parse_response(self, output: str) -> Dict[str, Any]:
        """
        Parses the model output into a JSON object.

        Args:
            output (str): The raw output from the model as a string.

        Returns:
            Dict[str, Any]: A dictionary containing the status of the parsing (success/error),
            the parsed result, and any error message if applicable.
        """
        try:
            response = json.loads(output)
            return {"status": "success", "result": response, 'error': None}
        except json.JSONDecodeError as e:
            return {"status": "error", "result": {}, "error": str(e)}


    def generate_response(self, image_Base64: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Generates a response from the LLM model using a Base64-encoded image.

        This method sends a request to the model with a Base64-encoded image and system prompt,
        waits for a response, and parses the result. It handles prediction timeouts and response
        parsing, returning the structured output or error details.

        Args:
            image_base64 (str): The Base64-encoded image to be used for generating a response.
            timeout (int, optional): The maximum amount of time (in seconds) to wait for the model's response.
            Defaults to 10 seconds.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'status' (str): 'success' if the process completed successfully, or 'error' if any issues occurred.
                - 'result' (Dict): The parsed model output if successful, or an empty dictionary if an error occurred.
                - 'error' (str): The error message, if any.

        Raises:
            ValueError: If the timeout is less than 1 second.
            TimeoutError: If the prediction process exceeds the specified timeout.

        Internal Methods:
            - `_generate_input`: Prepares the input payload with the Base64 image, system prompt, temperature,
                                 and max tokens.
            - `_parse_response`: Converts the raw model output into a structured dictionary, or handles parsing errors.
        """
        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        input_load = self._generate_input(image_Base64)

        try:
            prediction = self.client.predictions.create(
                version=self.model_id,
                input=input_load)
            start_time = time.time()

            while prediction.status not in ["succeeded", "failed"]:
                time.sleep(1)
                prediction = self.client.predictions.get(prediction.id)

                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    raise TimeoutError(
                        f"Prediction exceeded the timeout of {timeout} seconds."
                    )

            if prediction.status == "succeeded":
                raw_output = prediction.output
                parsed_output = self._parse_response("".join([str(prediction) for prediction in raw_output]))
                return parsed_output

        except Exception as e:
            return {
                "status": "error",
                "result": {},
                "error": f"An error occurred during prediction: {str(e)}",
            }
