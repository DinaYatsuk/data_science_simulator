from typing import List, Tuple
import json
import os


class BaseEval:
    def __init__(self, documents_path: str):
        """A class to evaluate differences between text and OCR text from documents."""
        self.documents = self.load_documents(documents_path)

        if not self.documents:
            raise ValueError("The documents list is empty.")

    def load_documents(self, path: str) -> List[Tuple[str, str]]:
        """
        Loads documents from a JSON file.

        Args:
            path (str): The file path to the JSON file.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing text and OCR text from the documents.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file does not contain a valid list of documents or is not a valid JSON file.
        """
        if not os.path.exists(path=path):
            raise FileNotFoundError("The file does not exist.")

        try:
            with open(path, "r", encoding="utf-8") as f:
                documents = json.load(f)
                if not isinstance(documents, list):
                    raise ValueError("The documents list is empty.")
                documents_list = [(document["text"], document["ocr_text"]) for document in documents]
            return documents_list

        except json.JSONDecodeError:
            raise ValueError("The file is not a valid JSON file.")

    def validate_document(self, doc: str) -> str:
        """
        Validates that a document contains the required fields.

        Args:
            doc (dict): The document to validate.

        Returns:
            Tuple[str, str]: A tuple containing the original text and OCR text from the document,
                in the order (text, ocr_text).

        Raises:
            ValueError: If the document is not a dictionary.
            ValueError: If the document does not contain the 'text' field.
            ValueError: If the document does not contain the 'ocr_text' field.
        """
        parsed_document = json.loads(doc)
        if not isinstance(parsed_document, dict):
            raise ValueError("The document is not a dictionary.")

        if 'text' not in parsed_document:
            raise ValueError("The document does not contain the 'text' field.")
        if 'ocr_text' not in parsed_document:
            raise ValueError("The document does not contain the 'ocr_text' field.")

        return (parsed_document['text'], parsed_document['ocr_text'])

    def _eval_func(self, text: str, ocr_text: str) -> float:
        """
        Evaluates the difference between text and OCR text.

        Args:
            text (str): The original text.
            ocr_text (str): The OCR extracted text.

        Returns:
            float: The quantity of differences between the text and OCR text.
        """

        matches = 0
        for (t, o) in zip(text, ocr_text):
            if t == o:
                matches += 1

        score = len(text) - matches
        return score

    def evaluate(self, limit: int = None) -> List[Tuple[float, str, str]]:
        """
        Evaluates all documents and returns the differences.

        This function iterates over the documents loaded in `self.documents` and
        calculates the difference quantity using the `_eval_func` method.
        The results are sorted by the difference quantity in descending order.

        Args:
            limit (int, optional): The maximum number of results to return.
                If None, all results are returned. Defaults to None.

        Returns:
            List[Tuple[float, str, str]]:
                A list of tuples, each containing:
                - float: The difference quantity.
                - str: The original text.
                - str: The OCR text.
                The list is sorted by the difference quantity in descending order.
        """
        results = []
        for document in self.documents:
            score = self._eval_func(document[0], document[1])
            results.append((score, document[0], document[1]))
        return sorted(results, key=lambda x: x[0], reverse=True)[:limit]

class JaccardEval(BaseEval):
    def _eval_func(self, text: str, ocr_text: str) -> float:
        """
        Evaluates the difference between text and OCR text using Jaccard similarity.

        Args:
            text (str): The original text.
            ocr_text (str): The OCR extracted text.

        Returns:
            float: The quantity of differences between the text and OCR text.
        """
        text_set = set(text.split(' '))
        ocr_set = set(ocr_text.split(' '))

        intersection = text_set.intersection(ocr_set)
        union = text_set.union(ocr_set)

        if len(union) == 0:
            return 0

        return 1 - (len(intersection) / (len(union)))

class LevenshteinEval(BaseEval):
    def _eval_func(self, text: str, ocr_text: str) -> float:
        """Evaluates the difference between text and OCR text using Levenshtein similarity.

        Args:
            text (str): The original text.
            ocr_text (str): The OCR extracted text.

        Returns:
            float: The quantity of differences between the text and OCR text."""

        n = len(text)
        m = len(ocr_text)
        matrix = [[0 for i in range(m + 1)] for j in range(n + 1)]
        for i in range(n + 1):
            for j in range(m + 1):
                matrix[i][j] = self._levanstein(i, j, text, ocr_text, matrix)
        return matrix[n][m]

    def _levanstein(self, i: int, j: int, s1: str, s2: str, matrix) -> float:
        """levanstein function for OCR EVAL"""
        if i == 0 and j == 0:
            return 0
        elif j == 0 and i > 0:
            return i
        elif i == 0 and j > 0:
            return j
        else:
            m = 0 if s1[i - 1] == s2[j - 1] else 1
            return min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + m)

class NormLevEval(BaseEval):
    def __init__(
        self,
        documents_path: str,
        insert_cost: float = 1,
        delete_cost: float = 1,
        substitute_cost: float = 1,
    ):
        """
        Initializes the NormLevEval class with the documents path and cost parameters.

        Args:
            documents_path (str): The file path to the JSON file containing the documents.
            insert_cost (float): The cost parameter for insertions.
            delete_cost (float): The cost parameter for deletions.
            substitute_cost (float): The cost parameter for substitutions.

        Raises:
            ValueError: If any cost parameter is not in the range [0, 2].
        """
        super().__init__(documents_path)

        if not (0 <= insert_cost <= 2) and (0 <= delete_cost <= 2) and (0 <= substitute_cost <= 2):
            raise ValueError("The cost parameter is not in the range [0, 2].")

        self.insert_cost = insert_cost
        self.delete_cost = delete_cost
        self.substitute_cost = substitute_cost

    def _eval_func(self, text: str, ocr_text: str) -> float:
        """
        Evaluates the normalized Levenshtein distance between two strings.

        The normalized Levenshtein distance is computed as the ratio of the
        Levenshtein distance to the maximum possible distance, which is the
        length of the longer string multiplied by the highest cost among the
        insertion, deletion, and substitution operations. This normalization
        scales the distance to a range of 0 to 1, where 0 indicates identical
        strings and 1 indicates maximum dissimilarity.

        Args:
            text (str): The reference text.
            ocr_text (str): The OCR output text.

        Returns:
            float: The normalized Levenshtein distance.
        """

        n = len(text)
        m = len(ocr_text)
        matrix = [[0 for i in range(m + 1)] for j in range(n + 1)]
        for i in range(n + 1):
            for j in range(m + 1):
                matrix[i][j] = self._levanstein(i, j, text, ocr_text, matrix)


        max_cost = max(self.delete_cost, self.insert_cost, self.substitute_cost)
        max_distance = max(n, m) * max_cost
        distance = matrix[n][m]
        return distance / max_distance if max_distance > 0 else 0


    def _levanstein(self, i: int, j: int, s1: str, s2: str, matrix) -> float:
        """levanstein function for OCR EVAL"""
        if i == 0 and j == 0:
            return 0
        elif j == 0 and i > 0:
            return i
        elif i == 0 and j > 0:
            return j
        else:
            cost = 0 if s1[i - 1] == s2[j - 1] else self.substitute_cost
            return min(matrix[i - 1][j] + self.delete_cost, matrix[i][j - 1] + self.insert_cost, matrix[i - 1][j - 1] + cost)


if __name__ == "__main__":
    eval = NormLevEval("/Users/dinayatsuk/PycharmProjects/OCR EVAL/data.json",  substitute_cost=1.2)
    result = eval.evaluate()

    for score, text, ocr_text in result:
        print(f"score: {score:.2f}")
        print(f"source text: {text}")
        print(f"parsed text: {ocr_text}")
        print()

