from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (PKU-Alignment/BeaverTails, split=30k_test):
    [prompt] (str): user prompt
    [response] (str): model response
    [category] (dict): dictionary of category, multi-label boolean format
        - data type: dict[str, bool], keys are category names, values are True/False
        - 14 category keys:
            - 'animal_abuse'
            - 'child_abuse'
            - 'controversial_topics,politics'
            - 'discrimination,stereotype,injustice'
            - 'drug_abuse,weapons,banned_substance'
            - 'financial_crime,property_crime,theft'
            - 'hate_speech,offensive_language'
            - 'misinformation_regarding_ethics,laws_and_safety'
            - 'non_violent_unethical_behavior'
            - 'privacy_violation'
            - 'self_harm'
            - 'sexually_explicit,adult_content'
            - 'terrorism,organized_crime'
            - 'violence,aiding_and_abetting,incitement'
    [is_safe] (bool): whether safe

data analysis (split=30k_test, 3021 samples):
    - category all False: 1288 (42.6%), corresponding to is_safe=True
    - is_safe=False but category all False: 0 (data consistency)
"""


class BeaverTailsAdapter(BaseAdapter):
    SPLIT = "30k_test"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # extract True categories from dict[str, bool]
        category_dict = example.get("category", {})
        harm_labels = [k for k, v in category_dict.items() if v]

        category = make_category(harm_labels=harm_labels if harm_labels else None)

        return {
            "prompt": example.get("prompt", ""),
            "response": example.get("response", ""),
            "prompt_label": None,  # dataset has no this field
            "response_label": "Safe" if example.get("is_safe") else "Unsafe",
            "category": category,
            "source": "BeaverTails"
        }


"""
original fields (BeaverTailsAugmented, split=augmented):
    same fields as BeaverTails, but with additional sanitized version:
    [prompt] (str): original user prompt
    [response] (str): original model response
    [category] (dict): category dictionary, same format as BeaverTails
    [is_safe] (bool): whether safe
    [sanitized_prompt] (str): sanitized prompt (adapter does not use this field)
    [sanitized_response] (str): sanitized response (adapter does not use this field)
    [original_id] (int): original ID in the dataset
    [status] (str): processing status

Adapter uses original fields (prompt, response), same as BeaverTailsAdapter.
"""


class BeaverTailsAugmentedAdapter(BaseAdapter):
    SPLIT = "flattened"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # extract True categories from dict[str, bool]
        category_dict = example.get("category", {})
        harm_labels = [k for k, v in category_dict.items() if v]

        category = make_category(harm_labels=harm_labels if harm_labels else None)

        return {
            "prompt": example.get("prompt", ""),
            "response": example.get("response", ""),
            # "sanitized_prompt": example.get("sanitized_prompt", ""),
            # "sanitized_response": example.get("sanitized_response", ""),
            "prompt_label": None,  # dataset has no this field
            "response_label": example.get("label"),
            "category": category,
            "source": "BeaverTailsAugmented"
        }
