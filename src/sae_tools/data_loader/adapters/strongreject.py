from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (walledai/StrongREJECT, split=train):
    [prompt] (str): harmful prompt text
    [category] (str): harm category
        - data type: single string
        - 6 values:
            - 'Disinformation and deception'
            - 'Hate, harassment and discrimination'
            - 'Illegal goods and services'
            - 'Non-violent crimes'
            - 'Sexual content'
            - 'Violence'
    [source] (str): data source (e.g. DAN)

data analysis (split=train, 313 samples):
    - no label field, all assumed to be Unsafe (for testing jailbreak defense)
    - all samples have category value

processing logic:
    - extract prompt and category field
    - category as harm_label added to category
"""


class StrongREJECTAdapter(BaseAdapter):
    SPLIT = "train"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        category_val = example.get("category", "")
        category = make_category(harm_labels=[category_val] if category_val else None)

        return {
            "prompt": example.get("prompt", ""),
            "response": "",
            "prompt_label": "Unsafe",  # 所有提示都是有害的
            "response_label": None,
            "category": category,
            "source": "StrongREJECT"
        }
