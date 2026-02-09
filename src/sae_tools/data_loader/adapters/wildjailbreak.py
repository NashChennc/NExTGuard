from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (allenai/wildjailbreak, subset=eval, split=train):
    [adversarial] (str): adversarial/jailbreak prompt text
    [label] (int): 标签
        - 0: benign (benign) (contrary to intuition!)
        - 1: harmful (harmful/jailbreak)
    [data_type] (str): data type
        - data type: single string
        - eval subset 2 values:
            - 'adversarial_benign': adversarial but benign (label=0)
            - 'adversarial_harmful': adversarial and harmful (label=1)
        - train subset 4 values (reference):
            - 'adversarial_benign': adversarial but benign
            - 'adversarial_harmful': adversarial and harmful
            - 'vanilla_benign': vanilla benign
            - 'vanilla_harmful': vanilla harmful

data analysis (subset=eval, split=train, 2210 samples):
    - data_type distribution: adversarial_benign=210, adversarial_harmful=2000
    - label distribution: 0=210 (corresponding to benign), 1=2000 (corresponding to harmful)
    - **correction**: label=0 represents Safe, label=1 represents Unsafe

note: this dataset is used to evaluate jailbreak detection ability
"""


class WildJailbreakAdapter(BaseAdapter):
    SUBSET = "eval"
    SPLIT = "train"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # correction: label=0 represents benign (Safe), label=1 represents harmful (Unsafe)
        prompt_label = "Safe" if example.get("label") == 0 else "Unsafe"

        data_type = example.get("data_type", "")
        category = make_category(metadata={"data_type": data_type} if data_type else None)

        return {
            "prompt": example.get("adversarial", ""),
            "response": "",
            "prompt_label": prompt_label,
            "response_label": None,
            "category": category,
            "source": "WildJailbreak"
        }
