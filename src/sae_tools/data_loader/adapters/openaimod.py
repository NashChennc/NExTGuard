from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (OpenAIMod, jsonl file):
    [prompt] (str): 用户提示
    [S] (int): Sexual label (0/1)
    [H] (int): Hate label (0/1)
    [V] (int): Violence label (0/1)
    [HR] (int): Harassment label (0/1)
    [SH] (int): Self-Harm label (0/1)
    [S3] (int): Sexual (minor) label (0/1)
    [H2] (int): Hate/Threatening label (0/1)
    [V2] (int): Violence/Graphic label (0/1)

标签映射:
    S → "Sexual"
    H → "Hate"
    V → "Violence"
    HR → "Harassment"
    SH → "Self-Harm"
    S3 → "Sexual (minor)"
    H2 → "Hate/Threatening"
    V2 → "Violence/Graphic"

处理逻辑:
    - check 8 binary labels, map the label with value 1 to the corresponding category name
    - build harm_labels list
    - set prompt_label based on whether there is harm_labels (Unsafe if yes, Safe if no)
"""

# label mapping table
LABEL_MAPPING = {
    "S": "Sexual",
    "H": "Hate",
    "V": "Violence",
    "HR": "Harassment",
    "SH": "Self-Harm",
    "S3": "Sexual (minor)",
    "H2": "Hate/Threatening",
    "V2": "Violence/Graphic"
}


class OpenAIModAdapter(BaseAdapter):
    SPLIT = "train"  # jsonl file default split

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.get("prompt", "")
        
        # check 8 binary labels, map the label with value 1 to the corresponding category name
        harm_labels = []
        for label_key, category_name in LABEL_MAPPING.items():
            if example.get(label_key) == 1:
                harm_labels.append(category_name)
        
        # set prompt_label based on whether there is harm_labels (Unsafe if yes, Safe if no)
        prompt_label = "Unsafe" if harm_labels else "Safe"
        
        category = make_category(harm_labels=harm_labels if harm_labels else None)
        
        return {
            "prompt": prompt,
            "response": "",
            "prompt_label": prompt_label,
            "response_label": None,
            "category": category,
            "source": "OpenAIMod"
        }
