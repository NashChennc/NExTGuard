from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (HarmBench standard, parquet file):
    [prompt] (str): 用户提示
    [category] (str): category string (e.g. "chemical_biological")

data analysis:
    - only prompt, no response
    - category field represents harm categories
    - samples with category should be marked as Unsafe

processing logic:
    - extract prompt and category fields
    - category as harm_label added to category
    - if category, prompt_label="Unsafe", otherwise "Safe"
    - response="", response_label=None
"""


class HarmBenchAdapter(BaseAdapter):
    SUBSET = "standard"
    SPLIT = "train"  # parquet file default split

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.get("prompt", "")
        category = example.get("category", "")
        
        # build harm_labels
        harm_labels = [category] if category else []
        
        # determine prompt_label based on whether there is category
        prompt_label = "Unsafe" if category else "Safe"
        
        category_json = make_category(harm_labels=harm_labels if harm_labels else None)
        
        return {
            "prompt": prompt,
            "response": "",
            "prompt_label": prompt_label,
            "response_label": None,
            "category": category_json,
            "source": "HarmBench"
        }


class HarmBenchResponseAdapter(BaseAdapter):
    SUBSET = "contextual"
    SPLIT = "train"  # parquet file default split

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.get("prompt", "")
        response = example.get("context", "")
        category = example.get("category", "")
        
        # build harm_labels
        harm_labels = [category] if category else []
        
        # determine response_label based on whether there is category
        response_label = "Unsafe" if category else "Safe"
        
        category_json = make_category(harm_labels=harm_labels if harm_labels else None)
        
        return {
            "prompt": prompt,
            "response": response,
            "prompt_label": None,
            "response_label": response_label,
            "category": category_json,
            "source": "HarmBenchResponse"
        }
