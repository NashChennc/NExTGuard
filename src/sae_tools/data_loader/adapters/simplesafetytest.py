from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (SimpleSafetyTests instruct, parquet file):
    [prompt] (str): user prompt
    [harm_type] (str): harm type string (e.g. "Suicide, Self-Harm, and Eating Disorders")

data analysis:
    - only prompt, no response
    - harm_type field represents harm type
    - samples with harm_type should be marked as Unsafe

processing logic:
    - extract prompt and harm_type field
    - harm_type as harm_label added to category
    - if harm_type, prompt_label="Unsafe", otherwise "Safe"
    - response="", response_label=None
"""


class SimpleSafetyTestAdapter(BaseAdapter):
    SPLIT = "instruct"
    SUBSET = None

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.get("prompt", "")
        harm_type = example.get("harm_type", "")
        
        # build harm_labels
        harm_labels = [harm_type] if harm_type else []
        
        # determine prompt_label based on whether there is harm_type
        prompt_label = "Unsafe" if harm_type else "Safe"
        
        category = make_category(harm_labels=harm_labels if harm_labels else None)
        
        return {
            "prompt": prompt,
            "response": "",
            "prompt_label": prompt_label,
            "response_label": None,
            "category": category,
            "source": "SimpleSafetyTest"
        }
