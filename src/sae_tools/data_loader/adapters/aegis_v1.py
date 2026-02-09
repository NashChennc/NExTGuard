from .base import BaseAdapter, make_category
from typing import Dict, Any, Optional

"""
original fields (Aegis 1.0, parquet file):
    [text] (str): text content
    [text_type] (str): 类型（如"llm_response", "combined", "user_message"）
    [labels_0]到[labels_4] (str/None): multiple annotator labels
        - may be "Safe" or category string
        - category string may contain multiple categories separated by commas (e.g. "Criminal Planning/Confessions, Violence")
    [num_annotations] (int): number of annotations

data analysis:
    - labels may be "Safe" or category string
    - category string may contain multiple categories separated by commas
    - need to select the main label from multiple annotators (using the first non-null label)

processing logic:
    - select the first non-null label from labels_0 to labels_4
    - if the label is "Safe", then prompt_label="Safe"
    - if the label is category string, then prompt_label="Unsafe", and parse categories (may contain commas)
    - text_type is used to distinguish prompt and response (need to determine based on text_type)
"""


class AegisAdapter(BaseAdapter):
    SPLIT = "test"

    def _get_primary_label(self, example: Dict[str, Any]) -> Optional[str]:
        """select the first non-null label from multiple annotators"""
        for i in range(5):
            label = example.get(f"labels_{i}")
            if label is not None and label != "":
                return label
        return None

    def _parse_categories(self, label: str) -> list:
        """parse category string, may contain multiple categories separated by commas"""
        if not label or label == "Safe":
            return []
        # split by commas and clean whitespace
        categories = [c.strip() for c in label.split(",") if c.strip()]
        return categories

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # get the main label (the first non-null label)
        primary_label = self._get_primary_label(example)
        text = example.get("text", "")
        
        # parse labels
        if primary_label == "Safe":
            prompt_label = "Safe"
            harm_labels = []
        else:
            # parse categories (may contain multiple categories separated by commas)
            harm_labels = self._parse_categories(primary_label) if primary_label else []
            prompt_label = "Unsafe" if harm_labels or (primary_label and primary_label != "Safe") else "Safe"
        
        category = make_category(harm_labels=harm_labels if harm_labels else None)
        
        # Aegis 1.0 is mainly used to evaluate the safety of prompt, and is treated as prompt
        return {
            "prompt": text,
            "response": "",
            "prompt_label": prompt_label,
            "response_label": None,
            "category": category,
            "source": "Aegis-1.0"
        }
