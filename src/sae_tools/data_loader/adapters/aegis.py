from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (nvidia/Aegis-AI-Content-Safety-Dataset-2.0, split=test):
    [id] (str): unique identifier
    [reconstruction_id_if_redacted] (NoneType): reconstruction ID (if redacted)
    [prompt] (str): user prompt
    [response] (str): model response, may be None
    [prompt_label] (str): prompt label (safe/unsafe)
    [response_label] (str): response label (safe/unsafe), may be None
    [violated_categories] (str): violated categories
        - data type: comma-separated string, supports multiple labels
        - 23 values:
            - 'Controlled/Regulated Substances'
            - 'Copyright/Trademark/Plagiarism'
            - 'Criminal Planning/Confessions'
            - 'Fraud/Deception'
            - 'Guns and Illegal Weapons'
            - 'Harassment'
            - 'Hate/Identity Hate'
            - 'High Risk Gov Decision Making'
            - 'Illegal Activity'
            - 'Immoral/Unethical'
            - 'Malware'
            - 'Manipulation'
            - 'Needs Caution'
            - 'Other'
            - 'PII/Privacy'
            - 'Political/Misinformation/Conspiracy'
            - 'Profanity'
            - 'Sexual'
            - 'Sexual (minor)'
            - 'Suicide and Self Harm'
            - 'Threat'
            - 'Unauthorized Advice'
            - 'Violence'
    [prompt_label_source] (str): prompt label source (human/llm_jury)
    [response_label_source] (str): response label source (human/llm_jury)

data analysis (split=test, 1964 samples):
    - prompt_label distribution: unsafe=1059, safe=905
    - response_label distribution: unsafe=394, safe=458, None=1112
    - response=None samples: 1112 (56.7%), corresponding to response_label=None
    - prompt_label != response_label: 1212 (normal phenomenon, both are independently labeled)
    - violated_categories None: 719, Unsafe samples have categories
"""


class Aegis2Adapter(BaseAdapter):
    SPLIT = "test"

    def _normalize_label(self, label: str) -> str:
        """normalize label to Safe/Unsafe/None"""
        if not label:
            return None
        if label.lower().startswith("s"):
            return "Safe"
        elif label.lower().startswith("u"):
            return "Unsafe"
        return label

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt_label = self._normalize_label(example.get("prompt_label", ""))
        response_label = self._normalize_label(example.get("response_label", ""))

        # parse comma-separated category string
        violated_categories = example.get("violated_categories", "")
        harm_labels = []
        if violated_categories:
            harm_labels = [c.strip() for c in violated_categories.split(",") if c.strip()]

        category = make_category(harm_labels=harm_labels if harm_labels else None)

        return {
            "prompt": example.get("prompt", ""),
            "response": example.get("response") or "",  # None -> ""
            "prompt_label": prompt_label,
            "response_label": response_label,
            "category": category,
            "source": "Aegis-2.0"
        }
