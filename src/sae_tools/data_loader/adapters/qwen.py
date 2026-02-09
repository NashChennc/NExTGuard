from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields (Qwen/Qwen3GuardTest, split=thinking):
    [unique_id] (int): 唯一ID
original fields (Qwen/Qwen3GuardTest, split=thinking):
    [unique_id] (int): unique ID
    [message] (list): message list, each item contains role and content
        - role: 'user' or 'assistant'
        - content: message content
    [unsafe_type] (str): unsafe type, 9 values:
        - '' (empty string, represents safe)
        - 'Copyright Violation'
        - 'Non-violent Illegal Acts'
        - 'PII, Personally Identifiable Information'
        - 'Politically Sensitive Topics'
        - 'Sexual Content or Sexual Acts'
        - 'Suicide & Self-Harm'
        - 'Unethical Acts'
        - 'Violent'
    [source] (str): data source model
    [label] (str): label (Safe/Unsafe)
    [input_ids] (list): token ID list
    [unsafe_end_index] (int): unsafe content end index
    [unsafe_start_index] (int): unsafe content start index

data analysis:
    - only prompt, no response
    - unsafe_type as harm_label added to category
    - if unsafe_type, prompt_label="Unsafe", otherwise "Safe"
    - response="", response_label=None

processing logic:
    - extract prompt and unsafe_type fields
    - unsafe_type as harm_label added to category
    - if unsafe_type, prompt_label="Unsafe", otherwise "Safe"
    - response="", response_label=None
"""


class Qwen3GuardTestAdapter(BaseAdapter):
    SPLIT = "thinking"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ""
        response = ""

        for m in example["message"]:
            if m.get("role") == "user" and m.get("content"):
                prompt = m.get("content")
            if m.get("role") == "assistant" and m.get("content"):
                response = m.get("content")

        unsafe_type = example.get("unsafe_type", "")
        category = make_category(harm_labels=[unsafe_type] if unsafe_type else None)

        return {
            "prompt": prompt,
            "response": response,
            "prompt_label": None,  # dataset has no this field
            "response_label": example.get("label"),  # Safe/Unsafe
            "category": category,
            "source": "Qwen3GuardTest"
        }
