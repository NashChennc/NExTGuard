from .base import BaseAdapter, make_category
from typing import Dict, Any

"""
original fields - XSTest (walledai/XSTest, split=test):
    [prompt] (str): user prompt
    [focus] (str): focus keywords (e.g. kill, steal, hack, etc.)
    [type] (str): type, 18 values:
        base types (10):
        - 'homonyms': homonyms
        - 'figurative_language': figurative language
        - 'safe_contexts': safe contexts
        - 'safe_targets': safe targets
        - 'definitions': definitions
        - 'historical_events': historical events
        - 'privacy_public': public figure privacy
        - 'privacy_fictional': fictional character privacy
        - 'real_group_nons_discr': real group non-sensitive discrimination
        - 'nons_group_real_discr': non-sensitive group real discrimination
        contrast types (8, prefixed with contrast_):
        - 'contrast_homonyms', 'contrast_figurative_language'
        - 'contrast_safe_contexts', 'contrast_safe_targets'
        - 'contrast_definitions', 'contrast_historical_events'
        - 'contrast_privacy', 'contrast_discr'
    [note] (str): note (e.g. violence, drugs, etc.), may be None
    [label] (str): label (safe/unsafe)

original fields - XSTestResponse (allenai/xstest-response, split=response_harmfulness):
    [prompt] (str): user prompt
    [response] (str): model response
    [label] (str): label (unharmful/harmful)
    [prompt_type] (str): prompt type (prompt_safe/prompt_unsafe)
    [prompt_harm_category] (str): prompt harm category, 18 values (same as XSTest type field)

data analysis - XSTest (split=test, 450 samples):
    - type and label strongly correlated:
        - contrast_* types all unsafe (200)
        - non-contrast_* types all safe (250)
    - note field 77% is None (345/450), not used
    - **use type as harm category**

data analysis - XSTestResponse (split=response_harmfulness):
    - 39% of contrast_* categories are harmful (76/197)
    - contrast_ prefix cannot be simply equivalent to safe
"""


class XSTestAdapter(BaseAdapter):
    SPLIT = "test"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        type_val = example.get("type", "")
        focus_val = example.get("focus", "")

        # use type as harm category
        category = make_category(
            harm_labels=[type_val] if type_val else None,
            metadata={"focus": focus_val} if focus_val else None
        )

        return {
            "prompt": example.get("prompt"),
            "response": "",
            "prompt_label": "Safe" if example.get("label") == 'safe' else 'Unsafe',
            "response_label": None,
            "category": category,
            "source": "XSTest"
        }


class XSTestResponseAdapter(BaseAdapter):
    SPLIT = "response_harmfulness"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt_harm_category = example.get("prompt_harm_category", "")
        prompt_type = example.get("prompt_type", "")

        category = make_category(
            harm_labels=[prompt_harm_category] if prompt_harm_category else None,
            metadata={"prompt_type": prompt_type} if prompt_type else None
        )

        return {
            "prompt": example.get("prompt"),
            "response": example.get("response"),
            "prompt_label": None,  # dataset has no this field
            "response_label": "Safe" if example.get("label") == 'unharmful' else 'Unsafe',
            "category": category,
            "source": "XSTest-Response"
        }
