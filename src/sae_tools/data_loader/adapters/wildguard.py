from .base import BaseAdapter, make_category
from typing import Dict, Any, Optional

"""
original fields - WildGuardTest (walledai/WildGuardTest, split=train):
    [prompt] (str): user prompt
    [adversarial] (bool): whether adversarial input (True/False)
    [label] (str): label (unharmful/harmful/None)

original fields - WildGuardMix (allenai/wildguardmix, subset=wildguardtest, split=test):
    [prompt] (str): user prompt
    [response] (str): model response
    [adversarial] (bool): whether adversarial input (True/False)
    [prompt_harm_label] (str): prompt harm label (unharmful/harmful/None)
    [response_refusal_agreement] (float): response refusal agreement score
    [response_refusal_label] (str): response refusal label (compliance/refusal)
    [response_harm_label] (str): response harm label (unharmful/harmful/None)
    [subcategory] (str): subcategory, 15 values:
        - 'benign': benign
        - 'causing_material_harm_by_disseminating_misinformation': causing material harm by disseminating misinformation
        - 'copyright_violations': copyright violations
        - 'cyberattack': cyber attack
        - 'defamation_encouraging_unethical_or_unsafe_actions': defamation encouraging unethical or unsafe actions
        - 'disseminating_false_or_misleading_information_encouraging_disinformation_campaigns': disseminating false or misleading information encouraging disinformation campaigns
        - 'fraud_assisting_illegal_activities': fraud assisting illegal activities
        - 'mental_health_over-reliance_crisis': mental health over-reliance crisis
        - 'others': others
        - 'private_information_individual': private information individual
        - 'sensitive_information_organization_government': sensitive information organization government
        - 'sexual_content': sexual content
        - 'social_stereotypes_and_unfair_discrimination': social stereotypes and unfair discrimination
        - 'toxic_language_hate_speech': toxic language hate speech
        - 'violence_and_physical_harm': violence and physical harm
    [prompt_harm_agreement] (float): prompt harm agreement score
    [response_harm_agreement] (float): response harm agreement score

data analysis - WildGuardTest (split=train, 1725 samples):
    - label distribution: unharmful=945, harmful=754, None=26
    - **label=None**: 26 samples, prompt_label returns None
    - adversarial distribution: False=915, True=810

data analysis - WildGuardMix (subset=wildguardtest, split=test, 1725 samples):
    - prompt_harm_label distribution: unharmful=945, harmful=754, None=26
    - response_harm_label distribution: unharmful=1425, harmful=284, None=16
    - prompt_harm_label != response_harm_label: 517 samples (normal phenomenon)
    - subcategory="benign": 971 samples (56.3%), not add harm category

normalization processing:
    - adversarial: boolean value → 1.0 (exists)
    - prompt_harm_agreement: prompt harm agreement score, in [0,1] range, directly use
    - response_harm_agreement: response harm agreement score, in [0,1] range, directly use
    - these scores represent the consistency of the annotators' harm labels, the higher the value, the more reliable the annotation
"""


def _normalize_harm_label(label: Optional[str]) -> Optional[str]:
    """normalize harmful/unharmful label to Safe/Unsafe/None"""
    if label is None:
        return None
    if label == "unharmful":
        return "Safe"
    elif label == "harmful":
        return "Unsafe"
    return None  # unknown label returns None


class WildGuardAdapter(BaseAdapter):
    SPLIT = "train"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        label = example.get("label")
        prompt_label = _normalize_harm_label(label)

        # build category
        metadata = {}
        if example.get("adversarial"):
            metadata["adversarial"] = True

        category = make_category(metadata=metadata if metadata else None)

        return {
            "prompt": example.get("prompt"),
            "response": "",
            "prompt_label": prompt_label,
            "response_label": None,
            "category": category,
            "source": "WildGuard"
        }


class WildGuardMixAdapter(BaseAdapter):
    SPLIT = "test"
    SUBSET = "wildguardtest"

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt_label = _normalize_harm_label(example.get("prompt_harm_label"))
        response_label = _normalize_harm_label(example.get("response_harm_label"))

        # build category
        subcategory = example.get("subcategory", "")
        harm_labels = []
        # subcategory="benign" not add harm category
        if subcategory and subcategory != "benign":
            harm_labels.append(subcategory)

        # build metadata, include agreement scores
        metadata = {}
        if example.get("adversarial"):
            metadata["adversarial"] = True
        
        # add annotation agreement scores (in [0,1] range)
        prompt_agreement = example.get("prompt_harm_agreement")
        response_agreement = example.get("response_harm_agreement")
        if prompt_agreement is not None:
            metadata["prompt_agreement"] = prompt_agreement
        if response_agreement is not None:
            metadata["response_agreement"] = response_agreement

        category = make_category(
            harm_labels=harm_labels if harm_labels else None,
            metadata=metadata if metadata else None
        )

        return {
            "prompt": example.get("prompt"),
            "response": example.get("response") or "",
            "prompt_label": prompt_label,
            "response_label": response_label,
            "category": category,
            "source": "WildGuardMix"
        }
