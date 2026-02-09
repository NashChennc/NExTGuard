from .base import BaseAdapter, make_category
from typing import Dict, Any, List, Tuple
import json

"""
original fields (lmsys/toxic-chat, subset=toxicchat0124, split=test):
    [conv_id] (str): conversation unique identifier
    [user_input] (str): user input
    [model_output] (str): model output
    [human_annotation] (bool): whether there is human annotation
    [toxicity] (int): toxicity label (0/1)
    [jailbreaking] (int): jailbreak attack label (0/1)
    [openai_moderation] (str): OpenAI Moderation API moderation result
        - data type: JSON string, parsed as list
        - format: [[category name, score], [category name, score], ...]
        - example categories:
            - 'self-harm/intent', 'self-harm/instructions', 'self-harm'
            - 'violence/graphic', 'violence'
            - 'sexual', 'sexual/minors'
            - 'hate', 'hate/threatening'
            - 'harassment', 'harassment/threatening'
        - score range: 0.0 ~ 1.0 (probability value)
        - max score: 0.9993
        - score > 0.5: 82 samples, score > 0.1: 327 samples

data analysis (subset=toxicchat0124, split=test, 5083 samples):
    - toxicity=1: 362 samples (7.1%)
    - jailbreaking=1: 91 samples (1.8%)
    - jailbreaking=1 必有 toxicity=1 (91 samples all overlap)
    - toxicity only (no jailbreaking): 271 samples
    - both are 0 (Safe): 4721 samples (92.9%)

normalization processing:
    - toxicity/jailbreaking: binary label → 1.0 (exists) or not added (not exists)
    - openai_moderation: parse JSON, extract probability scores for each category
        - scores are in [0,1] range, directly use
        - filter threshold: score > 0.01 (filter noise)
        - key prefix: "openai:" to distinguish source (e.g. "harm:openai:violence")
"""

# openai_moderation score filter threshold (filter noise below this value)
MODERATION_SCORE_THRESHOLD = 0.01


class ToxicChatAdapter(BaseAdapter):

    SUBSET = "toxicchat0124"
    SPLIT = "test"

    def _parse_openai_moderation(self, moderation_str: str) -> List[Tuple[str, float]]:
        """parse openai_moderation JSON string, return (category, score) list
        
        Args:
            moderation_str: JSON string, format: [[category, score], ...]
        
        Returns:
            filtered (category, score) list, only include items with score > MODERATION_SCORE_THRESHOLD
        """
        if not moderation_str:
            return []
        try:
            moderation_list = json.loads(moderation_str)
            # filter low score noise, return (category, score) tuple list
            return [
                (f"openai:{cat}", score) 
                for cat, score in moderation_list 
                if score > MODERATION_SCORE_THRESHOLD
            ]
        except (json.JSONDecodeError, TypeError):
            return []

    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        toxicity = example.get("toxicity") == 1
        jailbreaking = example.get("jailbreaking") == 1

        prompt_label = "Unsafe" if (toxicity or jailbreaking) else "Safe"
        # prompt_label = "Unsafe" if jailbreaking else "Safe"

        # 1. use toxicity/jailbreaking as main harm category (binary label → 1.0)
        harm_labels = []
        if toxicity:
            harm_labels.append("toxicity")  # string format, default 1.0
        if jailbreaking:
            harm_labels.append("jailbreaking")

        # 2. add probability scores for each category of openai_moderation
        moderation_scores = self._parse_openai_moderation(
            example.get("openai_moderation", "")
        )
        # tuple format: (category, score)
        harm_labels.extend(moderation_scores)

        category = make_category(harm_labels=harm_labels if harm_labels else None)

        return {
            "prompt": example.get("user_input", ""),
            "response": example.get("model_output", ""),
            "prompt_label": prompt_label,
            "response_label": None,  # dataset has no response label
            "category": category,
            "source": "ToxicChat"
        }
