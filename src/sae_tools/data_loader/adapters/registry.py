from .qwen import Qwen3GuardTestAdapter
from .toxic_chat import ToxicChatAdapter
from .aegis import Aegis2Adapter
from .aegis_v1 import AegisAdapter
from .wildguard import WildGuardAdapter, WildGuardMixAdapter
from .beavertails import BeaverTailsAdapter, BeaverTailsAugmentedAdapter
from .xs import XSTestAdapter, XSTestResponseAdapter
from .safe_rlhf import SafeRLHFAdapter
from .strongreject import StrongREJECTAdapter
from .wildjailbreak import WildJailbreakAdapter
from .harmbench import HarmBenchAdapter
from .harmbench import HarmBenchResponseAdapter
from .openaimod import OpenAIModAdapter
from .simplesafetytest import SimpleSafetyTestAdapter

from .base import BaseAdapter

ADAPTER_REGISTRY = {
    "Qwen3GuardTest": Qwen3GuardTestAdapter(),
    "ToxicChat": ToxicChatAdapter(),
    "Aegis2.0": Aegis2Adapter(),
    "Aegis1.0": AegisAdapter(),

    "BeaverTails": BeaverTailsAdapter(),
    "BeaverTailsAugmented": BeaverTailsAugmentedAdapter(),
    "SafeRLHF": SafeRLHFAdapter(),

    "WildGuardTest": WildGuardAdapter(),
    "WildGuardMix": WildGuardMixAdapter(),

    "XSTest": XSTestAdapter(),
    "XSTestResponse": XSTestResponseAdapter(),

    "StrongREJECT": StrongREJECTAdapter(),
    "WildJailbreak": WildJailbreakAdapter(),

    "HarmBench": HarmBenchAdapter(),
    "HarmBenchResponse": HarmBenchResponseAdapter(),
    "OpenAIMod": OpenAIModAdapter(),
    "SimpleSafetyTest": SimpleSafetyTestAdapter(),
}


def get_adapter(dataset_name: str) -> BaseAdapter:
    adapter = ADAPTER_REGISTRY.get(dataset_name)
    if adapter is None:
        raise KeyError(f"Unknown dataset name: {dataset_name}")
    return adapter
