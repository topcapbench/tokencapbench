from __future__ import annotations

from budget2success.datasets.base import AdapterConfig, BenchmarkSourceAdapter
from budget2success.datasets.aider_polyglot import AiderPolyglotAdapter
from budget2success.datasets.bigcodebench_hard import BigCodeBenchHardAdapter
from budget2success.datasets.canitedit import CanItEditAdapter
from budget2success.datasets.loaders.assistantbench import AssistantBenchAdapter
from budget2success.datasets.loaders.bfcl import BFCLAdapter
from budget2success.datasets.loaders.bigcodebench import BigCodeBenchAdapter
from budget2success.datasets.loaders.evalplus import EvalPlusAdapter
from budget2success.datasets.loaders.gsm8k import GSM8KAdapter
from budget2success.datasets.loaders.hendrycks_math import HendrycksMATHAdapter
from budget2success.datasets.loaders.livecodebench import LiveCodeBenchAdapter
from budget2success.datasets.loaders.local_jsonl import LocalJSONLAdapter
from budget2success.datasets.loaders.swebench import SWEBenchAdapter
from budget2success.datasets.loaders.tau_bench import TauBenchAdapter

_ADAPTERS: dict[str, type[BenchmarkSourceAdapter]] = {
    "local": LocalJSONLAdapter,
    "gsm8k": GSM8KAdapter,
    "math": HendrycksMATHAdapter,
    "hendrycks_math": HendrycksMATHAdapter,
    "evalplus": EvalPlusAdapter,
    "bigcodebench": BigCodeBenchAdapter,
    "bigcodebench_hard": BigCodeBenchHardAdapter,
    "canitedit": CanItEditAdapter,
    "aider_polyglot": AiderPolyglotAdapter,
    "livecodebench": LiveCodeBenchAdapter,
    "swebench": SWEBenchAdapter,
    "assistantbench": AssistantBenchAdapter,
    "bfcl": BFCLAdapter,
    "tau2": TauBenchAdapter,
    "tau_bench": TauBenchAdapter,
}


def get_adapter(name: str, config: AdapterConfig | None = None) -> BenchmarkSourceAdapter:
    key = name.lower()
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown benchmark source '{name}'. Known sources: {sorted(_ADAPTERS)}")
    return _ADAPTERS[key](config)


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)
