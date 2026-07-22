"""Small dependency-free localization helper shared by CLI and legacy UI."""

import locale
import os
from typing import Iterable, List, Optional, Tuple


_language = "zh-CN"


def normalize_language(value: Optional[str]) -> str:
    if value:
        lowered = value.replace("_", "-").lower()
        if lowered.startswith("zh"):
            return "zh-CN"
        if lowered.startswith("en"):
            return "en-US"
    return "zh-CN" if (locale.getdefaultlocale()[0] or "").lower().startswith("zh") else "en-US"


def set_language(value: Optional[str]) -> str:
    global _language
    _language = normalize_language(value or os.environ.get("LVE_LANG"))
    os.environ["LVE_LANG"] = _language
    return _language


def get_language() -> str:
    return _language


def is_chinese() -> bool:
    return _language == "zh-CN"


def tr(chinese: str, english: str) -> str:
    return chinese if is_chinese() else english


def extract_language(argv: Iterable[str]) -> Tuple[str, List[str]]:
    """Consume --language/-L before argparse builds localized help text."""
    values = list(argv)
    selected = None
    result: List[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value.startswith("--language="):
            selected = value.split("=", 1)[1]
        elif value in {"--language", "-L"}:
            index += 1
            if index >= len(values):
                raise ValueError("--language requires zh-CN or en-US")
            selected = values[index]
        else:
            result.append(value)
        index += 1
    language = set_language(selected)
    return language, result


set_language(None)
