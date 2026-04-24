# loader.py
# responsible for one thing: reading the /dictionary JSON files from disk
# and returning them as plain python dicts the rest of the library can use.
# nothing in here does logic — it only loads and caches.

import json
import os
from pathlib import Path
from typing import Optional


# default path assumes /dictionary sits next to the dscl package
# in the LITATURE project root — can be overridden at DSCL() init
DEFAULT_DICTIONARY_PATH = Path(__file__).parent.parent.parent / "dictionary"


class DictionaryLoader:
    """
    loads rule files from the /dictionary folder on demand.
    caches everything after first read so repeated calls are free.
    all methods return plain dicts — no special types, no surprises.
    """

    def __init__(self, dictionary_path: Optional[Path] = None):
        # where to look for the /dictionary folder
        self.dictionary_path = Path(dictionary_path) if dictionary_path else DEFAULT_DICTIONARY_PATH

        # simple in-memory cache — key is the relative file path
        self._cache: dict[str, dict] = {}

        self._verify_dictionary_exists()

    def _verify_dictionary_exists(self):
        # fail early and clearly if the dictionary folder cannot be found
        if not self.dictionary_path.exists():
            raise FileNotFoundError(
                f"dictionary folder not found at: {self.dictionary_path}\n"
                f"set the correct path with DSCL(dictionary_path='your/path')"
            )

    def _load(self, relative_path: str) -> dict:
        if relative_path in self._cache:
            return self._cache[relative_path]

        full_path = self.dictionary_path / relative_path

        if not full_path.exists():
            return {}

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return {}

        data = json.loads(content)
        self._cache[relative_path] = data
        return data

    def domain_vocab(self) -> dict:
        return self._load("vocabulary/domain_vocab.json")

    # ── vocabulary ────────────────────────────────────────────────────────────

    def primary_vocab(self) -> dict:
        return self._load("vocabulary/primary_vocab.json")

    def fallback_vocab(self) -> dict:
        return self._load("vocabulary/fallback_vocab.json")

    def domain_exceptions(self) -> dict:
        # words allowed to bypass vocabulary rules per domain
        return self._load("vocabulary/domain_exceptions.json")

    def templates(self) -> dict:
        # report structure templates — section order, sentence budgets, variables
        # custom templates in dictionary/custom/templates/ are merged at load time
        data = self._load("vocabulary/templates.json")
        custom = self._load_custom_templates()
        if custom:
            # custom keys override built-in keys of the same name
            merged = dict(data)
            merged.get("templates", {}).update(custom.get("templates", {}))
            return merged
        return data

    def _load_custom_templates(self) -> dict:
        # walk dictionary/custom/templates/ and merge any JSON files found
        custom_dir = self.dictionary_path / "custom" / "templates"
        if not custom_dir.exists():
            return {}
        merged: dict = {"templates": {}}
        for f in sorted(custom_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                if content:
                    data = json.loads(content)
                    merged["templates"].update(data.get("templates", {}))
            except (json.JSONDecodeError, OSError):
                pass
        return merged

    # ── syntax ────────────────────────────────────────────────────────────────

    def sentence_structure(self) -> dict:
        return self._load("syntax/sentence_structure.json")

    def clause_rules(self) -> dict:
        return self._load("syntax/clause_rules.json")

    def paragraph_structure(self) -> dict:
        return self._load("syntax/paragraph_structure.json")

    def punctuation_function(self) -> dict:
        return self._load("syntax/punctuation_function.json")

    # ── grammar ───────────────────────────────────────────────────────────────

    def voice_rules(self) -> dict:
        return self._load("grammar/voice_rules.json")

    def verb_placement(self) -> dict:
        return self._load("grammar/verb_placement.json")

    def modifier_rules(self) -> dict:
        return self._load("grammar/modifier_rules.json")

    def nominalization_blocklist(self) -> dict:
        # words ending in -tion, -ness, -ity etc that should be recast as verbs
        return self._load("grammar/nominalization_blocklist.json")

    # ── readability ───────────────────────────────────────────────────────────

    def grade_targets(self) -> dict:
        return self._load("readability/grade_targets.json")

    def sentence_length(self) -> dict:
        return self._load("readability/sentence_length.json")

    def syllable_load(self) -> dict:
        return self._load("readability/syllable_load.json")

    # ── context ───────────────────────────────────────────────────────────────

    def register_tags(self) -> dict:
        return self._load("context/register_tags.json")

    def domain_register(self) -> dict:
        # maps domain names to their register and fk target
        return self._load("context/domain_register.json")

    def rules(self) -> dict:
        # global and domain-scoped linguistic rules for validation and correction
        # custom overrides in dictionary/custom/rules/ are deep-merged on top
        data = self._load("context/rules.json")
        custom = self._load_custom_rules()
        if custom:
            merged = dict(data)
            # domain_overrides are merged per-key, not replaced wholesale
            base_overrides = merged.get("domain_overrides", {})
            for domain, domain_rules in custom.get("domain_overrides", {}).items():
                if domain in base_overrides:
                    base_overrides[domain].update(domain_rules)
                else:
                    base_overrides[domain] = domain_rules
            # global rules are also merged per-key
            base_global = merged.get("global", {})
            base_global.update(custom.get("global", {}))
            return merged
        return data

    def _load_custom_rules(self) -> dict:
        # walk dictionary/custom/rules/ and merge any JSON files found
        custom_dir = self.dictionary_path / "custom" / "rules"
        if not custom_dir.exists():
            return {}
        merged: dict = {"global": {}, "domain_overrides": {}}
        for f in sorted(custom_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                if content:
                    data = json.loads(content)
                    merged["global"].update(data.get("global", {}))
                    for domain, domain_rules in data.get("domain_overrides", {}).items():
                        if domain in merged["domain_overrides"]:
                            merged["domain_overrides"][domain].update(domain_rules)
                        else:
                            merged["domain_overrides"][domain] = domain_rules
            except (json.JSONDecodeError, OSError):
                pass
        return merged

    # ── system ────────────────────────────────────────────────────────────────

    def pipeline(self) -> dict:
        return self._load("system/pipeline.json")

    def entry_rules(self) -> dict:
        return self._load("system/entry_rules.json")

    def scope(self) -> dict:
        return self._load("system/scope.json")

    def defaults(self) -> dict:
        return self._load("system/defaults.json")

    # ── meta ──────────────────────────────────────────────────────────────────

    def conflict_resolution(self) -> dict:
        # rules for what wins when two constraints contradict each other
        return self._load("meta/conflict_resolution.json")

    def weight_definitions(self) -> dict:
        # hard / soft / contextual weight definitions
        return self._load("meta/weight_definitions.json")

    def clear_cache(self):
        # useful in tests or when dictionary files are updated during a session
        self._cache.clear()