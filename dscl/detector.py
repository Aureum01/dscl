# detector.py
# reads the narrative the user passes in and returns the best matching
# domain, register, and fk target from the dictionary.
# all signal words, domains, and fk targets live in domain_register.json —
# this file only reads and matches. nothing is hardcoded here.

import re
from typing import Optional
from .loader import DictionaryLoader


class DomainDetector:
    """
    detects the domain of a narrative by matching signal words and
    structural patterns against domain_register.json.
    falls back to defaults.json if no domain can be confidently identified.
    neutral by design — makes no assumptions about subject matter,
    geography, culture, or intent.
    """

    def __init__(self, loader: DictionaryLoader):
        self.loader = loader

        # signal word lists per domain — populated from domain_register.json
        self._domain_signals: dict[str, list[str]] = {}
        self._domain_register: dict = {}
        self._defaults: dict = {}
        self._ready = False

    def _build(self):
        if self._ready:
            return

        raw = self.loader.domain_register()
        self._defaults = self.loader.defaults()

        # domain_register.json uses a "domains" array of objects.
        # reshape it into a flat dict keyed by domain name
        # so the rest of the detector can look domains up by key.
        domains_list = raw.get("domains", [])
        self._domain_register = {
            entry["domain"]: entry
            for entry in domains_list
            if isinstance(entry, dict) and "domain" in entry
        }

        for domain, config in self._domain_register.items():
            signals = config.get("signals") or config.get("examples") or []
            if signals:
                self._domain_signals[domain] = [
                    s.lower() for s in signals
                ]

        self._ready = True

    def detect(
        self,
        narrative: str,
        override_domain: Optional[str] = None,
        override_audience: Optional[str] = None,
    ) -> dict:
        """
        takes a narrative string and returns a detection result dict with:
          domain     — the matched or overridden domain key
          fk_target  — the grade range string (e.g. "5-7")
          register   — the register tag (e.g. "abstract_general")
          confidence — how many signals matched (0 means fallback was used)
          exceptions — domain exception words active for this domain
          template   — the matched template dict for this domain, or {} if none
        """
        self._build()

        if override_domain:
            return self._build_result(
                domain=override_domain,
                audience=override_audience,
                confidence=-1,  # -1 signals a manual override, not a detection
            )

        domain, confidence = self._match_domain(narrative)

        return self._build_result(
            domain=domain,
            audience=override_audience,
            confidence=confidence,
        )

    def _match_domain(self, narrative: str) -> tuple[str, int]:
        # scores each domain by counting how many of its signal words
        # appear in the lowercased narrative.
        # returns the highest scoring domain and its signal count.
        # ties go to whichever domain appears first in domain_register.json.

        text = narrative.lower()
        scores: dict[str, int] = {}

        for domain, signals in self._domain_signals.items():
            score = sum(1 for signal in signals if signal in text)
            if score > 0:
                scores[domain] = score

        if not scores:
            # nothing matched — use the default domain from defaults.json
            fallback = self._defaults.get("domain", "plain_summary")
            return fallback, 0

        best_domain = max(scores, key=lambda d: scores[d])
        return best_domain, scores[best_domain]

    def _build_result(
        self,
        domain: str,
        audience: Optional[str],
        confidence: int,
    ) -> dict:
        # pulls the fk_target and register for the domain from domain_register.
        # if the domain is not in domain_register, uses defaults.json.

        config = self._domain_register.get(domain, {})

        fk_target = (
            self._resolve_fk(audience, config)
            if audience
            else config.get("fk_target", self._defaults.get("fk_target", "6-8"))
        )

        register = config.get(
            "register",
            self._defaults.get("register", "abstract_general")
        )

        exceptions = self._load_exceptions_for_domain(domain)
        template   = self._load_template_for_domain(domain)

        return {
            "domain":     domain,
            "fk_target":  fk_target,
            "register":   register,
            "confidence": confidence,
            "exceptions": exceptions,
            "template":   template,
        }

    def _resolve_fk(self, audience: str, config: dict) -> str:
        # audience overrides let callers request a different reading level
        # without changing the domain rules.
        # audience keys live in domain_register.json under "audience_fk":
        # e.g. {"general_public": "5-7", "professional": "9-11"}

        audience_map = config.get("audience_fk", {})
        return audience_map.get(
            audience.lower(),
            config.get("fk_target", self._defaults.get("fk_target", "6-8"))
        )

    def _load_exceptions_for_domain(self, domain: str) -> list[str]:
        # returns words allowed to bypass vocabulary rules for this domain.
        # domain-specific exceptions and global exceptions are merged.
        # both live in vocabulary/domain_exceptions.json.

        exceptions_data = self.loader.domain_exceptions()
        domain_exceptions = exceptions_data.get(domain, [])
        global_exceptions = exceptions_data.get("global", [])
        return list(set(domain_exceptions + global_exceptions))

    def _load_template_for_domain(self, domain: str) -> dict:
        # returns the template dict for the domain from templates.json.
        # if no template exists for this domain, returns {}.
        # core.py decides what to do when template is empty.

        templates_data = self.loader.templates()
        return templates_data.get("templates", {}).get(domain, {})