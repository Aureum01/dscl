# core.py
from pathlib import Path
from typing import Optional
import json

from .loader import DictionaryLoader
from .detector import DomainDetector
from .validator import OutputValidator
from .models import ConstrainedContext, ValidationResult


SYSTEM_PROMPT_TEMPLATE = """
You are a precise, plain-language writer.
Follow these constraints when composing your response:

Domain: {domain}
Register: {register}
Flesch-Kincaid target grade: {fk_target}

Vocabulary rules:
- Use the simplest word that carries the full meaning.
- Prefer active voice over passive.
- Avoid nominalizations where a verb is clearer.
- Do not use weak modifiers (very, quite, rather, somewhat).
- Where possible, prefer words from the domain vocabulary list below.

Domain vocabulary — reach for these words first:
{domain_vocab}

Structure rules:
- Keep sentences under 25 words where possible.
- One idea per sentence.
- One topic per paragraph.

The following words are permitted despite vocabulary constraints: {exceptions}

{rules_block}

{template_block}

{only_text_block}

{live_block}
""".strip()


# injected into the system prompt when only_text=True.
# tells the model to write in plain prose with no markdown of any kind.
# the template section order still guides content structure internally,
# but everything renders as flowing paragraphs — no headings, no bullets.
ONLY_TEXT_INSTRUCTION = """
FORMAT: Plain prose only.
Do not use any markdown formatting. This means:
- No headers (no #, ##, ###)
- No bullet points (no -, *, +)
- No numbered lists
- No bold or italic text (**word** or *word*)
- No tables
- No horizontal rules (---)
- No code blocks
Write in paragraphs only. Use the template section order to organise
the content internally, but render everything as flowing prose.
""".strip()


# injected into the system prompt when live=True.
# signals to a web-capable model that current information is required.
# for models without web access, this prevents confident statements
# of stale data — the model should flag uncertainty instead.
LIVE_MODE_INSTRUCTION = """
CURRENCY: This response requires current, real-time information.
Do not rely on training data for facts that change over time.
This includes: security conditions, travel advisories, threat levels,
flight status, political developments, and any time-sensitive data.
If you have web access, search for the latest information before responding.
If you do not have web access, state clearly that the information
may be outdated and tell the reader to verify with current sources
such as their government's official travel advisory page.
""".strip()


def _format_rules_block(rules: dict, domain: str) -> str:
    # builds a plain-text rules section injected into the system prompt.
    # merges global rules with any domain overrides for the active domain.
    # returns an empty string if rules.json is not present or empty.

    if not rules:
        return ""

    lines = ["Language rules (apply in order):"]

    global_rules = rules.get("global", {})
    domain_overrides = rules.get("domain_overrides", {}).get(domain, {})

    # banned words — merge global list with any domain exemptions
    banned = list(global_rules.get("banned_words", {}).get("terms", []))
    exempt = domain_overrides.get("banned_words_exempt", [])
    active_banned = [w for w in banned if w not in exempt]
    if active_banned:
        lines.append(f"- Never use: {', '.join(active_banned)}")

    # banned phrases specific to this domain
    domain_banned_phrases = domain_overrides.get("banned_phrases", [])
    global_banned_phrases = [
        t for t in global_rules.get("hedging_language", {}).get("terms", [])
    ]
    all_banned_phrases = list(set(global_banned_phrases + domain_banned_phrases))
    if all_banned_phrases:
        lines.append(f"- Never use these phrases: {', '.join(all_banned_phrases)}")

    # no unsolicited headers — global rule
    no_headers = global_rules.get("no_unsolicited_headers", {})
    if no_headers:
        examples = no_headers.get("examples_banned", [])
        note = no_headers.get("note", "")
        if note:
            lines.append(f"- {note}")
        if examples:
            lines.append(f"  Never add: {', '.join(examples[:5])}")

    # certainty language replacements (domain-specific)
    certainty = domain_overrides.get("certainty_language", {}).get("replacements", {})
    if certainty:
        replacements = "; ".join(f'"{k}" -> "{v}"' for k, v in certainty.items())
        lines.append(f"- Replace weak modals: {replacements}")

    # passive voice rule — domain override wins over global
    passive_rule = domain_overrides.get("passive_voice") or \
                   global_rules.get("passive_voice", {}).get("severity", "flag")
    if passive_rule == "flag":
        lines.append("- Prefer active voice. Flag passive constructions.")
    elif passive_rule == "allow":
        lines.append("- Passive voice is permitted in this domain.")

    # sentence length — domain override wins
    sent_len = domain_overrides.get("sentence_length") or global_rules.get("sentence_length", {})
    max_words = sent_len.get("max_words")
    if max_words:
        lines.append(f"- Keep sentences under {max_words} words.")

    # nominalizations
    noms = global_rules.get("nominalizations", {}).get("examples", {})
    if noms:
        nom_examples = "; ".join(f'"{k}" -> "{v}"' for k, v in list(noms.items())[:5])
        lines.append(f"- Avoid nominalizations. Examples: {nom_examples}")

    # redundant phrases
    redundant = global_rules.get("redundant_phrases", {}).get("replacements", {})
    if redundant:
        red_examples = "; ".join(f'"{k}" -> "{v}"' for k, v in list(redundant.items())[:5])
        lines.append(f"- Replace redundant phrases. Examples: {red_examples}")

    # anonymization (domain-specific)
    anon = domain_overrides.get("anonymization", {})
    if anon:
        lines.append(f"- {anon.get('description', 'Do not use personal names. Use titles only.')}")

    # objectivity (domain-specific)
    obj = domain_overrides.get("objectivity", {})
    if obj:
        tone_words = obj.get("banned_tone_words", [])
        if tone_words:
            lines.append(f"- Remove sensationalizing language. Never use: {', '.join(tone_words)}")

    # evidence grounding (domain-specific)
    evidence = domain_overrides.get("evidence_grounding", {})
    if evidence:
        lines.append(f"- {evidence.get('description', 'Ground all claims in the provided evidence.')}")

    return "\n".join(lines)


def _format_template_block(template: dict) -> str:
    # builds a plain-text template section injected into the system prompt.
    # tells the model the section order, sentence budgets, and per-section
    # instructions so output is consistent regardless of which model runs it.
    # returns an empty string if no template is defined for this domain.

    if not template:
        return ""

    lines = ["Report template (follow exactly):"]

    description = template.get("description", "")
    if description:
        lines.append(f"Purpose: {description}")

    composition_order = template.get("composition_order", [])
    render_order = template.get("render_order", [])
    note = template.get("note", "")

    if composition_order:
        lines.append(f"Compose in this order: {' -> '.join(composition_order)}")
    if render_order:
        lines.append(f"Render in this order: {' -> '.join(render_order)}")
    if note:
        lines.append(f"Note: {note}")

    # variables the caller must supply
    variables = template.get("variables", {})
    required_vars = [k for k, v in variables.items() if v.get("required")]
    if required_vars:
        lines.append(f"Required variables: {', '.join(required_vars)}")

    # per-section instructions
    sections = template.get("sections", {})
    if sections:
        lines.append("\nSections:")
        for section_key, section in sections.items():
            label = section.get("label", section_key)
            desc  = section.get("description", "")
            length = section.get("length", {})
            min_s  = length.get("min_sentences", "")
            max_s  = length.get("max_sentences", "")
            extend = length.get("extend_to", "")
            extend_cond = length.get("extend_condition", "")

            length_str = ""
            if min_s and max_s:
                length_str = f"{min_s}-{max_s} sentences"
                if extend:
                    length_str += f" (extend to {extend} only if: {extend_cond})"
            elif max_s:
                length_str = f"max {max_s} sentences"

            lines.append(f"\n[{label}]")
            if desc:
                lines.append(f"  {desc}")
            if length_str:
                lines.append(f"  Length: {length_str}")

            structure = section.get("structure", {})
            for part_key, part in structure.items():
                role        = part.get("role", part.get("type", part_key))
                instruction = part.get("instruction", "")
                if instruction:
                    lines.append(f"  {role}: {instruction}")

            section_rules = section.get("rules", {})
            if section_rules.get("no_new_information"):
                lines.append("  Do not introduce information not present in the source sections.")
            if section_rules.get("evidence_required"):
                lines.append("  Every claim must trace back to the latest developments.")
            priority = section_rules.get("priority_topic")
            priority_note = section_rules.get("priority_note")
            if priority and priority_note:
                lines.append(f"  Priority: {priority_note}")

    return "\n".join(lines)


class DSCL:
    """
    The main class. Instantiate once, call prepare() and validate() as needed.

    dscl = DSCL()
    constrained = dscl.prepare("your narrative here")
    result = dscl.validate(model_output)

    Optional flags on prepare():
      only_text=True  — output plain prose, no markdown of any kind
      live=True       — signal that current real-time information is required
    """

    def __init__(self, dictionary_path: Optional[str | Path] = None):
        self.loader    = DictionaryLoader(dictionary_path)
        self.detector  = DomainDetector(self.loader)
        self.validator = OutputValidator(self.loader)

    def prepare(
        self,
        narrative:  str,
        domain:     Optional[str] = None,
        audience:   Optional[str] = None,
        only_text:  bool = False,
        live:       bool = False,
    ) -> ConstrainedContext:
        """
        Reads the narrative, detects the domain, loads the right rules and
        template, and builds a system prompt with all constraints injected.

        Returns a ConstrainedContext. Pass its system_prompt and prompt
        fields directly to your model call.

        only_text — when True, adds a plain-prose-only formatting instruction.
                    No markdown, no bullets, no headers. Paragraphs only.

        live      — when True, adds a currency instruction telling the model
                    to use current information and not rely on training data
                    for time-sensitive facts.
        """
        detection = self.detector.detect(
            narrative,
            override_domain=domain,
            override_audience=audience,
        )

        domain_vocab   = self._resolve_domain_vocab(detection["domain"])
        rules          = self.loader.rules()
        template       = detection["template"]

        rules_block    = _format_rules_block(rules, detection["domain"])
        template_block = _format_template_block(template)
        only_text_block = ONLY_TEXT_INSTRUCTION if only_text else ""
        live_block      = LIVE_MODE_INSTRUCTION if live else ""

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            domain          = detection["domain"],
            register        = detection["register"],
            fk_target       = detection["fk_target"],
            domain_vocab    = domain_vocab,
            exceptions      = ", ".join(detection["exceptions"]) or "none",
            rules_block     = rules_block,
            template_block  = template_block,
            only_text_block = only_text_block,
            live_block      = live_block,
        )

        rules_loaded = self._list_rules_loaded(detection["domain"])

        return ConstrainedContext(
            narrative         = narrative,
            domain            = detection["domain"],
            fk_target         = detection["fk_target"],
            register          = detection["register"],
            system_prompt     = system_prompt,
            prompt            = narrative,
            rules_loaded      = rules_loaded,
            active_exceptions = detection["exceptions"],
            template          = template,
            only_text         = only_text,
            live_mode         = live,
        )

    def validate(
        self,
        text:       str,
        domain:     str = "",
        exceptions: Optional[list[str]] = None,
    ) -> ValidationResult:

        return self.validator.validate(
            text       = text,
            domain     = domain,
            exceptions = exceptions,
        )

    def prepare_and_validate(
        self,
        narrative:    str,
        model_output: str,
        domain:       Optional[str] = None,
        audience:     Optional[str] = None,
        only_text:    bool = False,
        live:         bool = False,
    ) -> tuple[ConstrainedContext, ValidationResult]:

        context = self.prepare(
            narrative,
            domain    = domain,
            audience  = audience,
            only_text = only_text,
            live      = live,
        )
        result = self.validate(
            model_output,
            domain     = context.domain,
            exceptions = context.active_exceptions,
        )
        return context, result

    def _resolve_domain_vocab(self, domain: str) -> str:
        # loads the preferred word list for the detected domain.
        # falls back to "general" if the domain has no specific list.
        # returns a comma-separated string for injection into the system prompt.

        vocab_data = self.loader.domain_vocab()

        words = (
            vocab_data.get(domain)
            or vocab_data.get("general")
            or []
        )

        return ", ".join(words) if words else "none"

    def _list_rules_loaded(self, domain: str) -> list[str]:
        candidates = {
            "sentence_length":           self.loader.sentence_length(),
            "voice_rules":               self.loader.voice_rules(),
            "modifier_rules":            self.loader.modifier_rules(),
            "nominalization_blocklist":  self.loader.nominalization_blocklist(),
            "grade_targets":             self.loader.grade_targets(),
            "domain_exceptions":         self.loader.domain_exceptions(),
            "domain_vocab":              self.loader.domain_vocab(),
            "conflict_resolution":       self.loader.conflict_resolution(),
            "weight_definitions":        self.loader.weight_definitions(),
            "rules":                     self.loader.rules(),
            "templates":                 self.loader.templates(),
        }

        return [name for name, data in candidates.items() if data]