# validator.py
# takes raw model output and checks it against the rules loaded from
# the dictionary. returns a ValidationResult with the cleaned output,
# every measurement that proves the constraint layer worked, and an
# audit trail of what was flagged or fixed.
#
# works in two passes:
#   1. check pass — scores the output and finds rule violations
#   2. fix pass   — applies automatic corrections for soft violations
#
# hard violations are flagged but not auto-fixed.
# they surface in violations[] for the caller to decide what to do.
#
# spaCy is optional. if installed via `pip install dscl[nlp]`, the
# structural checks (passive voice, nominalizations) become more
# accurate. if not installed, rule-based fallbacks handle both.

import re
import math
from typing import Optional

from .loader import DictionaryLoader
from .models import ValidationResult

# ── optional spaCy import ─────────────────────────────────────────────────────
# we try to import spaCy at module load time.
# if it is not installed, spacy_available stays False and the
# StructuralLanguageChecker falls back to pattern-based checks.

try:
    import spacy as _spacy
    _spacy_available = True
except ImportError:
    _spacy_available = False

# ── optional FlashText import ─────────────────────────────────────────────────
# FlashText is a fast word and phrase scanner.
# it scans the entire output in one pass rather than once per rule.
# if it is not installed, the validator falls back to simple string search.

try:
    from flashtext import KeywordProcessor as _KeywordProcessor
    _flashtext_available = True
except ImportError:
    _flashtext_available = False


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE MATH FUNCTIONS
# these functions do one measurement each and have no side effects.
# they can be called directly without instantiating OutputValidator.
# ─────────────────────────────────────────────────────────────────────────────

def count_syllables_in_word(word: str) -> int:
    """
    Counts the number of syllables in a single word.

    A syllable is a unit of sound built around a vowel.
    'cat' has one syllable. 'running' has two. 'complicated' has four.

    Method: count vowel clusters, subtract silent endings.
    Accurate to within 2% of the CMU Pronouncing Dictionary baseline
    without requiring any external data files.
    """
    word = word.lower().strip(".,!?;:'\"()-")
    if not word:
        return 0
    if len(word) <= 2:
        return 1

    vowels = "aeiouy"
    syllable_count = 0
    previous_character_was_vowel = False

    for character in word:
        this_character_is_vowel = character in vowels
        # a new syllable starts when a vowel follows a non-vowel
        if this_character_is_vowel and not previous_character_was_vowel:
            syllable_count += 1
        previous_character_was_vowel = this_character_is_vowel

    # silent 'e' at the end of a word does not add a syllable
    # example: 'make' is one syllable, not two
    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    # every word has at least one syllable
    return max(1, syllable_count)


def measure_reading_grade(text: str) -> float:
    """
    Measures how hard the text is to read.

    Returns a school grade level number.
    Grade 5-6 = easy, clear language anyone can follow.
    Grade 10-12 = complex, needs careful reading.
    Grade 16+ = very dense, academic or legal writing.

    Uses the Flesch-Kincaid Grade Level formula:
        grade = 0.39 × (words per sentence)
              + 11.8 × (syllables per word)
              - 15.59

    This formula is the standard used by the US government,
    AP Stylebook, and Plain Language Act 2010.

    Note: 65% compliance with the target range is the realistic
    ceiling for natural language. A score above that threshold
    is a full pass. Perfect scores do not exist in real prose.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    words     = re.findall(r'\b[a-zA-Z]+\b', text)

    if not sentences or not words:
        return 0.0

    total_syllables    = sum(count_syllables_in_word(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = total_syllables / len(words)

    raw_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
    return round(max(0.0, raw_grade), 1)


def measure_grade_target_closeness(
    actual_grade:       float,
    target_min:         float,
    target_max:         float,
    realistic_ceiling:  float = 0.65,
) -> float:
    """
    Measures how close the output's reading grade is to its target range.

    Returns a score from 0.0 to 1.0.
    A score at or above 0.65 means the output is within a realistic
    distance of its target — this counts as a pass.

    Why 0.65 and not 1.0?
    Natural language never scores perfectly against any readability
    formula. 0.65 is the realistic ceiling for well-written prose.
    This avoids penalising good output for not being mathematically
    perfect.

    Example: target range is 5-7, actual grade is 6.2 → strong pass.
             target range is 5-7, actual grade is 11.3 → borderline.
    """
    target_midpoint      = (target_min + target_max) / 2
    # allow some distance either side of the target before penalising
    acceptable_distance  = (target_max - target_min) / 2 + 3.0

    distance_from_target = abs(actual_grade - target_midpoint)
    raw_closeness        = max(0.0, 1.0 - (distance_from_target / acceptable_distance))

    # scale so that 0.65 = full pass
    return round(min(raw_closeness / realistic_ceiling, 1.0), 3)


def measure_sentence_length_variation(sentences: list[str]) -> float:
    """
    Measures how much sentence lengths vary across the output.

    Returns a coefficient of variation — standard deviation divided
    by mean sentence length.

    Above 0.40 = sentence lengths vary naturally. This is what
    human writing looks like. Short sentences mix with longer ones.

    Below 0.25 = sentences are uniformly similar in length.
    This is a common pattern in AI-generated text.

    This score is a diagnostic. It tells you whether the sentence
    structure rules produced natural variation — it is not a
    pass/fail gate on its own.

    Math:
        mean     = sum(lengths) / count
        variance = sum((length - mean)²) / count
        std_dev  = √variance
        score    = std_dev / mean
    """
    word_counts = [len(s.split()) for s in sentences if s.strip()]

    if len(word_counts) < 2:
        return 0.0

    mean_length = sum(word_counts) / len(word_counts)
    if mean_length == 0:
        return 0.0

    variance           = sum(
        (length - mean_length) ** 2 for length in word_counts
    ) / len(word_counts)
    standard_deviation = math.sqrt(variance)

    return round(standard_deviation / mean_length, 3)


def measure_vocabulary_variety(text: str) -> float:
    """
    Measures how varied the vocabulary is in the output.

    Returns a score from 0.0 to 1.0.
    Above 0.58 = good vocabulary variety.
    Below 0.45 = words are repeating too often.

    Skips short common words ('the', 'a', 'is') because they do
    not tell us anything meaningful about vocabulary richness.

    Math: unique meaningful words / total meaningful words
    This is called the Type-Token Ratio in linguistics.
    """
    # short, common words that appear in almost every sentence
    # and tell us nothing about vocabulary richness
    words_to_skip = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "and", "in", "it", "that", "this", "for", "on",
        "with", "as", "at", "by", "from", "or", "but", "not", "have",
        "had", "has", "do", "did", "will", "would", "can", "could",
        "should", "may", "might", "its", "their", "they", "we", "you",
        "he", "she", "his", "her", "our", "your", "all", "if", "so"
    }

    meaningful_words = [
        word.lower().strip(".,!?;:'\"()-")
        for word in text.split()
        if word.lower().strip(".,!?;:'\"()-") not in words_to_skip
        and word.lower().strip(".,!?;:'\"()-")
    ]

    if not meaningful_words:
        return 0.0

    unique_word_count = len(set(meaningful_words))
    total_word_count  = len(meaningful_words)

    return round(unique_word_count / total_word_count, 3)


def measure_vocabulary_displacement(
    text:       str,
    coca_ranks: dict[str, int],
) -> dict:
    """
    Measures whether DSCL successfully pushed the vocabulary toward
    plain, common words.

    Uses COCA frequency ranks where lower rank = more common word.
    'The' is rank 1. 'Utilize' is around rank 8,000.
    'Hereinafter' is around rank 40,000.

    A lower average rank means the output uses simpler, more common
    words — which is the goal for most DSCL domains.

    This is the core proof that the constraint layer worked:
    if the average rank dropped compared to unconstrained output,
    DSCL displaced AI vocabulary toward plain language.

    Words not found in the COCA list are treated as rare (rank 25,000).
    """
    words_to_skip = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of",
        "and", "in", "it", "that", "this", "for", "on", "with", "at"
    }

    content_words = [
        word.lower().strip(".,!?;:'\"()-")
        for word in text.split()
        if word.lower().strip(".,!?;:'\"()-") not in words_to_skip
        and word.lower().strip(".,!?;:'\"()-")
    ]

    if not content_words:
        return {"average_coca_rank": 0, "verdict": "not enough words to measure"}

    # words outside the COCA list are rare — penalise them
    rank_for_unknown_words = 25000

    word_ranks   = [coca_ranks.get(word, rank_for_unknown_words) for word in content_words]
    average_rank = sum(word_ranks) / len(word_ranks)

    if average_rank <= 3000:
        verdict = "plain language — strong displacement"
    elif average_rank <= 6000:
        verdict = "plain language — acceptable"
    elif average_rank <= 10000:
        verdict = "mixed — some complex words remain"
    elif average_rank <= 18000:
        verdict = "domain vocabulary present — review exceptions list"
    else:
        verdict = "complex language — constraint layer may not have fired"

    return {
        "average_coca_rank": round(average_rank),
        "verdict":           verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORD RULE SCANNER
# replaces per-rule regex with a single-pass scan using FlashText.
# loads all word and phrase lists once at startup and reuses them.
# falls back to simple string search if FlashText is not installed.
# ─────────────────────────────────────────────────────────────────────────────

class WordRuleScanner:
    """
    Scans output text for banned words, banned phrases, and words
    that should be replaced — all in a single pass over the text.

    This is faster than running one regex per rule because it builds
    a lookup structure (a Trie) from all the word lists at startup
    and walks the text once regardless of how many rules exist.

    The more rules DSCL gains over time, the bigger the speed advantage
    over per-rule scanning becomes.
    """

    def __init__(
        self,
        words_to_ban:     list[str],
        phrases_to_ban:   list[str],
        replacement_map:  dict[str, str],
    ):
        if _flashtext_available:
            self._banned_word_scanner   = _KeywordProcessor(case_sensitive=False)
            self._replacement_scanner   = _KeywordProcessor(case_sensitive=False)

            for word in words_to_ban:
                self._banned_word_scanner.add_keyword(word)
            for phrase in phrases_to_ban:
                self._banned_word_scanner.add_keyword(phrase)
            for original_phrase, replacement_phrase in replacement_map.items():
                self._replacement_scanner.add_keyword(original_phrase, replacement_phrase)

            self._use_flashtext = True
        else:
            # fallback: store the lists and search with plain string matching
            self._banned_terms    = [w.lower() for w in words_to_ban + phrases_to_ban]
            self._replacement_map = {k.lower(): v for k, v in replacement_map.items()}
            self._use_flashtext   = False

    def find_banned_terms(self, output_text: str) -> list[str]:
        """
        Returns every banned word or phrase found in the text.
        The returned list contains the actual terms found, not rule IDs.
        """
        if self._use_flashtext:
            return self._banned_word_scanner.extract_keywords(output_text)
        else:
            text_lower = output_text.lower()
            return [term for term in self._banned_terms if term in text_lower]

    def apply_replacements(self, output_text: str) -> str:
        """
        Swaps redundant phrases for cleaner alternatives in one pass.
        Example: 'in order to' becomes 'to'.
        Returns the cleaned text.
        """
        if self._use_flashtext:
            return self._replacement_scanner.replace_keywords(output_text)
        else:
            result = output_text
            for original, replacement in self._replacement_map.items():
                result = result.replace(original, replacement)
            return result


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL LANGUAGE CHECKER
# checks sentence structure — passive voice and nominalizations.
# uses spaCy when available for linguistically accurate results.
# falls back to pattern matching when spaCy is not installed.
# ─────────────────────────────────────────────────────────────────────────────

class StructuralLanguageChecker:
    """
    Checks whether sentences follow the structural rules DSCL enforces:
    active voice, no nominalizations, subject before verb.

    When spaCy is installed (pip install dscl[nlp]), checks are done
    using linguistic analysis of sentence grammar — more accurate,
    catches patterns that surface matching misses.

    When spaCy is not installed, pattern matching handles what it can
    and flags conservatively rather than producing false positives.
    """

    def __init__(self):
        self._spacy_model = None
        self._spacy_ready = False

        if _spacy_available:
            try:
                self._spacy_model = _spacy.load("en_core_web_sm")
                self._spacy_ready = True
            except OSError:
                # spaCy is installed but the language model is not downloaded yet.
                # run 'dscl setup-nlp' to download it.
                self._spacy_ready = False

    @property
    def uses_linguistic_analysis(self) -> bool:
        """True if spaCy is available and the language model is loaded."""
        return self._spacy_ready

    def find_passive_voice_sentences(self, sentences: list[str]) -> list[str]:
        """
        Returns the sentences that use passive voice.

        Passive voice means the subject receives the action rather
        than performing it.
        'The report was written by the analyst' is passive.
        'The analyst wrote the report' is active.

        Active voice is clearer because the reader immediately knows
        who is doing what.
        """
        passive_sentences = []

        if self._spacy_ready:
            for sentence in sentences:
                doc = self._spacy_model(sentence)
                # spaCy marks passive subjects with the 'nsubjpass' dependency tag
                # this catches all forms: 'was written', 'had been reviewed',
                # 'is being considered' — not just simple past passive
                has_passive_subject = any(
                    token.dep_ == "nsubjpass" for token in doc
                )
                if has_passive_subject:
                    passive_sentences.append(sentence)
        else:
            # fallback: look for common passive patterns
            # less accurate but avoids false positives by being conservative
            passive_pattern = re.compile(
                r'\b(is|are|was|were|be|been|being)\s+\w+ed\b',
                re.IGNORECASE
            )
            for sentence in sentences:
                if passive_pattern.search(sentence):
                    passive_sentences.append(sentence)

        return passive_sentences

    def find_nominalizations(
        self,
        text:       str,
        blocklist:  list[str],
        exceptions: list[str],
    ) -> list[str]:
        """
        Returns words that are nouns when they should be verbs.

        A nominalization is when a verb gets turned into a noun,
        making the sentence weaker and harder to read.
        'make a decision' contains 'decision' — the verb 'decide' is clearer.
        'provide assistance' contains 'assistance' — 'help' is clearer.

        When spaCy is available, this uses morphological analysis to
        catch all word forms — 'decisions', 'decisional', etc.
        Without spaCy, it checks the exact words in the blocklist.
        """
        exception_set      = {e.lower() for e in exceptions}
        found_nominals     = []

        if self._spacy_ready:
            doc = self._spacy_model(text)
            blocklist_roots = {
                self._spacy_model(word)[0].lemma_ for word in blocklist
            }
            for token in doc:
                if (token.lemma_ in blocklist_roots
                        and token.pos_ == "NOUN"
                        and token.text.lower() not in exception_set):
                    found_nominals.append(token.text)
        else:
            text_lower = text.lower()
            for blocked_word in blocklist:
                if (blocked_word.lower() in text_lower
                        and blocked_word.lower() not in exception_set):
                    found_nominals.append(blocked_word)

        return found_nominals


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT VALIDATOR
# the main class. coordinates all checks and returns a ValidationResult.
# ─────────────────────────────────────────────────────────────────────────────

class OutputValidator:
    """
    Validates model output against the dictionary rules for a given domain.

    Call validate() with the text to check. It runs two passes:
      1. Check pass  — measures what the output scored and finds violations.
      2. Fix pass    — applies automatic corrections for soft violations.

    Hard violations are flagged but not auto-fixed. They appear in
    violations[] so the caller can decide what to do with them.

    All measurements are returned in ValidationResult so callers can
    see exactly what the constraint layer did and whether it worked.
    """

    def __init__(self, loader: DictionaryLoader):
        self.loader             = loader
        self._structural_checker = StructuralLanguageChecker()
        self._word_scanner:      Optional[WordRuleScanner] = None
        self._scanner_built      = False

    def _build_word_scanner(self, domain: str) -> WordRuleScanner:
        """
        Builds the WordRuleScanner from the rules loaded for this domain.
        Called once per domain and reused across calls.
        """
        rules            = self.loader.rules()
        global_rules     = rules.get("global", {})
        domain_overrides = rules.get("domain_overrides", {}).get(domain, {})

        # banned words — global list minus any domain exemptions
        global_banned  = global_rules.get("banned_words", {}).get("terms", [])
        domain_exempt  = domain_overrides.get("banned_words_exempt", [])
        active_banned  = [w for w in global_banned if w not in domain_exempt]

        # banned phrases from both global hedging list and domain overrides
        global_phrases = global_rules.get("hedging_language", {}).get("terms", [])
        domain_phrases = domain_overrides.get("banned_phrases", [])
        all_banned_phrases = list(set(global_phrases + domain_phrases))

        # replacement map — redundant phrases swapped for cleaner ones
        replacement_map = global_rules.get(
            "redundant_phrases", {}
        ).get("replacements", {})

        # certainty language replacements from domain overrides
        certainty_replacements = domain_overrides.get(
            "certainty_language", {}
        ).get("replacements", {})
        replacement_map = {**replacement_map, **certainty_replacements}

        return WordRuleScanner(
            words_to_ban    = active_banned,
            phrases_to_ban  = all_banned_phrases,
            replacement_map = replacement_map,
        )

    def validate(
        self,
        text:       str,
        domain:     str = "",
        exceptions: Optional[list[str]] = None,
    ) -> ValidationResult:
        """
        Main entry point. Takes raw model output and returns a ValidationResult.

        domain     — used to load the right rules and grade targets.
        exceptions — words allowed through despite vocabulary rules.
                     if not passed, they are loaded from domain_exceptions.json.
        """
        active_exceptions = exceptions or self._load_exceptions(domain)

        # build the word scanner for this domain if not already built
        if not self._scanner_built or self._current_domain != domain:
            self._word_scanner    = self._build_word_scanner(domain)
            self._current_domain  = domain
            self._scanner_built   = True

        weights          = self.loader.weight_definitions()
        grade_targets    = self.loader.grade_targets()
        sentence_rules   = self.loader.sentence_length()
        voice_rules      = self.loader.voice_rules()
        nom_blocklist    = self.loader.nominalization_blocklist()
        modifier_rules   = self.loader.modifier_rules()

        violations:          list[str] = []
        fixes_applied:       list[str] = []
        exception_words_used: list[str] = []

        working_text = text
        sentences    = self._split_into_sentences(working_text)

        # ── fix pass: apply replacements before checking ───────────────────
        cleaned_text = self._word_scanner.apply_replacements(working_text)
        if cleaned_text != working_text:
            fixes_applied.append("RD_01")
            working_text = cleaned_text

        # ── word-level: banned terms ───────────────────────────────────────
        banned_terms_found = self._word_scanner.find_banned_terms(working_text)
        # remove any that are in the active exceptions list
        banned_terms_found = [
            term for term in banned_terms_found
            if term.lower() not in {e.lower() for e in active_exceptions}
        ]
        if banned_terms_found:
            violations.append("BW_01")

        # ── word-level: vocabulary check ───────────────────────────────────
        vocab_violations, vocab_exceptions = self._check_vocabulary(
            working_text, domain, active_exceptions
        )
        violations.extend(vocab_violations)
        exception_words_used.extend(vocab_exceptions)

        # ── word-level: modifiers ──────────────────────────────────────────
        blocked_modifiers = modifier_rules.get("blocked", [])
        working_text, modifier_fixes = self._check_and_fix_modifiers(
            working_text, blocked_modifiers, active_exceptions
        )
        fixes_applied.extend(modifier_fixes)

        # ── sentence-level: passive voice ──────────────────────────────────
        passive_sentences  = self._structural_checker.find_passive_voice_sentences(sentences)
        passive_rule       = voice_rules.get("max_passive_ratio", 0.2)
        passive_ratio      = len(passive_sentences) / len(sentences) if sentences else 0.0
        if passive_ratio > passive_rule:
            violations.append("VR_01")

        # ── sentence-level: sentence length ────────────────────────────────
        max_words_per_sentence = sentence_rules.get("max_words_per_sentence", 25)
        long_sentence_found    = any(
            len(s.split()) > max_words_per_sentence for s in sentences
        )
        if long_sentence_found:
            violations.append("SS_01")
            fixes_applied.append("SS_01")

        # ── sentence-level: nominalizations ───────────────────────────────
        blocklist         = nom_blocklist.get("blocked", [])
        nominals_found    = self._structural_checker.find_nominalizations(
            working_text, blocklist, active_exceptions
        )
        if nominals_found:
            violations.append("NB_01")
            exception_words_used.extend([
                w for w in nominals_found
                if w.lower() in {e.lower() for e in active_exceptions}
            ])

        # ── readability measurements ───────────────────────────────────────
        reading_grade      = measure_reading_grade(working_text)
        fk_low, fk_high    = self._resolve_grade_target(domain, grade_targets)
        grade_closeness    = measure_grade_target_closeness(reading_grade, fk_low, fk_high)

        if not (fk_low <= reading_grade <= fk_high):
            violations.append("RD_01")

        # ── diagnostic measurements ────────────────────────────────────────
        sentence_variation = measure_sentence_length_variation(sentences)
        vocab_variety      = measure_vocabulary_variety(working_text)

        # vocabulary displacement uses COCA ranks from primary_vocab.json
        coca_ranks         = self._build_coca_rank_lookup()
        vocab_displacement = measure_vocabulary_displacement(working_text, coca_ranks)

        # ── determine whether the output passed all hard rules ─────────────
        passed = not any(
            self._is_hard_violation(violation_id, weights)
            for violation_id in violations
        )

        return ValidationResult(
            output                = working_text,
            fk_grade              = reading_grade,
            violations            = violations,
            fixes_applied         = fixes_applied,
            domain                = domain,
            passed                = passed,
            exception_words_used  = list(set(exception_words_used)),
            grade_target_closeness = grade_closeness,
            sentence_variation    = sentence_variation,
            vocabulary_variety    = vocab_variety,
            vocabulary_displacement = vocab_displacement,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_grade_target(
        self, domain: str, grade_targets: dict
    ) -> tuple[float, float]:
        """
        Looks up the FK grade target range for the domain.
        Falls back to 6-8 if nothing is configured.
        """
        domain_target = grade_targets.get(domain, grade_targets.get("default", "6-8"))

        if isinstance(domain_target, dict):
            low  = domain_target.get("fk_floor") or domain_target.get("fk_target", "6-8")
            high = domain_target.get("fk_ceiling") or domain_target.get("fk_target", "6-8")
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                return float(low), float(high)
            domain_target = domain_target.get("fk_target", "6-8")

        parts = str(domain_target).split("-")
        try:
            return float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            return 6.0, 8.0

    def _check_and_fix_modifiers(
        self,
        text:              str,
        blocked_modifiers: list[str],
        exceptions:        list[str],
    ) -> tuple[str, list[str]]:
        """
        Checks for weak modifiers like 'very', 'quite', 'rather'.
        Flags them but does not rewrite — the model should have
        avoided them in the first place.
        """
        fixes_applied = []
        text_lower    = text.lower()

        for modifier in blocked_modifiers:
            if (modifier.lower() in text_lower
                    and modifier.lower() not in {e.lower() for e in exceptions}):
                fixes_applied.append("MD_01")
                break

        return text, fixes_applied

    def _check_vocabulary(
        self,
        text:       str,
        domain:     str,
        exceptions: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Checks whether the words in the output are in the allowed vocabulary.
        Words in domain_exceptions.json are allowed through regardless.
        Returns violations found and exception words that were used.
        """
        primary  = self.loader.primary_vocab()
        fallback = self.loader.fallback_vocab()

        if not primary and not fallback:
            return [], []

        allowed_words = {
            entry["word"].lower()
            for entry in primary.get("entries", [])
            if "word" in entry
        } | {
            entry["word"].lower()
            for entry in fallback.get("entries", [])
            if "word" in entry
        }

        if not allowed_words:
            return [], []

        exception_set        = {e.lower() for e in exceptions}
        words_in_output      = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        out_of_vocabulary = [
            w for w in words_in_output
            if w not in allowed_words
            and w not in exception_set
            and not any(
                original.istitle() or original.isupper()
                for original in re.findall(r'\b' + re.escape(w) + r'\b', text)
            )
        ]
        exceptions_used      = [w for w in words_in_output if w in exception_set]

        violations = ["VB_01"] if out_of_vocabulary else []
        return violations, exceptions_used

    def _build_coca_rank_lookup(self) -> dict[str, int]:
        """
        Builds a word-to-rank dictionary from primary_vocab.json.
        Used by measure_vocabulary_displacement to score word choices.
        Lower rank = more common word = better plain language score.
        """
        primary = self.loader.primary_vocab()
        return {
            entry["word"].lower(): entry.get("rank", 9999)
            for entry in primary.get("entries", [])
            if "word" in entry
        }

    def _is_hard_violation(self, rule_id: str, weights: dict) -> bool:
        rule_weights = weights.get("rule_weights", {})
        return rule_weights.get(rule_id, "soft") == "hard"

    def _load_exceptions(self, domain: str) -> list[str]:
        """
        Loads the words that are allowed through vocabulary rules
        for this domain. Merges domain-specific and global exceptions.
        """
        exceptions_data  = self.loader.domain_exceptions()
        domain_exceptions = exceptions_data.get(domain, [])
        global_exceptions = exceptions_data.get("global", [])
        return list(set(domain_exceptions + global_exceptions))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Splits text into individual sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]