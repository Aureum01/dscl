# models.py
# the two data structures DSCL passes around.
# ConstrainedContext is what prepare() hands back.
# ValidationResult is what validate() hands back.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConstrainedContext:
    """
    Everything dscl.prepare() returns.

    Contains the system prompt with all constraints injected,
    the wrapped narrative prompt, and the metadata about what
    domain, grade target, and template were detected.

    Pass system_prompt and prompt directly to your model call.
    Everything else is for your records and logging.
    """

    # the narrative the user originally passed in
    narrative: str

    # detected or manually set domain
    # examples: "travel_risk", "legal_brief", "corporate_security_intel"
    domain: str

    # target reading grade range as a string
    # examples: "5-7", "8-10", "12-14"
    fk_target: str

    # register ceiling detected for this domain
    # examples: "abstract_general", "abstract_technical"
    register: str

    # the system prompt with all DSCL rules injected
    # pass this directly to your model as the system message
    system_prompt: str

    # the user prompt, ready to send to the model
    prompt: str

    # names of the rule files that were loaded for this context
    rules_loaded: list[str] = field(default_factory=list)

    # words allowed through despite vocabulary rules
    # these come from domain_exceptions.json
    active_exceptions: list[str] = field(default_factory=list)

    # the report structure template matched for this domain
    # empty dict means no template is defined for this domain
    template: dict = field(default_factory=dict)

    # True when prepare() was called with only_text=True
    # means the system prompt instructs the model to write plain prose only
    # no markdown headers, bullets, bold, tables, or numbered lists
    only_text: bool = False

    # True when prepare() was called with live=True
    # means the system prompt instructs the model to use current information
    # and not rely on training data for time-sensitive facts
    live_mode: bool = False


@dataclass
class ValidationResult:
    """
    Everything dscl.validate() returns.

    Contains the cleaned output text, every measurement that shows
    whether the constraint layer worked, and a full audit trail of
    what was flagged and what was fixed.

    The measurements are diagnostics — they tell you whether the
    four constraint levels (word, sentence, paragraph, page) fired
    correctly. They are not grades to optimise for their own sake.
    """

    # the output text after validation and any automatic fixes
    output: str

    # reading grade the output scored on the Flesch-Kincaid formula
    # lower = easier to read
    # target range depends on the domain
    fk_grade: float

    # rule IDs that were violated
    # empty list means the output passed all checks cleanly
    violations: list[str] = field(default_factory=list)

    # rule IDs where automatic fixes were applied
    fixes_applied: list[str] = field(default_factory=list)

    # domain that was active during validation
    domain: str = ""

    # True if no hard rule violations were found
    # hard violations are defined in weight_definitions.json
    passed: bool = True

    # words that were allowed through via domain_exceptions.json
    exception_words_used: list[str] = field(default_factory=list)

    # how close the FK grade is to its domain target range
    # 0.0 = far off target. 1.0 = on target.
    # above 0.65 counts as a pass — natural language never scores perfectly
    grade_target_closeness: float = 0.0

    # how much sentence length varies across the output
    # above 0.40 = natural human-like variation
    # below 0.25 = uniform length, a common pattern in AI output
    # this is a diagnostic — it shows whether sentence rules worked
    sentence_variation: float = 0.0

    # how varied the vocabulary is
    # above 0.58 = good variety, words do not repeat too often
    # below 0.45 = repetitive vocabulary
    vocabulary_variety: float = 0.0

    # how close the vocabulary is to plain, common language
    # measured using COCA frequency ranks
    # lower average rank = simpler words = constraint layer worked
    # format: {"average_coca_rank": int, "verdict": str}
    vocabulary_displacement: dict = field(default_factory=dict)