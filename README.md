# DSCL — Dictionary-Scoped Composition Layer

A model-native linguistic constraint layer for LLM output.

The narrative is the input. The meaning already exists.  
DSCL governs how that meaning gets represented — at the word, sentence, paragraph, and page level — using frequency-grounded vocabulary, plain language standards, and structured composition rules.

> *As each word carries weight, every syllable carries tone, every sentence carries context, while every paragraph represents narrative.*

---

## The problem

LLMs default to a recognisable pattern: passive voice, long intro clauses, hedged language, abstract words, buried main points. The same five adjectives appear in every document. The reading grade sits at 12 when the domain needs 6. The output sounds like a machine wrote it because a machine did — and nothing constrained how it wrote.

DSCL intercepts before generation and constrains how meaning gets represented. Not what the meaning is — that is always the user's. Only how it is written.

```
Narrative → Words → Sentences → Paragraphs → Page
```

---

## How it works

At each level, a different set of rules fires:

| Level | What it controls | Source files |
|-------|-----------------|--------------|
| **Word** | Which words are permitted. Simple word first — always. Frequency-grounded against COCA data. | `primary_vocab.json`, `domain_exceptions.json`, `register_tags.json` |
| **Sentence** | How words form sentences. Subject leads. Actor before action. Main point first. | `sentence_structure.json`, `clause_rules.json`, `voice_rules.json` |
| **Paragraph** | How sentences build paragraphs. One idea per block. Topic sentence leads. No drift. | `paragraph_structure.json`, `sentence_length.json` |
| **Page** | How paragraphs assemble into a finished page a person can read and act on. | `domain_register.json`, `grade_targets.json`, `conflict_resolution.json` |

The LLM does not invent meaning. The user already has it. DSCL constrains the representation — not the content.

---

## Install

```bash
pip install dscl
# or
uv add dscl
```

For more accurate structural checking (passive voice, nominalization detection):

```bash
pip install "dscl[nlp]"
dscl setup-nlp
```

---

## Quick start

```python
from dscl import DSCL

dscl = DSCL()
constrained = dscl.prepare("Write a travel risk assessment for Beirut.")

print(constrained.domain)      # "travel_risk"
print(constrained.fk_target)   # "5-7"
print(constrained.register)    # "abstract_general"
```

Pass the constrained context directly to any model:

```python
# Anthropic
import anthropic
from dscl import DSCL

client = anthropic.Anthropic()
dscl   = DSCL()

constrained = dscl.prepare("Write a travel risk assessment for Beirut.")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=constrained.system_prompt,
    messages=[{"role": "user", "content": constrained.prompt}]
)

result = dscl.validate(response.content[0].text, domain=constrained.domain)

print(result.output)
print(result.fk_grade)            # 5.8
print(result.violations)          # [] — clean pass
print(result.fixes_applied)       # ["SS_01", "VR_01"]
print(result.sentence_variation)  # 0.43 — natural variation
print(result.vocabulary_variety)  # 0.89 — good range
```

```python
# OpenAI
from openai import OpenAI
from dscl import DSCL

client = OpenAI()
dscl   = DSCL()

constrained = dscl.prepare("Summarize the risks of this contract.")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": constrained.system_prompt},
        {"role": "user",   "content": constrained.prompt}
    ]
)

result = dscl.validate(response.choices[0].message.content, domain=constrained.domain)
print(result.output)
```

```python
# Ollama (local models)
import ollama
from dscl import DSCL

dscl = DSCL()

constrained = dscl.prepare("Explain this medical procedure to a patient.")

# recommended inference options for Qwen3 8b non-thinking mode
# adjust per model — see your model's documentation for optimal settings
response = ollama.chat(
    model="qwen3:8b",
    options={
        "temperature":    0.7,
        "top_p":          0.85,
        "top_k":          40,
        "min_p":          0.0,
        "repeat_penalty": 1.5,
        "num_predict":    2048,
    },
    messages=[
        {"role": "system", "content": constrained.system_prompt},
        {"role": "user",   "content": constrained.prompt}
    ]
)

result = dscl.validate(response["message"]["content"], domain=constrained.domain)
print(result.output)
```

---

## Flags

### `only_text` — plain prose output

Instructs the model to write in paragraphs only. No markdown headers, bullet points, bold text, tables, or numbered lists. Use when your pipeline or app renders plain text.

```python
constrained = dscl.prepare(narrative, only_text=True)
```

### `live` — current information

Signals that the model has web access and should use real-time information for time-sensitive topics. Prevents the model from confidently stating stale training data as current fact. For models without web access, instructs the model to flag uncertainty and direct the reader to current sources.

```python
constrained = dscl.prepare(narrative, live=True)
```

### Combined

```python
constrained = dscl.prepare(narrative, only_text=True, live=True)
```

The combination for live-feed product pipelines — current information rendered as clean plain prose.

---

## Test script flags

`test_dscl.py` ships with the repository and covers the full diagnostic workflow.

```bash
python test_dscl.py
python test_dscl.py --narrative "Write a travel advisory for Dubai."
python test_dscl.py --only-text
python test_dscl.py --live
python test_dscl.py --compare
python test_dscl.py --live --only-text --compare
```

| Flag | What it does |
|------|-------------|
| `--narrative "..."` | The narrative to process. Defaults to a software licence summary if omitted. |
| `--only-text` | Output plain prose only. No markdown, bullets, headers, or tables. |
| `--live` | Injects a currency instruction. Use with models that have web access. |
| `--compare` | Runs the narrative twice — once with DSCL, once without — and prints a side-by-side comparison of FK grade, sentence variation, vocabulary displacement, violations, and word count. |
| `--live --only-text` | Current information rendered as plain prose. The combination for live-feed pipelines. |
| `--live --only-text --compare` | Full diagnostic run. Constrained vs baseline with live signal and plain prose output. |

---

## Template commands

Prefix your narrative with `@template_name` to use a specific report template:

```python
narrative = "@travel_risk Write a country risk assessment for Dubai..."
narrative = "@corporate_security_intel India-Pakistan water treaty dispute..."
```

Any key in `dictionary/vocabulary/templates.json` is automatically a valid command. Add a template to the JSON and it is available immediately — no code changes needed.

---

## Override domain manually

```python
constrained = dscl.prepare(
    "Draft a liability clause for a SaaS agreement.",
    domain="legal_contract",
    audience="general_public"
)
```

---

## Validate existing output

```python
from dscl import DSCL

dscl   = DSCL()
result = dscl.validate(existing_text, domain="travel_risk")

print(result.fk_grade)                # Flesch-Kincaid grade
print(result.grade_target_closeness)  # above 0.65 = pass
print(result.sentence_variation)      # above 0.40 = natural variation
print(result.vocabulary_variety)      # above 0.58 = good vocabulary range
print(result.vocabulary_displacement) # avg COCA rank — lower = plainer language
print(result.violations)              # rule violations found
print(result.fixes_applied)           # rule IDs applied automatically
print(result.passed)                  # True if no hard violations
```

---

## What the validator measures

```
Word-level checks
  Banned words found         : 0
  Replacements applied       : 2
  Vocabulary displacement    : rank 2,840 — plain language, strong

Sentence-level checks
  Passive voice instances    : 0
  Nominalizations found      : 0
  Sentences checked          : 12

Readability (reference)
  FK grade                   : 6.9    target 6–8
  Grade target closeness     : 1.0    above 0.65 threshold — pass
  Sentence variation         : 0.429  above 0.40 — natural variation
  Vocabulary variety         : 0.890  above 0.58 — good range

Result
  Passed                     : True
  Fixes applied              : ['RD_01']
  Violations                 : []
```

**On the FK grade:** it is a reference number, not the primary target. 65% compliance with the target range is the realistic ceiling for natural language. A score above that threshold counts as a full pass. Perfect scores do not exist in real prose.

---

## Real output example

**Input narrative** — user-supplied, current events:

> USS Spruance fired on and seized Iranian cargo vessel MV Touska on April 19 in the Arabian Sea after a six-hour standoff. Iran closed the Strait of Hormuz to vessels bound for US and allied ports. Ship transits dropped 95% from 130 per day in February to 6 in March. Maersk, MSC, Hapag-Lloyd, and CMA CGM have suspended transits. QatarEnergy declared force majeure March 4 removing 20% of global LNG supply overnight. The IEA called this the largest supply disruption in oil market history. Brent crude is above 95 dollars. UNCTAD projects global trade growth falling from 4.7% in 2025 to 1.5–2.5% in 2026.

**DSCL output** (`only_text=True`):

> On April 19, the USS Spruance fired on and seized the Iranian cargo vessel MV Touska in the Arabian Sea. The standoff lasted six hours before US forces disabled the ship's engines and took it into custody. This action occurred under a US naval blockade of Iranian ports that began on April 13.
>
> Iran responded by closing the Strait of Hormuz to ships bound for US and allied ports. Iran has also threatened direct retaliation. A ceasefire between the US and Iran expires today, April 21.
>
> Ship traffic through the strait has collapsed. Daily transits fell from 130 in February to just 6 in March, a drop of 95%. Major shipping lines including Maersk, MSC, Hapag-Lloyd, and CMA CGM have all suspended transits.
>
> Energy markets have been hit hard. QatarEnergy declared force majeure on March 4, removing 20% of global LNG supply at once. Brent crude has risen above 95 dollars per barrel. The IEA called this the largest supply disruption in oil market history.
>
> The disruption reaches beyond energy. One third of global fertilizer trade and 85% of Middle East polyethylene exports move through this route. UNCTAD projects global trade growth falling from 4.7% in 2025 to between 1.5% and 2.5% in 2026.

**Validation result:**

```
FK grade               : 6.9    target 6–8     ✓
Grade target closeness : 1.0                   ✓ pass
Sentence variation     : 0.429                 ✓ natural
Vocabulary variety     : 0.890                 ✓ good
Passed                 : True
```

The meaning carried from the user's narrative is unchanged. Only the representation was constrained.

---

## The dictionary

```
dictionary/
├── system/
│   ├── pipeline.json           master load sequence
│   ├── entry_rules.json        three entry facts before anything else
│   ├── scope.json              what DSCL does not control
│   └── defaults.json           fallback values
├── vocabulary/
│   ├── primary_vocab.json      COCA top-5k frequency words
│   ├── fallback_vocab.json     COCA 5k–10k
│   ├── domain_vocab.json       preferred words per domain
│   ├── domain_exceptions.json  words allowed despite vocabulary rules
│   └── templates.json          report structure templates
├── context/
│   ├── domain_register.json    domain → register and FK target mapping
│   ├── register_tags.json      register definitions
│   └── rules.json              global and domain-scoped language rules
├── syntax/
│   ├── sentence_structure.json
│   ├── clause_rules.json
│   ├── paragraph_structure.json
│   └── punctuation_function.json
├── grammar/
│   ├── voice_rules.json
│   ├── verb_placement.json
│   ├── modifier_rules.json
│   └── nominalization_blocklist.json
├── readability/
│   ├── grade_targets.json
│   ├── sentence_length.json
│   └── syllable_load.json
└── meta/
    ├── conflict_resolution.json
    └── weight_definitions.json
```

The dictionary is the core of DSCL. The library code is the interface to it. Every word added, every domain mapped, every template built compounds over time.

---

## Adding custom templates

Copy `_template_blank` from `dictionary/vocabulary/templates.json`, give it a new key, fill in the structure. It becomes available immediately as an `@command` — no code changes needed.

```json
"my_report": {
  "description": "...",
  "composition_order": ["section_1", "section_2"],
  "render_order": ["section_1", "section_2"],
  "sections": { ... }
}
```

Then use it:

```python
constrained = dscl.prepare("@my_report ...")
```

Custom templates go in `dictionary/custom/templates/` and are never overwritten by library updates.

---

## NLP accuracy

| Check | Without `[nlp]` | With `[nlp]` |
|-------|----------------|-------------|
| Passive voice | Pattern matching — catches common forms | Dependency parsing — catches all forms |
| Nominalizations | Exact word matching | Morphological analysis — catches all inflected forms |
| Word scanning | FlashText single pass | FlashText single pass |
| FK grade | Vowel cluster syllable algorithm | Same algorithm |

The core library works without `[nlp]`. Install it when output accuracy matters more than install size.

---

## Grounded sources

The dictionary is built on established linguistic sources:

- Corpus of Contemporary American English (COCA) — frequency data
- Flesch-Kincaid Grade Level formula — readability standard
- Plain Language Act 2010
- Strunk & White, *The Elements of Style*
- AP Stylebook
- WordNet
- Penn Treebank
- BNC/COCA word family lists
- Google Trillion Word Corpus

---

## License

MIT