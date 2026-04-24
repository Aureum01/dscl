# DSCL — Dictionary-Scoped Composition Layer
## HOW TO USE THIS DICTIONARY

---

### What this is

A model-native linguistic constraint layer for LLM output.

The narrative is the input. DSCL governs how it gets represented — at the word, sentence, paragraph, and page level — using frequency-grounded vocabulary, plain language standards, and structured composition rules.

---

### What this is not

**Not a prompt.**
The `/dictionary` is not a system prompt. It is not a set of instructions you paste before your request. It is a structured rule system the LLM loads and applies at generation time before a single word of output is produced.

**Not a training set.**
The `/dictionary` does not fine-tune anything. It does not modify model weights. It operates at inference time — on every output, for every domain, every time.

**Not a style guide.**
A style guide gives preferences. The `/dictionary` gives constraints. Hard rules cannot be violated. Soft rules hold unless `conflict_resolution.json` permits an exception. The difference matters.

**Not a content layer.**
The `/dictionary` controls how meaning is expressed in language. It does not control what the meaning is. Facts, claims, accuracy, tone beyond register, formatting, and language other than English are all outside scope. See `/system/scope.json`.

---

### The core principle

**The LLM does not invent meaning. The user already has it.**

The user's narrative — what happened, what the risk is, what to do — is the input. The `/dictionary` is the path the LLM takes to represent that meaning correctly at every level of writing:

```
NARRATIVE (input — user's meaning)
    ↓
WORDS       — which word carries this meaning
    ↓
SENTENCES   — how words build a complete thought
    ↓
PARAGRAPHS  — how sentences develop one idea
    ↓
PAGE        — how paragraphs form something a person can read and act on
```

Every rule in every file serves that path. Nothing else.

---

### The four levels

**Level 1 — Word**
The vocabulary layer selects the right word for each unit of meaning. Primary vocabulary first. Fallback only when no primary word carries the precise meaning. Domain exceptions only when the domain tag matches and no simpler word exists. The register ceiling set by the domain blocks any word above it — including all AI signature words, which are always blocked regardless of domain.

Relevant files:
- `/vocabulary/primary_vocab.json`
- `/vocabulary/fallback_vocab.json`
- `/vocabulary/domain_exceptions.json`
- `/context/register_tags.json`
- `/context/domain_register.json`

**Level 2 — Sentence**
The grammar and syntax layers govern how words form sentences. Subject leads. Verb follows within 5 words. Active voice default. No nominalizations. No consecutive passive constructions. Main point in the main clause. Sentence length varied — short follows long. Hard ceiling of 35 words per sentence.

Relevant files:
- `/grammar/voice_rules.json`
- `/grammar/verb_placement.json`
- `/grammar/modifier_rules.json`
- `/grammar/nominalization_blocklist.json`
- `/syntax/sentence_structure.json`
- `/syntax/clause_rules.json`
- `/syntax/punctuation_function.json`

**Level 3 — Paragraph**
The paragraph layer governs how sentences build a complete unit of thought. Topic sentence leads. One idea per paragraph. New idea means new paragraph. Crisis output: 2–3 sentences. General output: 3–6 sentences. Final sentence emphasizes the topic or states a consequence.

Relevant files:
- `/syntax/paragraph_structure.json`

**Level 4 — Page**
The readability layer governs the FK grade of the complete output. FK grade is a composition constraint — not a scoring tool applied after the fact. The LLM must compose within the FK target from the first word. Syllable load (secondary FK driver) and sentence length (primary FK driver) are checked throughout generation, not at the end.

Relevant files:
- `/readability/grade_targets.json`
- `/readability/sentence_length.json`
- `/readability/syllable_load.json`

---

### The pipeline — order of operations

The LLM must follow this sequence exactly. No step may be skipped. No step may run before the one that precedes it. Output is not assembled until step 14.

```
 1.  load  /system/pipeline.json              ← you are here
 2.  load  /system/entry_rules.json           ← establish narrative, domain, audience
 3.  load  /context/domain_register.json      ← retrieve FK target + register ceiling
 4.  load  /context/register_tags.json        ← map register levels, block ai_signature
 5.  load  /vocabulary/primary_vocab.json     ← attempt primary word first
           /vocabulary/fallback_vocab.json    ← fallback if no primary word fits
           /vocabulary/domain_exceptions.json ← exception only if domain tag matches
 6.  check /readability/syllable_load.json    ← secondary FK driver
 7.  check /readability/grade_targets.json    ← composition constraint, not post-hoc score
 8.  apply /grammar/voice_rules.json          ← active voice default
           /grammar/verb_placement.json       ← verb within 5 words of subject
           /grammar/modifier_rules.json       ← one adjective per noun, no intensifiers
           /grammar/nominalization_blocklist.json ← replace verb-noun with direct verb
 9.  apply /syntax/sentence_structure.json    ← subject leads, main point in main clause
           /syntax/clause_rules.json          ← intro clause ≤6 words, tense consistency
           /syntax/punctuation_function.json  ← punctuation is structural, not decorative
10.  apply /syntax/paragraph_structure.json   ← topic sentence, one idea, correct length
11.  check /readability/sentence_length.json  ← average and hard ceiling enforced
12.  resolve /meta/conflict_resolution.json   ← five-level hierarchy, tiebreaker: shorter
13.  apply /meta/weight_definitions.json      ← hard rules absolute, soft rules logged
14.  assemble page
```

If any step returns no result, consult `/system/defaults.json` for the fallback value.

---

### For developers and model integrators

**System prompt integration**
Load the contents of `pipeline.json` and `entry_rules.json` into your system prompt. The LLM will sequence through the remaining files automatically once it understands the pipeline order and the three entry facts it must establish before generating.

**Domain override**
If you know the domain upfront, pass it directly rather than relying on auto-detection. Use the `domain` field from `domain_register.json`. This bypasses detection and is more reliable for production use.

```python
constrained = dscl.prepare(narrative, domain="legal_contract", audience="general_public")
```

**Validation**
Use `dscl.validate()` on any existing output to check it against DSCL rules without going through the full generation pipeline. This is useful for auditing output from other sources or checking output that was generated before DSCL was in place.

**Weight interpretation**
Every rule file uses one of three weights defined in `/meta/weight_definitions.json`:

- `hard` — absolute constraint. Rewrite until compliant. No exceptions except `domain_exceptions.json`.
- `soft` — default constraint. Follow unless `conflict_resolution.json` permits an exception.
- `contextual` — domain-dependent. Only applies when the domain tag activates it.

**Modifying the dictionary**
All files are plain JSON and can be edited directly in any text editor. Changes take effect immediately on the next pipeline run — there is no compilation step, no cache to clear, nothing to restart. If you add a domain to `domain_register.json`, add its register ceiling to `register_tags.json` and its FK target to `grade_targets.json` at the same time. The three files must stay consistent.

---

### What the output looks like

When the pipeline runs correctly, the output has these properties:

- No AI signature words (`robust`, `leverage`, `delve`, `holistic`, `nuanced`, `comprehensive`, etc.)
- Active voice throughout except where passive is structurally justified
- Subject leads every sentence
- Verb follows subject within 5 words
- One idea per paragraph
- Topic sentence leads every paragraph
- FK grade within the domain target
- No nominalizations (`make a decision` → `decide`, `provide assistance` → `help`)
- No intensifiers (`very`, `extremely`, `highly`, etc.)
- Sentence length varied — short follows long
- Punctuation structural, not decorative

The output reads like a person wrote it. That is the point.

---

### Source attribution

This architecture is original. The rules it encodes are grounded in:

- Corpus of Contemporary American English (COCA) — wordfrequency.info
- Google Trillion Word Corpus — github.com/first20hours/google-10000-english
- BNC/COCA Word Family Lists — eapfoundation.com
- WordNet — wordnet.princeton.edu
- Strunk & White, Elements of Style — gutenberg.org/ebooks/37134 (public domain)
- Purdue OWL — owl.purdue.edu
- Plain Language Act 2010 — plainlanguage.gov
- NARA 10 Principles of Plain Language — archives.gov
- Flesch-Kincaid Grade Level Formula — public domain
- Williams, J. — Style: Lessons in Clarity and Grace

Every rule file carries its own `_citation` block with direct source URLs.

---

*DSCL — Dictionary-Scoped Composition Layer*
*The narrative is the input. The dictionary is the path.*