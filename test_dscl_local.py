import ollama
from dscl import DSCL

# ── narrative built from one topic sentence per source ────────────────────────
# 1. Wikipedia      — strait status and cause
# 2. Lloyd's List   — shipping collapse figures
# 3. CNBC           — major carriers suspended
# 4. UNCTAD         — trade and economic impact
# 5. UN News        — humanitarian and supply chain spillover

narrative = """
Shipping traffic through the Strait of Hormuz has been largely blocked since
28 February 2026 after the United States and Israel launched strikes on Iran
and Iran retaliated by closing the strait to vessels bound for US and allied
ports. Transits of all vessel types fell 81% in the first week, with just one
crude oil tanker recorded passing through on 1 March and no LNG carriers at
all. Maersk, MSC, Hapag-Lloyd, and CMA CGM have all suspended operations
through the strait and rerouted vessels around the Cape of Good Hope. Ship
transits dropped from around 130 per day in February to just 6 in March — a
collapse of 95% — with UNCTAD projecting global merchandise trade growth
slowing from 4.7% in 2025 to between 1.5% and 2.5% in 2026. Nearly 20,000
seafarers remain stranded, fertilizer shipments are disrupted ahead of
planting season, and UN agencies warn that a temporary shock risks becoming a
prolonged humanitarian crisis across Asia and the Pacific.

Summarise the current situation in the Strait of Hormuz as a travel risk
assessment for maritime and logistics professionals operating in the region.
"""

# ── Qwen3 8b inference parameters (source: Qwen official HuggingFace docs) ───
# Qwen3 8b runs in non-thinking mode by default via Ollama.
# Official recommended settings for non-thinking / instruct mode:
#
#   temperature   : 0.7   — output randomness. 0.7 is the Qwen3 instruct default.
#                           do not use 0 (greedy decoding) — Qwen explicitly warns
#                           this causes degradation and endless repetition.
#
#   top_p         : 0.8   — nucleus sampling. model samples only from tokens whose
#                           cumulative probability reaches 0.8. tightens coherence.
#
#   top_k         : 20    — limits the sampling pool to the top 20 tokens per step.
#                           reduces incoherence without over-constraining output.
#
#   min_p         : 0.0   — minimum token probability floor. 0.0 disables it.
#
#   repeat_penalty: 1.5   — penalises recently used tokens to suppress repetition.
#                           Qwen recommends 1.5 for GGUF quantised models (what
#                           Ollama runs). range 0–2; above 2.0 risks language mixing.
#
#   num_predict   : 2048  — max tokens to generate. Qwen3 ceiling is 32768 but
#                           2048 is enough for a risk summary and keeps inference
#                           fast on local hardware.

QWEN3_INSTRUCT_OPTIONS = {
    "temperature":    0.7,
    "top_p":          0.85,   # opened from 0.8 — wider nucleus for sentence variety
    "top_k":          40,     # opened from 20 — more tokens per step, breaks rhythm
    "min_p":          0.0,
    "repeat_penalty": 1.5,
    "num_predict":    2048,
}

dscl        = DSCL()
constrained = dscl.prepare(narrative, domain="travel_risk", only_text=True)

response = ollama.chat(
    model="qwen3:8b",
    options=QWEN3_INSTRUCT_OPTIONS,
    messages=[
        {"role": "system", "content": constrained.system_prompt},
        {"role": "user",   "content": constrained.prompt},
    ]
)

raw_text = response["message"]["content"]
result   = dscl.validate(raw_text, domain=constrained.domain)

print("── OUTPUT ───────────────────────────────────────────────────────────────")
print(result.output)
print()
print("── VALIDATION ───────────────────────────────────────────────────────────")
print(f"FK grade               : {result.fk_grade}")
print(f"Grade target closeness : {result.grade_target_closeness}")
print(f"Sentence variation     : {result.sentence_variation}")
print(f"Vocabulary variety     : {result.vocabulary_variety}")
print(f"Vocabulary displacement: {result.vocabulary_displacement}")
print(f"Violations             : {result.violations}")
print(f"Fixes applied          : {result.fixes_applied}")
print(f"Passed                 : {result.passed}")