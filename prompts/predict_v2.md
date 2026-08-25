<!--
=============================================================================
predict_v2.md — live prediction prompt
Explaining Markets · Q3 2026 · version 4.0.0 · owner: Will · 2026-08-24

Pared it down. The goal isn't to tell the model what to do, it's to see how 
well it can reason from the information we give it.
=============================================================================
-->

# Task

Your task is to predict what percentile the stock's next day trading performance will fall in relative to the market.

---

# Summary of the Earnings Call

{summary_text}

# Prediction Objective

You are predicting **belief revision**, not company performance. Strong results
can be neutral when already priced; weak results can be positive when the market
feared worse.

# This company's dossier

**`prior_reactions` and `reaction_statistics`** — how this stock has actually
behaved on its own earnings days, measured net of the market exactly as this
competition scores. Use it for **magnitude and asymmetry, not direction**.

**`forward_estimates`, when present** — consensus analyst estimates: `eps_avg`
with its `low`/`high` range, the number of analysts, `eps_year_ago`, and recent
estimate revisions. **This is the expectations baseline.** When it is here, use
it.

**If the `forward_estimates` block is absent, you have no consensus baseline.**

{dossier}

# Industry Trends

These are observed trends in the industry to consider before making a prediction.

{industry_rules}

---

# Global Observations

These are selected empirical patterns that may be useful context. They are neither exhaustive nor deterministic; assess their relevance alongside all available evidence.
{global_rules}

---

# Output

Return JSON only, with the keys in exactly this order. The order matters: you
are establishing the facts before you interpret them.

```json
{
  "relevant_context": ["<trend or observation ID, only if materially relevant>"],
  "predicted_percentile": <number between 0 and 1>
}
```