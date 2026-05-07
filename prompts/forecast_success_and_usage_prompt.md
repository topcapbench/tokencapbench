You are estimating your own chance of verified success under explicit token budgets and your visible token needs.

You will see a task, the solver scaffold, and a list of token budgets. Do not solve the task. Do not include solution steps. Return JSON only.

The success probabilities should estimate whether a fresh solver run using the same model and scaffold will reach a verified correct result before each token budget. The token estimates should be pre-execution estimates, not a solution attempt.

Return exactly this schema:

{
  "p_success_by_budget": {
    "64": 0.0,
    "128": 0.0,
    "256": 0.0,
    "512": 0.0,
    "1024": 0.0,
    "2048": 0.0
  },
  "predicted_first_success_budget": 512,
  "predicted_visible_tokens_unconstrained": 900,
  "predicted_output_tokens_unconstrained": 700,
  "predicted_total_visible_tokens": 900,
  "predicted_unconstrained_output_tokens": 700,
  "predicted_total_visible_tokens_to_solve": 900,
  "confidence": 0.5
}
