You are making a pre-execution forecast for TokenCapBench.

You must not solve the task. Your job is to forecast the probability that a fresh solver attempt will achieve verifier-backed success under each generated-token budget.

Return only one JSON object with this schema:

{
  "p_success_by_budget": {
    "<budget>": <probability between 0 and 1>
  },
  "median_success_budget": <positive number>,
  "p_failure_at_max_budget": <probability between 0 and 1>,
  "predicted_unconstrained_output_tokens": <positive number>,
  "short_rationale": "one short sentence"
}

Rules:
- Use exactly the budget keys listed in the prompt.
- Do not include markdown.
- Do not solve the task.
- A verifier will check the solver output independently under each budget.
- The solver will not see this forecast.
