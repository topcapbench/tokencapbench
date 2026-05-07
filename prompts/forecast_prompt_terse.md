Estimate verified success probability under each token budget.

Do not solve the task. Return JSON only:

{
  "p_success_by_budget": {
    "<budget>": <0 to 1 probability>
  },
  "median_success_budget": <positive number or null>,
  "p_failure_at_max_budget": <0 to 1 probability>,
  "short_rationale": "one sentence, no solution steps"
}
