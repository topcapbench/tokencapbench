You are making a pre-execution TokenCapBench forecast.

You will see a task, the solver scaffold, and a list of generated-token budgets. Do not solve the task, do not reason through solution steps, and do not include any answer content. Your job is only to estimate how likely a fresh solver context using the same model and scaffold is to pass the verifier before each budget.

The solver will not see this forecast. The budget caps are enforced externally by the harness, so estimate the probability of verified success under each imposed cap.

Return JSON only with this schema:

{
  "p_success_by_budget": {
    "<budget>": <probability between 0 and 1>
  },
  "median_success_budget": <positive number or null>,
  "p_failure_at_max_budget": <probability between 0 and 1>,
  "short_rationale": "one sentence, no solution steps"
}
