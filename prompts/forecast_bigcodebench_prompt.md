You are making a pre-execution TokenCapBench forecast.

Forecast the probability that a fresh solve attempt will pass the BigCodeBench verifier under each generated-token budget.

Do not solve the task. Do not include solution steps or answer content. The solver will run in a separate context and will not see this forecast. The generated-token caps are enforced by the harness/API.

Return JSON only with this schema:

{
  "p_success_by_budget": {
    "<budget>": <probability between 0 and 1>
  },
  "median_success_budget": <positive number or null>,
  "p_failure_at_max_budget": <probability between 0 and 1>,
  "short_rationale": "one sentence, no solution steps"
}
