# Data

Use JSONL task files. Keep restricted or held-out evaluation data out of public commits.

Required fields:

- `task_id`: unique ID
- `track`: `math`, `coding`, or `swe`
- `prompt`: task text
- `verifier`: verifier name
- `answer`: optional, for answer-key tasks
- `metadata`: source, difficulty, tags, etc.
