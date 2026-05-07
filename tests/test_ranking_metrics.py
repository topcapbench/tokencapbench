from budget2success.analysis.paper import sampled_task_budget_ranking_accuracy


def test_sampled_ranking_perfect_and_reverse():
    observed = {"a": 64, "b": 128, "c": 256}
    assert sampled_task_budget_ranking_accuracy({"a": 64, "b": 128, "c": 256}, observed, max_pairs=100, seed=1) == 1.0
    assert sampled_task_budget_ranking_accuracy({"a": 256, "b": 128, "c": 64}, observed, max_pairs=100, seed=1) == 0.0


def test_sampled_ranking_censored_harder_and_deterministic():
    predicted = {"easy": 64, "hard": 512, "unknown": 1024}
    observed = {"easy": 64, "hard": 512, "unknown": None}
    first = sampled_task_budget_ranking_accuracy(predicted, observed, censored_tasks={"unknown"}, max_pairs=2, seed=7)
    second = sampled_task_budget_ranking_accuracy(predicted, observed, censored_tasks={"unknown"}, max_pairs=2, seed=7)

    assert first == second
    assert first == 1.0
