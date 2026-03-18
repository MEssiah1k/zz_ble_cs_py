from src.metrics import rmse, summarize_errors


def test_rmse_and_summary_are_consistent():
    true_values = [1.0, 2.0, 3.0]
    est_values = [1.1, 2.1, 2.9]
    value = rmse(true_values, est_values)
    summary = summarize_errors(true_values, est_values, thresholds=[0.2])
    assert value > 0.0
    assert abs(summary["rmse"] - value) < 1e-12
    assert 0.0 <= summary["success_rate_le_0.2m"] <= 1.0
