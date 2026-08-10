from train.evaluate_physics_sensitivity import build_conditions, summarize


def test_conditions_change_one_uncertain_parameter_at_a_time():
    conditions = build_conditions()
    assert conditions[0]["id"] == "nominal"
    assert len({row["id"] for row in conditions}) == len(conditions)
    assert {row["parameter"] for row in conditions} >= {
        "ball.rolling_friction_length", "camera.latency",
        "actuator.centre_bias"}
    wall_low = next(row for row in conditions
                    if row["id"] == "ball.wall_restitution.low")
    assert wall_low["params"]["sim.wall_dampratio"] == 2.0


def test_summary_ranks_failures_before_successes():
    common = {"axis": None, "value": 1.0,
              "mean_cross_track_m": 0.001, "seconds": 100.0}
    rows = [
        {**common, "condition": "good", "parameter": "x", "level": "high",
         "outcome": "goal", "completion": 0.99},
        {**common, "condition": "bad", "parameter": "y", "level": "low",
         "outcome": "timeout", "completion": 0.5},
    ]
    ranked = summarize(rows)
    assert [row["condition"] for row in ranked] == ["bad", "good"]
    assert ranked[0]["mean_seconds_to_goal"] is None
