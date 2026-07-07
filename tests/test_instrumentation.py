from zkllms import instrumentation


def test_tee_phase_records_a_named_phase_with_non_negative_timings():
    phases = []

    with instrumentation.tee_phase("prove", phases):
        sum(range(10000))

    assert len(phases) == 1
    metric = phases[0]
    assert metric.name == "prove"
    assert metric.wall_s >= 0.0
    assert metric.cpu_s >= 0.0
    assert metric.peak_rss_mb >= 0.0


def test_estimate_transformer_flops_is_positive_and_scales_with_layers():
    one_layer = instrumentation.estimate_transformer_flops(
        d_model=896, n_heads=14, seq_len=4, n_layers=1, intermediate=4864
    )
    two_layers = instrumentation.estimate_transformer_flops(
        d_model=896, n_heads=14, seq_len=4, n_layers=2, intermediate=4864
    )

    assert one_layer > 0
    assert two_layers == 2 * one_layer


def test_tee_profile_summary_table_reports_phases_and_circuit_facts():
    phases = [
        instrumentation.PhaseMetrics("prove", 11.24, 44.81, 398.7, 891.2),
    ]
    profile = instrumentation.TEEProfile(
        phases=phases,
        proof_size_bytes=7065,
        constraint_count=3_145_728,
        inference_flops=186_400_000,
    )

    table = profile.summary_table()

    assert "prove" in table
    assert "3,145,728" in table
    assert "Proof size" in table
    assert "Wall" in table and "CPU" in table
