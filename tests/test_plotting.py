from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from nf2pino.plotting import plot_convergence


def test_plot_convergence_supports_mixed_history_schemas(tmp_path: Path):
    histories = {
        "PINO → NF2 prefit": [{"step": 1, "mse": 1.0}, {"step": 2, "mse": 0.5}],
        "Hybrid NF2": [{"epoch": 0, "global_step": 10, "total_loss": 3.0}],
        "NF2 scratch": [{"epoch": 0, "global_step": 10, "total_loss": 4.0}],
    }

    output_path = tmp_path / "convergence.png"
    returned = plot_convergence(histories, output_path=output_path)

    assert returned == output_path
    assert output_path.is_file()


def test_plot_convergence_falls_back_to_any_numeric_metric(tmp_path: Path):
    histories = {
        "Custom run": [{"epoch": 0, "global_step": 5, "weird_metric": 2.0}],
    }

    output_path = tmp_path / "custom_convergence.png"
    returned = plot_convergence(histories, output_path=output_path)

    assert returned == output_path
    assert output_path.is_file()
