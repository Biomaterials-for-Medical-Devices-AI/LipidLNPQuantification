from __future__ import annotations

from pathlib import Path

from lipid_quantification.train_model.runner import TrainingRun, load_config


def main() -> None:
    cfg = load_config(Path(__file__).parent / "args.yaml")
    TrainingRun(cfg, root=cfg["paths"]["run"]).run()


if __name__ == "__main__":
    main()
