from pathlib import Path

from experiments.runtime.locations import experiment_folder

HERE = Path(__file__).resolve().parent


def __getattr__(name: str) -> Path:
    # INPUTS, WORK, DATA and RAW come from the environment when a script first uses them.
    return experiment_folder(HERE.name, name)
