from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

PATHS = {
    "models": {
        "non_modular": DATA_DIR / "wiki_non_modular_mlp_in_out.pt",
        "modular": DATA_DIR / "wiki_fully_modular_mlp_in_out.pt"
    },
    "activations": DATA_DIR / "activations",
    "logs": LOGS_DIR / "model_analysis.log"
}