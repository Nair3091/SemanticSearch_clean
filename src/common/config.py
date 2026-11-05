from pathlib import Path
import yaml

_CONFIG = None


def load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
        with open(config_path) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG