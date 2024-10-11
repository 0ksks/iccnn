from datetime import datetime
from typing import Any

from yaml import safe_load
import platform

EPSILON = 1e-4

CONFIG = safe_load(open("config.yaml", encoding="utf-8"))
platform_info = platform.platform()
run_time = datetime.now().strftime("%m_%d(%H-%M)")
RUN_NAME = f"[{platform_info}][{run_time}]"


def parse_config_path(config_path: list[str]) -> str:
    import os
    path = ""
    for file in config_path:
        path = os.path.join(path, file)
    return path


def get_config_value(config_key: str) -> Any:
    value = CONFIG
    config_key = config_key.split(".")
    for key in config_key:
        if key in value:
            value = value[key]
        else:
            return None
    return value
