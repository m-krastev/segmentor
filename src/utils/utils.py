from dataclasses import _DataclassT
from typing import Any, Dict

def create_base(model_class: _DataclassT, data: Dict[str, Any]) -> _DataclassT:
    valid_keys = set(_DataclassT.__dataclass_fields__.keys())
    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
    return model_class(**filtered_data)