from dataclasses import dataclass, field, fields as dfields, is_dataclass
from typing import Optional
import torch
import json

class ModelDataClass:
    def get(self, name):
        path = name.split(".")
        name = path[0]
        assert name in self.fields, f"{name} not found"

        if len(path) == 1:
            return getattr(self, name)
        else:
            sub_field = getattr(self, name)
            if not isinstance(sub_field, ModelDataClass):
                raise RuntimeError("Failed to find path")
            return sub_field.get(".".join(path[1:]))

    @property
    def fields(self):
        return [_field.name for _field in dfields(self.__class__)]

    def dump(self):
        data = {}
        for field_name in self.fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, ModelDataClass):
                data[field_name] = field_value.dump()
            else:
                data[field_name] = field_value
        return data

    def display(self):
        print(json.dumps(self.dump(), indent=2, default=str))

    def __post_init__(self):
        for f in dfields(self):
            value = getattr(self, f.name)
            if is_dataclass(f.type) and isinstance(value, dict):
                setattr(self, f.name, f.type(**value))
