from __future__ import annotations


class History:
    def __init__(self):
        self.data = {}

    def log(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def get(self, key: str):
        return self.data.get(key, [])

    def keys(self):
        return self.data.keys()

    def as_dict(self):
        return dict(self.data)