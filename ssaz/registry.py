"""Generic component registry used by every extensible SAZ family.

Registration works as a call or a decorator, e.g. embedder registration:

    from ssaz import register_embedder

    @register_embedder("my-model")
    def build(**kwargs):
        return MyEmbedder(**kwargs)

    engine = AzSearchEngine(embedder="my-model")
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


class Registry:
    """Name to factory mapping with decorator-style registration"""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: Dict[str, Callable] = {}

    def register(self, name: str,
                 factory: Optional[Callable] = None) -> Callable:
        if factory is None:
            def decorator(f: Callable) -> Callable:
                self._factories[name] = f
                return f
            return decorator
        self._factories[name] = factory
        return factory

    def create(self, name: str, **kwargs):
        if name not in self._factories:
            raise KeyError(
                f"Unknown {self.kind} {name!r}. Available: {self.names()}")
        return self._factories[name](**kwargs)

    def names(self) -> List[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories
