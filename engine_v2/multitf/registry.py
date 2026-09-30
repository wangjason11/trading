"""StructureRegistry — per-entity df isolation for the multi-TF refactor.

Each ``structure_path_id`` (e.g. ``H1.main``, ``H1.main >> M15.counter``)
maps to one ``EntityState`` that owns the candle df + downstream artifacts
for that entity. The artifacts live in that df's ``attrs`` (``kl_zones``,
``poi_zones``, ``fib_states``, ...; PART4 §9.3), so ``registry.get(path).df.attrs``
IS the entity's store — both charts resolve their data this way.

Spec: PART4_REFACTOR_SPEC.md §9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

import pandas as pd


PATH_SEPARATOR = " >> "


@dataclass(frozen=True)
class EntityState:
    """One entity in the multi-TF hierarchy.

    Identity (``timeframe``, ``role``, ``parent_path_id``,
    ``starting_alignment``) is immutable. The owned ``df`` reference is
    fixed at registration; per-candle columns and ``df.attrs`` evolve as
    the pipeline writes into them.
    """
    structure_path_id: str
    df: pd.DataFrame
    timeframe: str
    role: Literal["main", "subordinate"]
    parent_path_id: Optional[str] = None
    starting_alignment: Optional[Literal["confluence", "counter"]] = None


def _parent_path_id(path_id: str) -> Optional[str]:
    """Strip the trailing segment of a path. ``H1.main`` → None."""
    if PATH_SEPARATOR not in path_id:
        return None
    return path_id.rsplit(PATH_SEPARATOR, 1)[0]


class StructureRegistry:
    """Owner of every entity's df for a single replay/session."""

    def __init__(self) -> None:
        self._entities: Dict[str, EntityState] = {}

    def register(
        self,
        path_id: str,
        *,
        df: pd.DataFrame,
        timeframe: str,
        role: Literal["main", "subordinate"],
        starting_alignment: Optional[Literal["confluence", "counter"]] = None,
    ) -> EntityState:
        if path_id in self._entities:
            raise ValueError(f"Entity already registered: {path_id}")

        parent_path_id = _parent_path_id(path_id)
        if role == "main":
            if parent_path_id is not None:
                raise ValueError(
                    f"main role requires top-level path_id, got '{path_id}'")
            if starting_alignment is not None:
                raise ValueError("main role does not carry starting_alignment")
        else:  # subordinate
            if parent_path_id is None:
                raise ValueError(
                    f"subordinate role requires parent path, got '{path_id}'")
            if starting_alignment not in ("confluence", "counter"):
                raise ValueError(
                    "subordinate role requires starting_alignment "
                    "in {'confluence', 'counter'}")

        df.attrs["structure_path_id"] = path_id
        df.attrs["parent_path_id"] = parent_path_id

        entity = EntityState(
            structure_path_id=path_id,
            df=df,
            timeframe=timeframe,
            role=role,
            parent_path_id=parent_path_id,
            starting_alignment=starting_alignment,
        )
        self._entities[path_id] = entity
        return entity

    def get(self, path_id: str) -> Optional[EntityState]:
        return self._entities.get(path_id)

    def parent_of(self, path_id: str) -> Optional[EntityState]:
        parent_id = _parent_path_id(path_id)
        if parent_id is None:
            return None
        return self._entities.get(parent_id)

    def children_of(self, path_id: str) -> List[EntityState]:
        return [e for e in self._entities.values()
                if e.parent_path_id == path_id]

    def all(self) -> Iterable[EntityState]:
        return self._entities.values()
