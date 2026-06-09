"""Shared StructureMatcher helpers for MP20 crystal generation evaluation."""

from typing import Iterable, Optional, Tuple

from pymatgen.analysis.structure_matcher import StructureMatcher


DEFAULT_MATCHER_PARAMS = {
    "stol": 0.5,
    "angle_tol": 10,
    "ltol": 0.3,
}
DEFAULT_SKIP_STRUCTURE_REDUCTION = True


def build_structure_matcher(**overrides) -> StructureMatcher:
    """Build the StructureMatcher used by the existing MP20 evaluator."""
    params = dict(DEFAULT_MATCHER_PARAMS)
    params.update({key: value for key, value in overrides.items() if value is not None})
    return StructureMatcher(**params)


def group_structures_for_uniqueness(
    structures: Iterable,
    matcher: Optional[StructureMatcher] = None,
    **matcher_overrides,
):
    """Group generated structures exactly as the existing uniqueness metric does."""
    structures = list(structures)
    if matcher is None:
        matcher = build_structure_matcher(**matcher_overrides)
    return matcher.group_structures(structures)


def compute_uniqueness_rate(
    structures: Iterable,
    matcher: Optional[StructureMatcher] = None,
    **matcher_overrides,
) -> Tuple[float, list]:
    """Return ``len(group_structures(valid_structs)) / len(valid_structs)``."""
    structures = list(structures)
    groups = group_structures_for_uniqueness(
        structures,
        matcher=matcher,
        **matcher_overrides,
    )
    if not structures:
        return 0.0, groups
    return len(groups) / len(structures), groups


def find_matching_reference(
    struct,
    reference_structs: Iterable,
    matcher: StructureMatcher,
    skip_structure_reduction: bool = DEFAULT_SKIP_STRUCTURE_REDUCTION,
) -> Optional[Tuple[int, object]]:
    """Return the first reference that matches ``struct``, or None."""
    for idx, other in enumerate(reference_structs):
        other_struct = getattr(other, "structure", other)
        if matcher.fit(
            struct,
            other_struct,
            skip_structure_reduction=skip_structure_reduction,
        ):
            return idx, other
    return None


def is_structure_novel(
    struct,
    reference_structs: Iterable,
    matcher: StructureMatcher,
    skip_structure_reduction: bool = DEFAULT_SKIP_STRUCTURE_REDUCTION,
) -> bool:
    """Return True when ``struct`` does not match any reference structure."""
    return find_matching_reference(
        struct,
        reference_structs,
        matcher,
        skip_structure_reduction=skip_structure_reduction,
    ) is None
