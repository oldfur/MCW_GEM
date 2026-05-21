import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


PAD_ATOM_TYPE_SYMBOL = "PAD"
UNKNOWN_ATOM_TYPE_SYMBOL = "UNKNOWN"
ATOM_TYPE_SOFTMAX_DIM = -1

# A conservative fallback shortlist for inorganic crystals.
# Class indices follow the existing atomic-number-like convention:
# 1->H, 6->C, 8->O, 11->Na, ...
DEFAULT_FALLBACK_CLASS_IDS = [8, 14, 11, 20, 22, 13, 26, 16, 15, 17, 29, 30, 6, 1]


def build_canonical_atom_decoder(
    atom_encoder: Dict[str, int],
    num_classes: int,
    pad_symbol: str = PAD_ATOM_TYPE_SYMBOL,
) -> Tuple[List[str], Dict[str, int]]:
    assert num_classes >= 2, f"num_classes must be >= 2, got {num_classes}"
    decoder = [pad_symbol] + [f"UNUSED_{idx}" for idx in range(1, num_classes)]
    out_of_range = {}

    for symbol, class_idx in atom_encoder.items():
        if class_idx <= 0:
            raise AssertionError(f"Atom encoder index must be positive for real elements, got {symbol}->{class_idx}")
        if class_idx >= num_classes:
            out_of_range[symbol] = int(class_idx)
            continue
        if decoder[class_idx] != f"UNUSED_{class_idx}":
            raise AssertionError(
                f"Duplicate atom decoder assignment for class {class_idx}: "
                f"{decoder[class_idx]} vs {symbol}"
            )
        decoder[class_idx] = symbol

    if decoder[0] == "H":
        raise AssertionError("Canonical atom decoder must reserve class 0 for PAD/UNKNOWN, not H.")
    return decoder, out_of_range


def get_known_atom_class_ids(atom_decoder: Sequence[str]) -> List[int]:
    known_ids = []
    for class_idx, symbol in enumerate(atom_decoder):
        if class_idx == 0:
            continue
        if symbol.startswith("UNUSED_"):
            continue
        if symbol in {PAD_ATOM_TYPE_SYMBOL, UNKNOWN_ATOM_TYPE_SYMBOL}:
            continue
        known_ids.append(class_idx)
    return known_ids


def assert_atom_mapping_roundtrip(
    atom_encoder: Dict[str, int],
    atom_decoder: Sequence[str],
    required_symbols: Iterable[str] = ("H", "C", "O", "Na"),
) -> None:
    if atom_decoder[0] == "H":
        raise AssertionError("atom_decoder[0] must not be H.")

    for symbol in required_symbols:
        if symbol not in atom_encoder:
            continue
        class_idx = atom_encoder[symbol]
        if class_idx >= len(atom_decoder):
            raise AssertionError(
                f"Roundtrip check failed for {symbol}: class index {class_idx} "
                f"is outside atom_decoder length {len(atom_decoder)}"
            )
        decoded_symbol = atom_decoder[class_idx]
        if decoded_symbol != symbol:
            raise AssertionError(
                f"Roundtrip check failed for {symbol}: "
                f"{symbol}->{class_idx}->{decoded_symbol}"
            )


def canonicalize_dataset_info(dataset_info: Dict) -> Dict:
    if dataset_info.get("_canonical_atom_type_mapping", False):
        return dataset_info

    atom_encoder = dataset_info["atom_encoder"]
    num_classes = max(atom_encoder.values())
    atom_decoder, out_of_range = build_canonical_atom_decoder(atom_encoder, num_classes=num_classes)
    assert_atom_mapping_roundtrip(atom_encoder, atom_decoder)

    dataset_info["atom_decoder"] = atom_decoder
    dataset_info["atom_pad_idx"] = 0
    dataset_info["atom_unknown_idx"] = 0
    dataset_info["known_atom_class_ids"] = get_known_atom_class_ids(atom_decoder)
    dataset_info["atom_encoder_out_of_range"] = out_of_range
    dataset_info["_canonical_atom_type_mapping"] = True
    return dataset_info


def class_index_to_symbol(atom_decoder: Sequence[str], class_idx: int) -> str:
    if class_idx < 0 or class_idx >= len(atom_decoder):
        return f"OUT_OF_RANGE_{class_idx}"
    return atom_decoder[class_idx]


def build_safe_fallback_atom_logits(
    num_nodes: int,
    num_classes: int,
    device,
    dtype,
    previous_atom_logits: Optional[torch.Tensor] = None,
    known_atom_class_ids: Optional[Sequence[int]] = None,
    preferred_class_ids: Optional[Sequence[int]] = None,
):
    meta = {
        "source": None,
        "used_previous_logits": False,
        "known_atom_class_ids": None,
        "preferred_class_ids": None,
    }

    if previous_atom_logits is not None:
        if previous_atom_logits.shape == (num_nodes, num_classes) and torch.isfinite(previous_atom_logits).all():
            logits = previous_atom_logits.detach().clone().to(device=device, dtype=dtype)
            logits[:, 0] = -1e9
            meta["source"] = "previous_atom_logits"
            meta["used_previous_logits"] = True
            return logits, meta

    if known_atom_class_ids is None:
        known_atom_class_ids = list(range(1, num_classes))
    else:
        known_atom_class_ids = [int(idx) for idx in known_atom_class_ids if 0 < int(idx) < num_classes]
    if not known_atom_class_ids:
        known_atom_class_ids = list(range(1, num_classes))

    preferred = preferred_class_ids or DEFAULT_FALLBACK_CLASS_IDS
    preferred = [int(idx) for idx in preferred if idx in known_atom_class_ids]
    if not preferred:
        preferred = known_atom_class_ids[: min(len(known_atom_class_ids), 16)]

    logits = torch.full((num_nodes, num_classes), -12.0, device=device, dtype=dtype)
    logits[:, 0] = -1e9
    logits[:, known_atom_class_ids] = -4.0

    for rank, class_idx in enumerate(preferred):
        logits[:, class_idx] = 1.5 - 0.05 * rank

    if num_nodes > 0 and preferred:
        node_indices = torch.arange(num_nodes, device=device)
        primary = torch.tensor(
            [preferred[i % len(preferred)] for i in range(num_nodes)],
            device=device,
            dtype=torch.long,
        )
        secondary = torch.tensor(
            [preferred[(i + 3) % len(preferred)] for i in range(num_nodes)],
            device=device,
            dtype=torch.long,
        )
        logits[node_indices, primary] += 0.75
        logits[node_indices, secondary] += 0.15

    meta["source"] = "safe_prior_logits"
    meta["known_atom_class_ids"] = list(known_atom_class_ids)
    meta["preferred_class_ids"] = list(preferred)
    return logits, meta
