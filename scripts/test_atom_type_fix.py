#!/usr/bin/env python3
import torch
import torch.nn.functional as F

from mp20.atom_type_mapping import (
    ATOM_TYPE_SOFTMAX_DIM,
    PAD_ATOM_TYPE_SYMBOL,
    assert_atom_mapping_roundtrip,
    build_safe_fallback_atom_logits,
    canonicalize_dataset_info,
    class_index_to_symbol,
)


def build_test_dataset_info():
    dataset_info = {
        "atom_encoder": {
            "H": 1,
            "C": 6,
            "O": 8,
            "Na": 11,
            "Si": 14,
            "Ca": 20,
        }
    }
    canonicalize_dataset_info(dataset_info)
    return dataset_info


def test_canonical_mapping():
    dataset_info = build_test_dataset_info()
    atom_decoder = dataset_info["atom_decoder"]
    atom_encoder = dataset_info["atom_encoder"]

    assert atom_decoder[0] == PAD_ATOM_TYPE_SYMBOL
    assert atom_decoder[0] != "H"
    assert class_index_to_symbol(atom_decoder, 1) == "H"
    assert class_index_to_symbol(atom_decoder, 6) == "C"
    assert class_index_to_symbol(atom_decoder, 8) == "O"
    assert class_index_to_symbol(atom_decoder, 11) == "Na"
    assert_atom_mapping_roundtrip(atom_encoder, atom_decoder, required_symbols=("H", "C", "O", "Na"))


def test_empty_graph_fallback_logits():
    dataset_info = build_test_dataset_info()
    num_classes = len(dataset_info["atom_decoder"])
    logits, meta = build_safe_fallback_atom_logits(
        num_nodes=16,
        num_classes=num_classes,
        device=torch.device("cpu"),
        dtype=torch.float32,
        previous_atom_logits=None,
        known_atom_class_ids=dataset_info["known_atom_class_ids"],
    )

    assert logits.shape == (16, num_classes)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits, torch.zeros_like(logits), atol=1e-8, rtol=0.0)
    assert torch.all(logits[:, 0] < -1e8)
    decoded = torch.argmax(logits, dim=-1)
    assert not torch.any(decoded == 0)
    assert not torch.all(decoded == 1), "empty-graph fallback still decodes to deterministic all-H"
    assert len(torch.unique(decoded)) > 1, "fallback logits should not collapse every node to one element"
    assert meta["source"] == "safe_prior_logits"


def test_softmax_and_padding_decode():
    dataset_info = build_test_dataset_info()
    atom_decoder = dataset_info["atom_decoder"]
    num_classes = len(atom_decoder)

    logits = torch.full((2, 4, num_classes), -5.0, dtype=torch.float32)
    logits[0, 0, 6] = 3.0
    logits[0, 1, 8] = 4.0
    logits[0, 2, 1] = 5.0
    logits[0, 3, 0] = 9.0
    logits[1, 0, 11] = 2.5
    logits[1, 1, 14] = 2.0
    logits[1, 2, 0] = 7.0
    logits[1, 3, 0] = 7.0

    node_mask = torch.tensor(
        [
            [[1.0], [1.0], [1.0], [0.0]],
            [[1.0], [1.0], [0.0], [0.0]],
        ]
    )
    valid_mask = node_mask.squeeze(-1).bool()

    probs = torch.softmax(logits, dim=ATOM_TYPE_SOFTMAX_DIM)
    assert torch.allclose(
        probs.sum(dim=-1)[valid_mask],
        torch.ones_like(probs.sum(dim=-1)[valid_mask]),
        atol=1e-5,
        rtol=1e-5,
    )

    masked_logits = logits.clone()
    masked_logits[:, :, 0] = -1e9
    decoded = torch.argmax(masked_logits, dim=-1)
    assert decoded.shape == (2, 4)
    assert not torch.any(decoded[valid_mask] == 0)
    assert class_index_to_symbol(atom_decoder, 0) != "H"
    assert class_index_to_symbol(atom_decoder, int(decoded[0, 2].item())) == "H"

    one_hot = F.one_hot(decoded, num_classes=num_classes) * node_mask
    assert one_hot.shape == logits.shape
    assert torch.all(one_hot[~valid_mask] == 0), "padding nodes must stay zero after decode"


def main():
    test_canonical_mapping()
    test_empty_graph_fallback_logits()
    test_softmax_and_padding_decode()
    print("atom type fix checks passed.")


if __name__ == "__main__":
    main()
