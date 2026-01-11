from build_structuredata_inputs import load_dataframe,  JSON_PATH, \
    build_structuredata_inputs, build_multitask_property_targets
from chgnet.data.dataset import get_train_val_test_loader
import torch
from multitask_chgnet import load_multitask_chgnet
from multitask_trainer import MultiTaskTrainer
from multitask_dataset import PropertyStructureData

# ================================================================
# ---- Main training routine for multitask CHGNet ----
# ================================================================

def train_multitask_chgnet():
    df = load_dataframe(JSON_PATH)

    structures, energies, forces, stresses, magmoms = \
        build_structuredata_inputs(df)
    
    # ---- Multi-property targets ----
    property_targets = build_multitask_property_targets(df)
    print(property_targets.shape)

    # ---- Multi-property dataset ----
    dataset = PropertyStructureData(
        structures=structures,
        energies=energies,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
        property_targets=property_targets,
        prop_dim=4,
    )

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, batch_size=16, train_ratio=0.9, val_ratio=0.05
    )

    print("[INFO] DataLoaders ready.")

    model = load_multitask_chgnet()

    trainer = MultiTaskTrainer(
        model=model,
        targets="e",          # energy only base head
        optimizer="Adam",
        criterion="MSE",
        learning_rate=1e-2,
        epochs=50,
        use_device="cuda",
        prop_weight=1.0,      # λ
    )

    trainer.train(train_loader, val_loader, test_loader)

    print("[OK] Multitask CHGNet training complete.")

if __name__ == "__main__":
    train_multitask_chgnet()