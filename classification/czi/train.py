import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import matplotlib.pyplot as plt
import numpy as np

# --- Local Project Imports ---
# Configuration parameters for the project
import config.config_classification as config
# Custom dataset and dataloader for Cryo-ET data
from dataloader import CryoETClassificationDataset
# 3D ResNet model implementation
from models.resnet3d import ResNet3D

# For detailed metrics and confusion matrix plotting
try:
    import torchmetrics
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("WARNING: 'torchmetrics' is not installed. Detailed metrics (Accuracy, F1) will not be available.")
    print("Please install with 'pip install torchmetrics'.")
    TORCHMETRICS_AVAILABLE = False


class CryoETClassificationDataModule(pl.LightningDataModule):
    """
    Encapsulates all data loading logic for Cryo-ET patch classification.
    Handles dataset creation for training, validation, and testing splits.
    """
    def __init__(self, data_root, target_patch_size, batch_size, num_workers, use_augmentation):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        data_root = Path(self.hparams.data_root)

        if stage in ('fit', None):
            train_dir = data_root / 'train'
            val_dir = data_root / 'val'

            if not train_dir.exists() or not val_dir.exists():
                raise FileNotFoundError(f"The 'train' and/or 'val' directories do not exist in {data_root}. "
                                        "Please run prepare_classification_data.py first to generate the dataset.")
            self.train_dataset = CryoETClassificationDataset(
                data_root=train_dir,
                target_patch_size=self.hparams.target_patch_size,
                mode='train',
                use_augmentation=self.hparams.use_augmentation
            )
            self.val_dataset = CryoETClassificationDataset(
                data_root=val_dir,
                target_patch_size=self.hparams.target_patch_size,
                mode='val',
                use_augmentation=False
            )

        if stage in ('test', None):
            test_dir = data_root / 'test'
            if test_dir.exists():
                self.test_dataset = CryoETClassificationDataset(
                    data_root=test_dir,
                    target_patch_size=self.hparams.target_patch_size,
                    mode='val', # Use 'val' mode to ensure no augmentation is applied during testing
                    use_augmentation=False
                )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        if not self.test_dataset:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True
        )


class ClassificationSystem(pl.LightningModule):
    """
    Encapsulates the model, loss, and training logic in a PyTorch Lightning module.
    This structure integrates the model, loss function, optimizer, and metrics
    for a complete training and evaluation pipeline.
    """
    def __init__(self, learning_rate, num_classes, use_scheduler, model_mode):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet3D(
            in_channels=config.IN_CHANNELS,
            num_classes=num_classes,
            # This parameter allows switching between different ResNet3D variants.
            mode=self.hparams.model_mode
        )
        self.loss_fn = nn.CrossEntropyLoss()

        if TORCHMETRICS_AVAILABLE:
            # Metrics for training and validation
            self.train_acc = MulticlassAccuracy(num_classes=num_classes)
            self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
            self.val_acc = MulticlassAccuracy(num_classes=num_classes)
            self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
            self.val_cm = MulticlassConfusionMatrix(num_classes=num_classes)
            # Metrics for testing
            self.test_acc = MulticlassAccuracy(num_classes=num_classes)
            self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
            self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        patches, labels = batch["patch"], batch["label"]
        logits = self(patches)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=bool(self.logger))
        if TORCHMETRICS_AVAILABLE:
            self.train_acc(logits, labels)
            self.train_f1(logits, labels)
            self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, logger=bool(self.logger))
            self.log('train/f1_macro', self.train_f1, on_step=False, on_epoch=True, logger=bool(self.logger))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=bool(self.logger))
        if TORCHMETRICS_AVAILABLE:
            self.val_acc(logits, labels)
            self.val_f1(logits, labels)
            self.val_cm(logits, labels)
            self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, logger=bool(self.logger))
            self.log('val/f1_macro', self.val_f1, on_step=False, on_epoch=True, logger=bool(self.logger))

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        if TORCHMETRICS_AVAILABLE:
            self.test_acc(logits, labels)
            self.test_f1(logits, labels)
            self.test_cm(logits, labels)
            self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('test/f1_macro', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if TORCHMETRICS_AVAILABLE and not self.trainer.sanity_checking and self.logger:
            fig, _ = self.val_cm.plot(labels=config.CLASS_NAMES)
            self.logger.log_image(key="validation/confusion_matrix", images=[fig], caption=[f"Epoch {self.current_epoch + 1}"])
            plt.close(fig)
            self.val_cm.reset()

    def on_test_epoch_end(self):
        if TORCHMETRICS_AVAILABLE:
            print("\n--- Test Set Evaluation Results ---")

            # Log confusion matrix to the logger (e.g., W&B)
            if self.logger and self.logger.experiment:
                fig, _ = self.test_cm.plot(labels=config.CLASS_NAMES)
                self.logger.log_image(key="test/confusion_matrix", images=[fig], caption=["Final Confusion Matrix (Test)"])
                plt.close(fig)

            # Display final metrics and log to the experiment summary
            test_acc_val = self.test_acc.compute().item()
            test_f1_val = self.test_f1.compute().item()
            print(f"  - Test Accuracy: {test_acc_val:.4f}")
            print(f"  - Test F1-Score (Macro): {test_f1_val:.4f}")
            
            # Add final metrics to the experiment summary for a concise overview
            if self.logger and self.logger.experiment:
                self.logger.experiment.summary["test_accuracy"] = test_acc_val
                self.logger.experiment.summary["test_f1_macro"] = test_f1_val

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        if not self.hparams.use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main():
    parser = argparse.ArgumentParser(description="Train a 3D CNN for Cryo-ET patch classification.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for classification data (containing subfolders per tomogram type, e.g., '.../classification/'). If not provided, it's inferred from config.DATASET_DIR's parent.")
    parser.add_argument("--tomo_type_train", type=str, default="wbp", help="Tomogram type to use for training (e.g., 'denoised', 'wbp').")
    parser.add_argument("--comment", type=str, default="", help="A comment for this run, used in the logger.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a .ckpt file to resume training.")
    parser.add_argument("--mode", type=str, default=config.MODEL_MODE, help="ResNet3D model mode ('standard', 'seqsi_avg', etc.).")
    parser.add_argument("--use_wandb", action='store_true', help="Enable logging with Weights & Biases. Disabled by default.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # --- Determine Data Directory ---
    if args.data_root:
        base_data_root = Path(args.data_root)
    else:
        # Assume config.DATASET_DIR is '/path/to/classification/'
        base_data_root = Path(config.DATASET_DIR)
        print(f"INFO: --data_root not provided. Using parent directory of config.DATASET_DIR: {base_data_root}")

    train_data_root = base_data_root / args.tomo_type_train
    if not train_data_root.exists():
        print(f"ERROR: The training data directory '{train_data_root}' does not exist.")
        return

    # --- DataModule ---
    datamodule = CryoETClassificationDataModule(
        data_root=train_data_root,
        target_patch_size=config.TARGET_PATCH_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        use_augmentation=config.USE_AUGMENTATION
    )

    # --- Model ---
    model = ClassificationSystem(
        learning_rate=args.lr,
        num_classes=config.NUM_CLASSES,
        use_scheduler=config.USE_SCHEDULER,
        model_mode=args.mode
    )

    # --- Logging and Callbacks Setup ---
    logger = False # Disable logging by default
    run_name = f"{args.mode}_{args.tomo_type_train}"
    if args.comment:
        run_name += f"_{args.comment}"

    if args.use_wandb:
        print("Logging with Weights & Biases enabled.")
        wandb_save_dir = os.path.join("wandb_logs")
        os.makedirs(wandb_save_dir, exist_ok=True)

        wandb_config = {
            "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE, "epochs": config.EPOCHS,
            "use_scheduler": config.USE_SCHEDULER, "use_augmentation": config.USE_AUGMENTATION,
            "target_patch_size": config.TARGET_PATCH_SIZE, "model_mode": args.mode,
            "tomo_type_train": args.tomo_type_train, "num_classes": config.NUM_CLASSES,
            "dataset_dir": str(train_data_root),
        }
        # log_model="all" saves the model checkpoint to W&B at the end of training.
        logger = WandbLogger(project="cryoet-classification", name=run_name, log_model="all", save_dir=wandb_save_dir, config=wandb_config)

    # Checkpoints are organized by seed and run name for easy retrieval.
    base_checkpoint_path = Path("checkpoints") # Base directory for all checkpoints
    checkpoint_dir = base_checkpoint_path / str(args.seed) / run_name

    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc', dirpath=checkpoint_dir, filename='bestmodel', save_top_k=1, mode='max',
    )

    # --- Callbacks ---
    callbacks = [checkpoint_callback]
    if logger: # Only add LR Monitor if a logger is active
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator="auto", devices="auto", max_epochs=config.EPOCHS,
        logger=logger,
        callbacks=callbacks,
        precision='bf16-mixed' if config.DEVICE == 'cuda' and torch.cuda.is_bf16_supported() else '32-true',
        check_val_every_n_epoch=config.VALIDATE_EVERY_N_EPOCHS,
        # 'deterministic=True' ensures reproducibility but might slightly impact performance.
        deterministic=True,
        # Gradient clipping is applied to prevent exploding gradients, which can stabilize training.
        gradient_clip_val=1.0,
    )

    # --- Start Training ---
    print(f"Starting training run: {run_name}")
    print(f"Checkpoints will be saved in: {checkpoint_dir}")
    trainer.fit(model, datamodule, ckpt_path=args.resume_from_checkpoint)

    # --- Post-training Evaluation on All Tomogram Types with the Best Model ---
    print("\n--- Starting evaluation on different tomogram types with the best model ---")

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not Path(best_model_path).exists():
        print("No 'best model' found. Final evaluation is skipped.")
    else:
        print(f"The best model will be loaded from: {best_model_path}")

        try:
            # Automatically detect tomogram types that have a 'test' folder
            tomo_types = sorted([d.name for d in base_data_root.iterdir() if d.is_dir() and (d / 'test').exists()])
        except FileNotFoundError:
            tomo_types = []
            print(f"ERROR: The parent data directory '{base_data_root}' was not found.")

        if not tomo_types:
            print(f"WARNING: No valid test datasets found in {base_data_root}. Evaluation is skipped.")
        else:
            print(f"Tomogram types found for evaluation: {tomo_types}")
            all_results = []
            # Explicitly load the best model to ensure the correct one is used
            best_model = ClassificationSystem.load_from_checkpoint(best_model_path)

            # Evaluate the best model on all available tomogram types to assess its generalization capabilities.
            for tomo_type in tomo_types:
                print(f"\n--- Evaluating on tomogram type: {tomo_type} ---")
                current_data_root = base_data_root / tomo_type

                # Configure a specific DataModule for the current tomogram type
                eval_datamodule = CryoETClassificationDataModule(
                    data_root=current_data_root,
                    target_patch_size=config.TARGET_PATCH_SIZE,
                    batch_size=config.BATCH_SIZE * 2,  # Larger batch size for inference
                    num_workers=config.NUM_WORKERS,
                    use_augmentation=False
                )

                # Reset model metrics before each test to ensure independent evaluation
                for metric in [best_model.test_acc, best_model.test_f1, best_model.test_cm]:
                    if hasattr(metric, 'reset'):
                        metric.reset()

                # Run the test on the best model
                trainer.test(model=best_model, datamodule=eval_datamodule, verbose=False)

                # Retrieve metrics from the model's state after testing
                acc = best_model.test_acc.compute().cpu()
                f1 = best_model.test_f1.compute().cpu()
                cm = best_model.test_cm.compute().cpu()

                # Save detailed metrics and confusion matrix to a file
                results_path = Path(checkpoint_dir) / f"test_results_{tomo_type}.pt"
                torch.save({'accuracy': acc, 'f1_macro': f1, 'confusion_matrix': cm}, results_path)
                print(f"  -> Detailed results (including CM) saved to: {results_path}")

                all_results.append({
                    'tomo_type': tomo_type,
                    'test_accuracy': acc.item(),
                    'test_f1_macro': f1.item(),
                })

            if all_results:
                results_df = pd.DataFrame(all_results)
                print("\n--- Summary of Evaluation Results ---")
                print(results_df.to_string(index=False))

                if isinstance(logger, WandbLogger) and logger.experiment:
                    logger.log_table(key="evaluation_summary", dataframe=results_df)

                output_csv = Path(checkpoint_dir) / "evaluation_summary.csv"
                results_df.to_csv(output_csv, index=False, float_format='%.4f')
                print(f"\nComplete results saved to: {output_csv}")


if __name__ == "__main__":
    main()
