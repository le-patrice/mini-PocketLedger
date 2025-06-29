import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Config,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
from collections import Counter
import pandas as pd

# Import centralized data preparation utilities
from data_preparation_utils import DataPreparationUtils

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
LAYOUTLMV3_MODEL_NAME = "microsoft/layoutlmv3-base"
OUTPUT_DIR = Path("models/layoutlmv3_invoice_extractor")
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
EPOCHS = 10
VAL_SPLIT = 0.15
MAX_LENGTH = 512
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001


def create_dynamic_label_mappings(unified_samples: List[Dict]) -> Tuple[Dict[int, str], Dict[str, int], int]:
    """
    Dynamically create label mappings based on unique NER tags in the dataset.
    
    This function generates B-I-O tag mappings from the unique NER tags found in the dataset,
    ensuring proper label ID assignment for token classification.
    
    Args:
        unified_samples: List of unified sample dictionaries from DataPreparationUtils
        
    Returns:
        Tuple of (id_to_label, label_to_id, num_labels)
    """
    unique_labels = set()
    
    # Collect all unique NER tags from the dataset
    for sample in unified_samples:
        ner_tags = sample.get("ner_tags", [])
        unique_labels.update(ner_tags)
    
    # Remove 'O' if present and sort the rest
    unique_labels.discard('O')
    sorted_labels = sorted(unique_labels)
    
    # Create ID to label mapping starting with 'O' at index 0
    id_to_label = {0: "O"}
    current_id = 1
    
    # Add B- and I- tags for each unique label
    for label in sorted_labels:
        if label.startswith('B-'):
            base_label = label[2:]  # Remove 'B-' prefix
            if f"I-{base_label}" in sorted_labels:
                id_to_label[current_id] = label
                id_to_label[current_id + 1] = f"I-{base_label}"
                current_id += 2
            else:
                id_to_label[current_id] = label
                current_id += 1
        elif label.startswith('I-') and label not in id_to_label.values():
            # Handle I- tags that don't have corresponding B- tags
            id_to_label[current_id] = label
            current_id += 1
    
    label_to_id = {v: k for k, v in id_to_label.items()}
    num_labels = len(id_to_label)
    
    logger.info(f"Dynamic label mapping created with {num_labels} labels")
    logger.info(f"Label distribution: {list(id_to_label.values())}")
    
    return id_to_label, label_to_id, num_labels


class LayoutLMDataset(Dataset):
    """
    Custom Dataset for LayoutLMv3 that works with unified sample format from DataPreparationUtils.
    
    This dataset handles the preprocessing of invoice data for LayoutLMv3 training,
    ensuring proper tokenization, bounding box validation, and label alignment.
    """

    def __init__(
        self,
        data: List[Dict],
        processor: LayoutLMv3Processor,
        max_length: int,
        id_to_label_map: Dict[int, str],
    ):
        """
        Initialize LayoutLMDataset with unified sample data.
        
        Args:
            data: List of unified sample dictionaries from DataPreparationUtils
            processor: LayoutLMv3Processor instance
            max_length: Maximum sequence length for tokenization
            id_to_label_map: Mapping from label IDs to label names
        """
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.id_to_label = id_to_label_map
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        logger.info(f"LayoutLMDataset initialized with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample for training/evaluation.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed tensors for the model
        """
        item = self.data[idx]
        image_path = item["image_path"]
        words = item["tokens"]
        boxes = item["bboxes"]
        ner_tags = item["ner_tags"]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(
                f"Image file not found: {image_path}. This sample will be skipped."
            )
            raise
        except Exception as e:
            logger.error(
                f"Error loading image {image_path}: {e}. This sample will be skipped."
            )
            raise

        # Convert NER tags to label IDs
        word_labels = []
        for tag in ner_tags:
            if tag in self.label_to_id:
                word_labels.append(self.label_to_id[tag])
            else:
                logger.warning(f"Unknown label '{tag}' found, mapping to 'O'")
                word_labels.append(self.label_to_id["O"])

        # Final validation check for bounding boxes 
        # (they should already be normalized by DataPreparationUtils)
        validated_boxes = []
        for bbox in boxes:
            # Ensure all coordinates are within 0-1000 range as expected by LayoutLMv3
            if not all(0 <= coord <= 1000 for coord in bbox):
                logger.warning(f"Bounding box {bbox} is outside expected range [0-1000]. Clamping values.")
            
            validated_bbox = [max(0, min(1000, coord)) for coord in bbox]
            
            # Ensure proper bbox format (x1 > x0, y1 > y0)
            if validated_bbox[2] <= validated_bbox[0]:
                validated_bbox[2] = min(1000, validated_bbox[0] + 1)
            if validated_bbox[3] <= validated_bbox[1]:
                validated_bbox[3] = min(1000, validated_bbox[1] + 1)
            
            validated_boxes.append(validated_bbox)

        try:
            # Process with LayoutLMv3Processor
            encoding = self.processor(
                image,
                words,
                boxes=validated_boxes,
                word_labels=word_labels,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception as e:
            logger.error(f"Error processing sample {idx} with image {image_path}: {e}")
            logger.error(
                f"Boxes range: {[min(bbox) for bbox in validated_boxes]} to {[max(bbox) for bbox in validated_boxes]}"
            )
            raise

        return {
            key: encoding[key].squeeze(0) if encoding[key].dim() > 1 else encoding[key]
            for key in encoding.keys()
        }


class LayoutLMv3Learner:
    """
    Handles training and evaluation of the LayoutLMv3 model with enhanced features.
    
    This class encapsulates all training logic for LayoutLMv3, including model initialization,
    training configuration, and evaluation metrics computation.
    """

    def __init__(self, num_labels: int, output_dir: str):
        """
        Initialize the LayoutLMv3 learner.
        
        Args:
            num_labels: Number of unique labels for token classification
            output_dir: Directory to save model outputs and logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processor with apply_ocr=False since we have pre-extracted text
        self.processor = LayoutLMv3Processor.from_pretrained(
            LAYOUTLMV3_MODEL_NAME, apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            LAYOUTLMV3_MODEL_NAME, num_labels=num_labels
        )
        logger.info(f"LayoutLMv3 model initialized with {num_labels} labels.")

    def compute_metrics(
        self, p: EvalPrediction, label_id_to_name_map: Dict[int, str]
    ) -> Dict:
        """
        Compute evaluation metrics for the model predictions.
        
        This method calculates various F1 scores, accuracy, and generates detailed
        classification reports for model evaluation.
        
        Args:
            p: EvalPrediction containing predictions and labels
            label_id_to_name_map: Mapping from label IDs to label names
            
        Returns:
            Dictionary containing various evaluation metrics
        """
        predictions = p.predictions.argmax(axis=2)
        labels = p.label_ids

        # Extract true predictions and labels, ignoring -100 (padding) labels
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for p, l in zip(prediction, label):
                if l != -100:  # Ignore padding tokens
                    pred_label = label_id_to_name_map.get(p, "O")
                    true_label = label_id_to_name_map.get(l, "O")
                    true_predictions.append(pred_label)
                    true_labels.append(true_label)

        # Handle edge case where no valid predictions/labels exist
        if not true_predictions or not true_labels:
            logger.warning("No valid predictions or labels found for metric computation")
            return {
                "f1_micro": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "accuracy": 0.0,
            }

        # Calculate various F1 scores and accuracy
        f1_micro = f1_score(true_labels, true_predictions, average="micro", zero_division=0)
        f1_macro = f1_score(true_labels, true_predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(true_labels, true_predictions, average="weighted", zero_division=0)
        accuracy = accuracy_score(true_labels, true_predictions)

        # Generate detailed classification report
        try:
            report = classification_report(
                true_labels, true_predictions, output_dict=True, zero_division=0
            )
            
            # Extract per-label F1 scores (excluding 'O' tag and aggregate scores)
            per_label_f1 = {}
            for label in report:
                if (label != "O" and 
                    isinstance(report[label], dict) and 
                    label not in ["accuracy", "macro avg", "weighted avg"]):
                    per_label_f1[f"f1_{label.lower().replace('-', '_')}"] = report[label]["f1-score"]
        except Exception as e:
            logger.warning(f"Error generating classification report: {e}")
            per_label_f1 = {}

        metrics = {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
            **per_label_f1,
        }
        
        return metrics

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        label_id_to_name_map: Dict[int, str],
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
    ):
        """
        Train the LayoutLMv3 model with enhanced training configuration.
        
        This method configures and executes the training process with optimizations
        including mixed precision, gradient accumulation, and early stopping.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            label_id_to_name_map: Mapping from label IDs to label names
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            
        Returns:
            Training result object
        """
        # Determine if mixed precision training should be enabled
        use_fp16 = torch.cuda.is_available()
        
        # Calculate evaluation and save steps based on dataset size
        train_size = len(train_dataset)
        eval_steps = max(50, train_size // (batch_size * 4))  # Evaluate 4 times per epoch
        save_steps = eval_steps
        
        # Configure training arguments with optimizations
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "training_logs"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=min(50, eval_steps // 2),
            seed=42,
            dataloader_num_workers=0,  # Keep 0 for wider compatibility
            report_to="none",
            remove_unused_columns=False,
            push_to_hub=False,
            dataloader_pin_memory=False,
            fp16=use_fp16,  # Enable mixed precision if CUDA is available
            gradient_accumulation_steps=2 if batch_size < 8 else 1,  # Simulate larger batch size
            warmup_ratio=0.1,  # Warmup for better training stability
            weight_decay=0.01,  # L2 regularization
            save_total_limit=3,  # Keep only the best 3 checkpoints
        )

        # Configure early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD
        )

        # Initialize trainer with enhanced configuration
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            data_collator=None,
            compute_metrics=lambda p: self.compute_metrics(p, label_id_to_name_map),
            callbacks=[early_stopping],
        )

        logger.info("Starting enhanced training with the following configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Mixed precision (FP16): {use_fp16}")
        logger.info(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        logger.info(f"  - Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        logger.info(f"  - Evaluation steps: {eval_steps}")
        
        # Start training
        train_result = trainer.train()
        logger.info("Training finished!")

        # Save the fine-tuned model and processor
        model_save_path = self.output_dir / "fine_tuned_layoutlmv3"
        self.model.save_pretrained(model_save_path)
        self.processor.save_pretrained(model_save_path)
        logger.info(f"Fine-tuned model and processor saved to {model_save_path}")

        # CRITICAL: Save label mappings for future inference
        label_mapping_path = model_save_path / "label_mappings.json"
        with open(label_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(label_id_to_name_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Label mappings saved to {label_mapping_path}")

        return train_result


def main():
    """
    Main training function that demonstrates the enhanced LayoutLMv3 training pipeline
    with centralized data loading through DataPreparationUtils.
    
    This function orchestrates the entire training process from data loading to model saving,
    ensuring all components work together seamlessly.
    """
    # --- Configuration ---
    KAGGLE_INVOICES_DIR = "data/kaggle_invoices"
    SYNTHETIC_INVOICES_DIR = "data/synthetic_invoices"
    
    # Training hyperparameters
    VAL_SPLIT = 0.15
    MAX_LENGTH = 512
    EPOCHS = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5

    try:
        logger.info("=== Starting Enhanced LayoutLMv3 Training Pipeline ===")
        
        # Initialize centralized data preparation utilities
        logger.info("Initializing DataPreparationUtils...")
        data_utils = DataPreparationUtils()
        
        # Load raw annotation data from both sources
        logger.info("Loading Kaggle invoice dataset...")
        kaggle_samples = data_utils.load_kaggle_invoice_dataset(KAGGLE_INVOICES_DIR)
        
        logger.info("Loading synthetic invoice dataset...")
        synthetic_samples = data_utils.load_synthetic_invoice_dataset(SYNTHETIC_INVOICES_DIR)
        
        # Combine all loaded samples
        all_loaded_samples = kaggle_samples + synthetic_samples
        
        if not all_loaded_samples:
            logger.error("No samples loaded from either dataset. Please check your data paths.")
            return
        
        logger.info(f"Total samples loaded: {len(all_loaded_samples)} "
                   f"(Kaggle: {len(kaggle_samples)}, Synthetic: {len(synthetic_samples)})")
        
        # Unify dataset format using DataPreparationUtils
        logger.info("Unifying dataset format...")
        unified_samples = data_utils.unify_dataset_format(all_loaded_samples)
        
        if not unified_samples:
            logger.error("No valid samples after unification. Please check your data quality.")
            return
        
        logger.info(f"Unified samples: {len(unified_samples)}")
        
        # Create train/validation split using DataPreparationUtils
        logger.info(f"Creating train/validation split with {VAL_SPLIT:.1%} validation ratio...")
        train_data, val_data = data_utils.create_training_split(unified_samples, VAL_SPLIT)
        
        logger.info(f"Data split complete: {len(train_data)} training, {len(val_data)} validation samples")
        
        # Generate dynamic label mappings based on actual data
        logger.info("Generating dynamic label mappings...")
        id_to_label, label_to_id, num_labels = create_dynamic_label_mappings(unified_samples)
        
        # Print data statistics
        logger.info("=== Dataset Statistics ===")
        total_tokens = sum(len(sample.get("tokens", [])) for sample in unified_samples)
        logger.info(f"Total documents: {len(unified_samples)}")
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Average tokens per document: {total_tokens / len(unified_samples):.1f}")
        
        # Count label distribution
        label_counts = Counter()
        for sample in unified_samples:
            label_counts.update(sample.get("ner_tags", []))
        
        logger.info("Label distribution:")
        for label, count in label_counts.most_common():
            logger.info(f"  {label}: {count} instances")
        logger.info("=" * 50)
        
        # Initialize the LayoutLMv3 learner
        logger.info(f"Initializing LayoutLMv3Learner with {num_labels} labels...")
        learner = LayoutLMv3Learner(num_labels=num_labels, output_dir=str(OUTPUT_DIR))
        
        # Create datasets for training and validation
        logger.info("Creating training and validation datasets...")
        train_dataset = LayoutLMDataset(
            train_data, learner.processor, MAX_LENGTH, id_to_label
        )
        val_dataset = LayoutLMDataset(
            val_data, learner.processor, MAX_LENGTH, id_to_label
        )
        
        # Start training with enhanced configuration
        logger.info("Starting model training...")
        train_result = learner.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label_id_to_name_map=id_to_label,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
        )
        
        # Log training completion and results
        logger.info("=== Training Completed Successfully! ===")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        if hasattr(train_result, 'metrics') and train_result.metrics:
            logger.info("Final evaluation metrics:")
            for metric, value in train_result.metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Model saved to: {OUTPUT_DIR}/fine_tuned_layoutlmv3")
        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()