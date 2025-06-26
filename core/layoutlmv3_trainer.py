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
    EvalPrediction
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
from collections import Counter
import pandas as pd 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
# Updated model name for LayoutLMv3
LAYOUTLMV3_MODEL_NAME = "microsoft/layoutlmv3-base" 
OUTPUT_DIR = "models/layoutlmv3_invoice_extractor" # Updated output directory for v3
BATCH_SIZE = 4 # Reduced for potentially higher GPU memory usage
LEARNING_RATE = 5e-5
EPOCHS = 10
VAL_SPLIT = 0.15
MAX_LENGTH = 512 # Max sequence length for tokenization


def normalize_bbox(bbox, width, height):
    """
    Normalize bounding box coordinates to 0-1000 range as expected by LayoutLMv3.
    
    Args:
        bbox: List of [x0, y0, x1, y1] coordinates
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Normalized bbox coordinates in 0-1000 range
    """
    x0, y0, x1, y1 = bbox
    
    # Normalize to 0-1 range first
    x0_norm = x0 / width
    y0_norm = y0 / height
    x1_norm = x1 / width
    y1_norm = y1 / height
    
    # Scale to 0-1000 range and ensure within bounds
    x0_scaled = max(0, min(1000, int(x0_norm * 1000)))
    y0_scaled = max(0, min(1000, int(y0_norm * 1000)))
    x1_scaled = max(0, min(1000, int(x1_norm * 1000)))
    y1_scaled = max(0, min(1000, int(y1_norm * 1000)))
    
    # Ensure x1 > x0 and y1 > y0 (minimum 1 pixel difference)
    if x1_scaled <= x0_scaled:
        x1_scaled = min(1000, x0_scaled + 1)
    if y1_scaled <= y0_scaled:
        y1_scaled = min(1000, y0_scaled + 1)
    
    return [x0_scaled, y0_scaled, x1_scaled, y1_scaled]


def validate_and_fix_bbox(bbox, width, height):
    """
    Validate and fix bounding box coordinates.
    
    Args:
        bbox: List of [x0, y0, x1, y1] coordinates
        width: Image width
        height: Image height
    
    Returns:
        Fixed bbox coordinates
    """
    x0, y0, x1, y1 = bbox
    
    # Ensure coordinates are within image bounds
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    
    # Ensure x1 > x0 and y1 > y0
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    
    return [x0, y0, x1, y1]


class KaggleInvoiceDataLoader:
    """
    Data loader for Kaggle invoice annotations.
    Handles loading, parsing, inferring image paths,
    and translating NER tags to a unified scheme.
    """
    
    def __init__(self, annotations_dir: str, images_dir: str, output_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Define the unified target labels for the model ---
        # These are the granular labels we want the model to learn
        self.base_labels = [
            "INVOICE_NUM", "DATE", "DUE_DATE", "VENDOR_NAME", 
            "VENDOR_ADDRESS", "CUSTOMER_NAME", "CUSTOMER_ADDRESS", 
            "ITEM_DESCRIPTION", "QUANTITY", "UNIT_PRICE", # UNIT_PRICE (from B-PRICE)
            "LINE_TOTAL", "SUBTOTAL", "TAX_AMOUNT", # TAX_AMOUNT (from B-TAX)
            "DISCOUNT_AMOUNT", "TOTAL_AMOUNT", "CURRENCY", "HEADER",
            "ACCOUNT_NUM" # Added based on your log output
        ]
        
        # --- Define a translation map for incoming JSON ner_tags to unified base_labels ---
        # This maps inconsistent labels in your JSONs to the standard ones for training
        self.ner_tag_translation_map = {
            "ITEM_DESC": "ITEM_DESCRIPTION",
            "QTY": "QUANTITY",
            "PRICE": "UNIT_PRICE",
            "TOTAL": "LINE_TOTAL", # Assuming B-TOTAL usually means line total, not grand total
            "VENDOR": "VENDOR_NAME",
            "TAX": "TAX_AMOUNT",
            "TOTAL_AMT": "TOTAL_AMOUNT", # Ensure this is explicitly mapped if it exists
            "ACCOUNT_NUM": "ACCOUNT_NUM" # Ensure consistent mapping for new label
            # Add any other observed discrepancies if they arise
        }

        # Create the comprehensive ID to label map (O, B-, I- prefixes)
        self.id_to_label = {0: "O"} # 0 is always for "outside"
        for i, label in enumerate(self.base_labels):
            self.id_to_label[len(self.id_to_label)] = f"B-{label}"
            self.id_to_label[len(self.id_to_label)] = f"I-{label}"
            
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.num_labels = len(self.id_to_label)
        logger.info(f"Initialized with {self.num_labels} labels: {self.id_to_label}")

    def load_annotations(self) -> List[Dict]:
        """
        Loads JSON annotation files, infers image_path if missing,
        and translates/renames keys to match LayoutLMDataset's expectations.
        """
        all_data = []
        json_files = list(self.annotations_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON annotation files found in {self.annotations_dir}")
            return []

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # --- Step 1: Handle/Infer 'image_path' ---
                    image_full_path = None
                    if "image_path" in data and data["image_path"]:
                        # If image_path exists in JSON, construct absolute path
                        image_filename = Path(data["image_path"]).name 
                        potential_path = self.images_dir / image_filename
                        if potential_path.exists():
                            image_full_path = str(potential_path)
                        else:
                            logger.error(f"Image specified in JSON '{data['image_path']}' ({potential_path}) does not exist for {json_path.name}. Attempting inference.")
                    
                    if image_full_path is None: # If not found via JSON path, or if JSON path was missing/invalid
                        # Infer image filename from JSON filename (e.g., batch1-0001.json -> batch1-0001.jpg)
                        image_filename_stem = json_path.stem
                        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]: # Check common extensions
                            potential_image_path = self.images_dir / f"{image_filename_stem}{ext}"
                            if potential_image_path.exists():
                                image_full_path = str(potential_image_path)
                                logger.info(f"Inferred image_path for {json_path.name}: {image_full_path}")
                                break
                    
                    if image_full_path is None:
                        logger.error(f"Could not find any corresponding image for {json_path.name} in {self.images_dir}. Skipping annotation.")
                        continue # Skip this data entry if image cannot be found

                    # Load image to get dimensions for bbox normalization
                    try:
                        with Image.open(image_full_path) as img:
                            img_width, img_height = img.size
                    except Exception as e:
                        logger.error(f"Error loading image {image_full_path} to get dimensions: {e}. Skipping.")
                        continue

                    # --- Step 2: Translate and Rename keys for LayoutLMDataset expectations ---
                    incoming_words = data.get("words", [])
                    incoming_boxes = data.get("boxes", [])
                    incoming_ner_tags = data.get("ner_tags", [])

                    # Validate and normalize bounding boxes
                    normalized_boxes = []
                    for i, bbox in enumerate(incoming_boxes):
                        try:
                            # Validate and fix bbox first
                            fixed_bbox = validate_and_fix_bbox(bbox, img_width, img_height)
                            # Then normalize to 0-1000 range
                            normalized_bbox = normalize_bbox(fixed_bbox, img_width, img_height)
                            normalized_boxes.append(normalized_bbox)
                        except Exception as e:
                            logger.warning(f"Error processing bbox {bbox} for word '{incoming_words[i] if i < len(incoming_words) else 'N/A'}' in {json_path.name}: {e}. Using default bbox.")
                            # Use a default small bbox in the top-left corner
                            normalized_boxes.append([0, 0, 50, 50])

                    # Translate incoming NER tags to our unified base_labels
                    translated_ner_tags = []
                    for tag in incoming_ner_tags:
                        if tag == "O":
                            translated_ner_tags.append("O")
                        elif tag.startswith("B-") or tag.startswith("I-"):
                            prefix = tag[0] # 'B' or 'I'
                            original_label = tag[2:] # e.g., 'PRICE'
                            
                            # Check if original_label is in our translation map
                            if original_label in self.ner_tag_translation_map:
                                translated_label = self.ner_tag_translation_map[original_label]
                                translated_ner_tags.append(f"{prefix}-{translated_label}")
                            elif original_label in self.base_labels: # Already a direct match
                                translated_ner_tags.append(tag)
                            else:
                                logger.warning(f"Unknown NER tag '{tag}' in {json_path.name}. Mapping to 'O'.")
                                translated_ner_tags.append("O") # Default to 'O' for unknown tags
                        else:
                            logger.warning(f"Malformed NER tag '{tag}' in {json_path.name}. Mapping to 'O'.")
                            translated_ner_tags.append("O") # Default to 'O' for malformed tags

                    processed_item = {
                        "image_path": image_full_path,
                        "tokens": incoming_words, # Using 'words' from your JSON, renamed to 'tokens'
                        "bboxes": normalized_boxes, # Using normalized boxes
                        "ner_tags": translated_ner_tags, # Use the translated NER tags
                        "image_width": img_width,
                        "image_height": img_height
                    }

                    # --- Step 3: Basic validation ---
                    if not processed_item["tokens"] or not processed_item["bboxes"] or not processed_item["ner_tags"]:
                        logger.warning(f"Skipping {json_path.name}: Missing 'tokens', 'bboxes', or 'ner_tags' data after processing.")
                        continue
                    if not (len(processed_item["tokens"]) == len(processed_item["bboxes"]) == len(processed_item["ner_tags"])):
                        logger.warning(f"Skipping {json_path.name}: Mismatched lengths in 'tokens' ({len(processed_item['tokens'])}), 'bboxes' ({len(processed_item['bboxes'])}), or 'ner_tags' ({len(processed_item['ner_tags'])}) after processing.")
                        continue

                    all_data.append(processed_item)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {json_path}: {e}. Skipping.")
            except Exception as e:
                logger.error(f"Error processing annotation file {json_path}: {e}. Skipping.")
        return all_data

    def get_train_val_split(self, data: List[Dict], val_split: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        if not data:
            return [], []
        train_data, val_data = train_test_split(data, test_size=val_split, random_state=42)
        logger.info(f"Data split: {len(train_data)} training samples, {len(val_data)} validation samples.")
        return train_data, val_data

    def print_data_statistics(self, data: List[Dict]):
        total_tokens = 0
        label_counts = Counter()
        bbox_stats = {'min_vals': [], 'max_vals': []}
        
        for item in data:
            # Use 'ner_tags' as they are now translated
            label_counts.update(item.get("ner_tags", []))
            total_tokens += len(item.get("tokens", [])) # Use 'tokens'
            
            # Collect bbox statistics for validation
            for bbox in item.get("bboxes", []):
                bbox_stats['min_vals'].append(min(bbox))
                bbox_stats['max_vals'].append(max(bbox))
        
        logger.info("\n--- Data Statistics ---")
        logger.info(f"Total documents loaded: {len(data)}")
        logger.info(f"Total tokens processed: {total_tokens}")
        if bbox_stats['min_vals'] and bbox_stats['max_vals']:
            logger.info(f"Bbox coordinate range: {min(bbox_stats['min_vals'])} to {max(bbox_stats['max_vals'])}")
        logger.info("Label distribution (B- and I- tags):")
        for label, count in label_counts.most_common():
            logger.info(f"  {label}: {count} instances")
        logger.info("-----------------------")


class LayoutLMDataset(Dataset):
    """Custom Dataset for LayoutLMv3."""

    def __init__(self, data: List[Dict], processor: LayoutLMv3Processor, max_length: int, id_to_label_map: Dict[int, str]):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.id_to_label = id_to_label_map
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        logger.info(f"LayoutLMDataset initialized with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        image_path = item["image_path"]
        words = item["tokens"] 
        boxes = item["bboxes"] 
        ner_tags = item["ner_tags"]

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}. This sample will be skipped.")
            raise
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}. This sample will be skipped.")
            raise

        word_labels = [self.label_to_id[tag] for tag in ner_tags]

        # Validate bbox coordinates one more time before processing
        validated_boxes = []
        for bbox in boxes:
            # Ensure all coordinates are within 0-1000 range
            validated_bbox = [max(0, min(1000, coord)) for coord in bbox]
            # Ensure proper bbox format (x1 > x0, y1 > y0)
            if validated_bbox[2] <= validated_bbox[0]:
                validated_bbox[2] = min(1000, validated_bbox[0] + 1)
            if validated_bbox[3] <= validated_bbox[1]:
                validated_bbox[3] = min(1000, validated_bbox[1] + 1)
            validated_boxes.append(validated_bbox)

        try:
            # Process with validated boxes
            encoding = self.processor(
                image,
                words,
                boxes=validated_boxes,
                word_labels=word_labels,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Error processing sample {idx} with image {image_path}: {e}")
            logger.error(f"Boxes range: {[min(bbox) for bbox in validated_boxes]} to {[max(bbox) for bbox in validated_boxes]}")
            raise
        
        return {
            key: encoding[key].squeeze(0) if encoding[key].dim() > 1 else encoding[key]
            for key in encoding.keys()
        }


class LayoutLMv3Learner:
    """Handles training and evaluation of the LayoutLMv3 model."""

    def __init__(self, num_labels: int, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor with apply_ocr=False
        self.processor = LayoutLMv3Processor.from_pretrained(LAYOUTLMV3_MODEL_NAME, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            LAYOUTLMV3_MODEL_NAME,
            num_labels=num_labels
        )
        logger.info(f"LayoutLMv3 model initialized with {num_labels} labels.")

    def compute_metrics(self, p: EvalPrediction, label_id_to_name_map: Dict[int, str]) -> Dict:
        predictions = p.predictions.argmax(axis=2)
        labels = p.label_ids

        true_predictions = [
            label_id_to_name_map[p] for prediction, label in zip(predictions, labels)
            for p, l in zip(prediction, label) if l != -100
        ]
        true_labels = [
            label_id_to_name_map[l] for prediction, label in zip(predictions, labels)
            for p, l in zip(prediction, label) if l != -100
        ]

        f1_micro = f1_score(true_labels, true_predictions, average="micro")
        f1_macro = f1_score(true_labels, true_predictions, average="macro")
        f1_weighted = f1_score(true_labels, true_predictions, average="weighted")
        accuracy = accuracy_score(true_labels, true_predictions)

        # Ensure zero_division=0 to prevent warnings/errors if a label has no true samples
        report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
        
        per_label_f1 = {
            label: report[label]['f1-score']
            for label in report if label != 'O' and isinstance(report[label], dict)
        }

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
            **{f"f1_{k.lower()}": v for k, v in per_label_f1.items()}
        }

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        label_id_to_name_map: Dict[int, str],
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE
    ):
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "training_logs"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            seed=42,
            dataloader_num_workers=0,
            report_to="none",
            remove_unused_columns=False,
            push_to_hub=False,
            dataloader_pin_memory=False,  # Disable pin_memory to avoid warnings
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,  # Use processing_class instead of tokenizer
            data_collator=None, 
            compute_metrics=lambda p: self.compute_metrics(p, label_id_to_name_map),
        )
        
        logger.info("Starting training...")
        train_result = trainer.train()
        logger.info("Training finished!")
        
        self.model.save_pretrained(self.output_dir / "fine_tuned_layoutlmv3")
        self.processor.save_pretrained(self.output_dir / "fine_tuned_layoutlmv3")
        logger.info(f"Fine-tuned model and processor saved to {self.output_dir / 'fine_tuned_layoutlmv3'}")
        
        return train_result


def main():
    # --- Configuration ---
    KAGGLE_ANNOTATIONS_DIR = "data/kaggle_invoices/annotations"
    KAGGLE_IMAGES_DIR = "data/kaggle_invoices/images"
    SYNTHETIC_ANNOTATIONS_DIR = "data/synthetic_invoices/annotations"
    SYNTHETIC_IMAGES_DIR = "data/synthetic_invoices/images"
    OUTPUT_DIR = "models/layoutlmv3_invoice_extractor"

    VAL_SPLIT = 0.15
    MAX_LENGTH = 512
    EPOCHS = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5

    try:
        kaggle_data_loader = KaggleInvoiceDataLoader(KAGGLE_ANNOTATIONS_DIR, KAGGLE_IMAGES_DIR, OUTPUT_DIR)
        synthetic_data_loader = KaggleInvoiceDataLoader(SYNTHETIC_ANNOTATIONS_DIR, SYNTHETIC_IMAGES_DIR, OUTPUT_DIR)

        kaggle_data = kaggle_data_loader.load_annotations()
        synthetic_data = synthetic_data_loader.load_annotations()
        
        all_data = kaggle_data + synthetic_data
        
        if not all_data:
            logger.error("No valid data found. Please check your annotation files and images paths.")
            return
            
        kaggle_data_loader.print_data_statistics(all_data)
        
        train_data, val_data = kaggle_data_loader.get_train_val_split(all_data, VAL_SPLIT)
        
        learner = LayoutLMv3Learner(
            num_labels=kaggle_data_loader.num_labels,
            output_dir=OUTPUT_DIR
        )
        
        train_dataset = LayoutLMDataset(train_data, learner.processor, MAX_LENGTH, kaggle_data_loader.id_to_label)
        val_dataset = LayoutLMDataset(val_data, learner.processor, MAX_LENGTH, kaggle_data_loader.id_to_label)
        
        train_result = learner.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label_id_to_name_map=kaggle_data_loader.id_to_label,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()