import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from PIL import Image
from datasets import load_dataset
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreparationUtils:
    """Essential utilities for loading and processing datasets for IDP training."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.psc_data = None

    def load_psc_data(
        self,
        url: str = "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/pscs.json",
    ) -> Dict:
        """Load and structure PSC classification data from GitHub."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            raw_data = response.json()

            # Create comprehensive mapping structure
            psc_mapping = {}
            category_mapping = {}
            portfolio_mapping = {}

            for item in raw_data:
                psc_code = item.get("PSC", "")  # Corrected key to "PSC"
                short_name = item.get("shortName", "")
                spend_category = item.get("spendCategoryTitle", "")
                portfolio_group = item.get("portfolioGroup", "")

                psc_mapping[psc_code] = {
                    "psc": psc_code,
                    "shortName": short_name,
                    "spendCategoryTitle": spend_category,
                    "portfolioGroup": portfolio_group,
                }

                # Create reverse mappings for classification
                if spend_category:
                    if spend_category not in category_mapping:
                        category_mapping[spend_category] = []
                    category_mapping[spend_category].append(psc_code)

                if portfolio_group:
                    if portfolio_group not in portfolio_mapping:
                        portfolio_mapping[portfolio_group] = []
                    portfolio_mapping[portfolio_group].append(psc_code)

            self.psc_data = {
                "psc_mapping": psc_mapping,
                "category_mapping": category_mapping,
                "portfolio_mapping": portfolio_mapping,
                "all_pscs": list(psc_mapping.keys()),
                "all_categories": list(category_mapping.keys()),
                "all_portfolios": list(portfolio_mapping.keys()),
            }

            print(
                f"Loaded {len(psc_mapping)} PSC codes across {len(category_mapping)} categories"
            )
            return self.psc_data

        except Exception as e:
            print(f"Error loading PSC data: {e}")
            return {}

    def _load_json_annotations_with_images(
        self, dataset_path: str, dataset_name: str
    ) -> List[Dict]:
        """Helper to load JSON annotations and find corresponding images."""
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            print(
                f"Dataset directory {dataset_path} for {dataset_name} does not exist."
            )
            return []

        images_dir = dataset_dir / "images"
        annotations_dir = dataset_dir / "annotations"

        if not images_dir.exists() or not annotations_dir.exists():
            print(
                f"Required subdirectories 'images' and 'annotations' not found in {dataset_path} for {dataset_name}."
            )
            return []

        processed_samples = []
        annotation_files = list(annotations_dir.glob("*.json"))

        logger.info(
            f"Found {len(annotation_files)} annotation files for {dataset_name}."
        )

        for ann_file in annotation_files:
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)

                # Find corresponding image file (assuming same stem, different extensions)
                image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
                image_path = None
                for ext in image_extensions:
                    potential_img_path = images_dir / (ann_file.stem + ext)
                    if potential_img_path.exists():
                        image_path = potential_img_path
                        break

                if not image_path:
                    logger.warning(
                        f"No corresponding image found for annotation {ann_file.name} in {dataset_name}. Skipping."
                    )
                    continue

                # Extract words, boxes, and ner_tags. Your Kaggle format has these directly.
                words = annotation_data.get("words")
                boxes = annotation_data.get("boxes")
                ner_tags = annotation_data.get("ner_tags")

                if not words or not boxes or not ner_tags:
                    logger.warning(
                        f"Skipping {ann_file.name} in {dataset_name}: missing 'words', 'boxes', or 'ner_tags'."
                    )
                    continue

                # Basic consistency check
                if not (len(words) == len(boxes) == len(ner_tags)):
                    logger.warning(
                        f"Skipping {ann_file.name} in {dataset_name}: inconsistent lengths of words, boxes, or ner_tags."
                    )
                    continue

                processed_sample = {
                    "image_path": str(image_path),
                    "words": words,
                    "boxes": boxes,
                    "ner_tags": ner_tags,
                    "relations": annotation_data.get(
                        "relations", []
                    ),  # Include relations if present
                    "dataset_source": dataset_name,  # Track original dataset
                    "annotation_data": annotation_data,  # Keep original data for reference if needed
                }

                processed_samples.append(processed_sample)

            except Exception as e:
                logger.error(
                    f"Error processing annotation file {ann_file.name} in {dataset_name}: {e}"
                )
                continue

        logger.info(f"Loaded {len(processed_samples)} samples from {dataset_name}.")
        return processed_samples

    def load_kaggle_invoice_dataset(self, dataset_path: str) -> List[Dict]:
        """Load Kaggle invoice dataset (images and JSON annotations)."""
        return self._load_json_annotations_with_images(dataset_path, "KAGGLE_INVOICE")

    def load_synthetic_invoice_dataset(self, dataset_path: str) -> List[Dict]:
        """Load synthetic invoice dataset (images and JSON annotations), assuming same format as Kaggle."""
        return self._load_json_annotations_with_images(
            dataset_path, "SYNTHETIC_INVOICE"
        )

    def load_cord_dataset(self, split: str = "train") -> List[Dict]:
        """Load CORD dataset via Hugging Face datasets library."""
        try:
            dataset = load_dataset("naver-clova-ix/cord-v2", split=split)
            processed_samples = []

            for sample in dataset:
                image = sample["image"]  # PIL Image object
                ground_truth = sample["ground_truth"]

                words = []
                boxes = []
                ner_tags = []

                for line in ground_truth["valid_line"]:
                    for word_info in line.get("words", []):
                        words.append(word_info.get("text", ""))
                        quad = word_info.get("quad", {})
                        if quad:
                            x_coords = [quad.get(f"x{i}", 0) for i in range(1, 5)]
                            y_coords = [quad.get(f"y{i}", 0) for i in range(1, 5)]
                            bbox = [
                                min(x_coords),
                                min(y_coords),
                                max(x_coords),
                                max(y_coords),
                            ]
                            boxes.append(bbox)
                        else:
                            boxes.append([0, 0, 0, 0])  # Default if box is missing

                        ner_tags.append(word_info.get("label", "O"))

                # Check for empty samples after parsing (e.g., if no valid words)
                if not words:
                    logger.warning(
                        f"Skipping CORD sample with no words in split {split}."
                    )
                    continue

                processed_samples.append(
                    {
                        "image": image,  # Keep PIL image object for CORD
                        "words": words,
                        "boxes": boxes,
                        "ner_tags": ner_tags,
                        "relations": [],  # CORD doesn't have explicit relations
                        "dataset_source": "CORD",
                    }
                )

            logger.info(
                f"Loaded {len(processed_samples)} samples from CORD {split} split."
            )
            return processed_samples

        except Exception as e:
            logger.error(f"Error loading CORD dataset: {e}")
            return []

    def load_funsd_dataset(self, funsd_path: str) -> List[Dict]:
        """Load FUNSD dataset from local directory."""
        try:
            funsd_dir = Path(funsd_path)
            annotation_files = list(
                funsd_dir.glob("**/annotations/*.json")
            )  # Ensure correct glob pattern
            processed_samples = []

            for ann_file in annotation_files:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)

                # Find corresponding image
                img_name = ann_file.stem + ".png"
                img_path = (
                    ann_file.parent.parent / "images" / ann_file.parent.name / img_name
                )  # Adjust path to images

                if not img_path.exists():
                    logger.warning(
                        f"Image {img_path} not found for FUNSD annotation {ann_file.name}. Skipping."
                    )
                    continue

                words = []
                boxes = []
                ner_tags = []
                relations = []

                # Parse FUNSD annotation format - this is entity-level, will need conversion to token-level
                for form_entry in annotation_data.get("form", []):
                    entity_label = form_entry.get("label", "O")
                    entity_words = form_entry.get("words", [])

                    for i, word_info in enumerate(entity_words):
                        words.append(word_info.get("text", ""))
                        boxes.append(word_info.get("box", [0, 0, 0, 0]))

                        # Apply B-I-O tagging based on the entity_label
                        if i == 0:
                            ner_tags.append(
                                f"B-{entity_label}" if entity_label != "O" else "O"
                            )
                        else:
                            ner_tags.append(
                                f"I-{entity_label}" if entity_label != "O" else "O"
                            )

                    # Extract linking information
                    for link in form_entry.get("linking", []):
                        relations.append(link)

                if not words:
                    logger.warning(
                        f"Skipping FUNSD annotation {ann_file.name}: no words extracted."
                    )
                    continue

                processed_samples.append(
                    {
                        "image_path": str(img_path),
                        "words": words,
                        "boxes": boxes,
                        "ner_tags": ner_tags,
                        "relations": relations,
                        "dataset_source": "FUNSD",
                    }
                )

            logger.info(f"Loaded {len(processed_samples)} samples from FUNSD.")
            return processed_samples

        except Exception as e:
            logger.error(f"Error loading FUNSD dataset: {e}")
            return []

    def load_sroie_dataset(self, sroie_path: str) -> List[Dict]:
        """Load SROIE dataset from local directory with original annotations."""
        try:
            sroie_dir = Path(sroie_path)
            processed_samples = []

            # SROIE has separate text and entity files
            # Look for .txt files that are NOT _box.txt
            text_files = [
                f for f in sroie_dir.glob("**/*.txt") if not f.stem.endswith("_box")
            ]

            for txt_file in text_files:
                # Load text and boxes from the .txt file directly
                # SROIE format: x1,y1,x2,y2,x3,y3,x4,y4,text
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                words = []
                boxes = []
                # SROIE's main file doesn't have NER tags; these would be derived from _entities.txt
                # For basic NER, we default to "O" or you'd need a separate parser for _entities.txt
                ner_tags = ["O"] * len(lines)  # Default labels to "O" for all tokens

                for line in lines:
                    parts = line.strip().split(
                        ",", 8
                    )  # Split up to 8 times for coords, rest is text
                    if len(parts) == 9:  # x1,y1,x2,y2,x3,y3,x4,y4,text
                        # Extract bounding box (convert from quad to bbox)
                        try:
                            coords = [int(x) for x in parts[:8]]
                            x_coords = coords[::2]
                            y_coords = coords[1::2]
                            bbox = [
                                min(x_coords),
                                min(y_coords),
                                max(x_coords),
                                max(y_coords),
                            ]
                        except ValueError:
                            logger.warning(
                                f"Invalid coordinates in SROIE file {txt_file.name}: {parts[:8]}. Skipping line."
                            )
                            continue

                        text = parts[8].strip()
                        words.append(text)
                        boxes.append(bbox)
                    elif (
                        len(parts) > 1
                    ):  # Handle cases where a line might just be text, no box
                        words.append(parts[-1].strip())
                        boxes.append([0, 0, 0, 0])  # Placeholder box
                    else:
                        logger.warning(
                            f"Skipping malformed line in SROIE file {txt_file.name}: {line.strip()}"
                        )

                # Find corresponding image
                img_name = txt_file.stem + ".jpg"  # SROIE images are typically JPG
                img_path = txt_file.parent / img_name

                if not img_path.exists():
                    logger.warning(
                        f"Image {img_path} not found for SROIE text file {txt_file.name}. Skipping."
                    )
                    continue

                if not words:
                    logger.warning(
                        f"Skipping SROIE file {txt_file.name}: no words extracted."
                    )
                    continue

                processed_samples.append(
                    {
                        "image_path": str(img_path),
                        "words": words,
                        "boxes": boxes,
                        "ner_tags": ner_tags,  # These are mostly "O" unless _entities.txt is parsed
                        "relations": [],
                        "dataset_source": "SROIE",
                    }
                )

            logger.info(f"Loaded {len(processed_samples)} samples from SROIE.")
            return processed_samples

        except Exception as e:
            logger.error(f"Error loading SROIE dataset: {e}")
            return []

    def unify_dataset_format(self, samples: List[Dict]) -> List[Dict]:
        """Convert all dataset samples to unified format for LayoutLMv2 training.
        Ensures 'image_path', 'words', 'boxes', 'ner_tags' are present and clean.
        """
        unified_samples = []
        for sample in samples:
            # Ensure essential fields are present with defaults
            image_path = sample.get("image_path")
            if not image_path and "image" in sample:  # Handle CORD's PIL Image object
                # If image is a PIL object, save it temporarily or handle in Dataset
                # For LayoutLMv2Processor, it expects a path or the PIL object directly in __getitem__
                # For unification, let's prioritize path for consistency.
                # This is a simplification; for production, consider saving PIL to temp file.
                logger.warning(
                    f"PIL Image object encountered from {sample.get('dataset_source')}. LayoutLMDataset needs a path. Skipping for now or enhance handling."
                )
                continue  # Skip samples with only PIL image for now if no path provided

            words = sample.get("words", [])
            boxes = sample.get("boxes", [])
            ner_tags = sample.get("ner_tags", [])

            # Basic validation after unification
            if not (
                words
                and boxes
                and ner_tags
                and len(words) == len(boxes) == len(ner_tags)
            ):
                logger.warning(
                    f"Skipping malformed sample from {sample.get('dataset_source')}: inconsistent or missing words/boxes/ner_tags."
                )
                continue

            # Ensure boxes are integers
            cleaned_boxes = []
            for box in boxes:
                if len(box) == 4 and all(isinstance(c, (int, float)) for c in box):
                    cleaned_boxes.append([int(coord) for coord in box])
                else:
                    logger.warning(
                        f"Invalid box format found: {box}. Using default [0,0,0,0]."
                    )
                    cleaned_boxes.append([0, 0, 0, 0])  # Fallback for malformed boxes

            unified_sample = {
                "image_path": image_path,
                "words": words,
                "boxes": cleaned_boxes,
                "ner_tags": ner_tags,  # These are the token-level string tags
                "relations": sample.get("relations", []),
                "dataset_source": sample.get("dataset_source", "unknown"),
            }
            unified_samples.append(unified_sample)

        return unified_samples

    def get_psc_by_description(self, description: str) -> Optional[Dict]:
        """Find best matching PSC based on item description.
        This is a simple keyword matching for quick demos.
        For actual inference, the trained PSC model should be used.
        """
        if not self.psc_data or not self.psc_data.get("psc_mapping"):
            print(
                "PSC data not loaded or mapping not available for description lookup."
            )
            return None

        description_lower = description.lower()
        best_match = None
        max_score = 0

        for psc_code, psc_info in self.psc_data["psc_mapping"].items():
            short_name = psc_info["shortName"].lower()
            category = psc_info["spendCategoryTitle"].lower()
            long_name = psc_info.get("longName", "").lower()

            score = 0
            # Prioritize exact short name match
            if short_name == description_lower:
                score += 10
            elif short_name in description_lower:
                score += 5
            elif description_lower in short_name:
                score += 4

            # Fuzzy match with words
            desc_words = set(description_lower.split())
            short_name_words = set(short_name.split())
            category_words = set(category.split())
            long_name_words = set(long_name.split())

            score += len(desc_words.intersection(short_name_words)) * 2
            score += len(desc_words.intersection(category_words)) * 1
            score += len(desc_words.intersection(long_name_words)) * 0.5

            if score > max_score:
                max_score = score
                best_match = psc_info

            # If perfect match found, no need to continue
            if max_score >= 10:  # Perfect match for short name
                break

        return (
            best_match if best_match and max_score > 0 else None
        )  # Return None if no reasonable match

    def create_training_split(
        self, samples: List[Dict], train_ratio: float = 0.8
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into training and validation sets."""
        # Use sklearn's train_test_split for more robust splitting
        if len(samples) < 2:
            logger.warning(
                "Not enough samples for splitting. Returning all samples as train, empty as validation."
            )
            return samples, []

        train_samples, val_samples = train_test_split(
            samples, test_size=(1 - train_ratio), random_state=42, shuffle=True
        )

        logger.info(
            f"Created training split: {len(train_samples)} train, {len(val_samples)} validation"
        )
        return train_samples, val_samples

    def save_processed_data(self, data: Any, filename: str):
        """Save processed data to disk."""
        filepath = self.data_dir / filename

        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, (dict, list)):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            # For other types, use pickle
            import pickle

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

        print(f"Data saved to {filepath}")

    def load_processed_data(self, filename: str):
        """Load processed data from disk."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        if filename.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            import pickle

            with open(filepath, "rb") as f:
                return pickle.load(f)


# Usage example (moved outside class, often into a dedicated script or main entry)
if __name__ == "__main__":
    data_prep = DataPreparationUtils()
    psc_data = data_prep.load_psc_data()
    kaggle_invoice_samples = data_prep.load_kaggle_invoice_dataset(
        "data/kaggle_invoices"
    )
    synthetic_invoice_samples = data_prep.load_synthetic_invoice_dataset(
        "data/synthetic_invoices"
    )
    all_samples = kaggle_invoice_samples + synthetic_invoice_samples
    unified_samples = data_prep.unify_dataset_format(all_samples)
    train_samples, val_samples = data_prep.create_training_split(unified_samples)
    data_prep.save_processed_data(train_samples, "train_samples.json")
    data_prep.save_processed_data(val_samples, "val_samples.json")
    print("Data preparation utilities ready for use.")
