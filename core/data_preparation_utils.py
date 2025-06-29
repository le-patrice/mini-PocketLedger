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
        self.unspsc_data = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.MAX_IMAGE_SIDE = 1000
        
        # Enhanced NER tag standardization map
        self.ner_tag_standardization_map = {
            '0': 'O', '1': 'O', '2': 'O', '3': 'O', '4': 'O', '5': 'O', 
            '7': 'O', '9': 'O', '11': 'O', '13': 'O', '15': 'O', 
            '17': 'B-QUANTITY', '19': 'B-TOTAL', '21': 'O', '23': 'O', 
            '25': 'O', '27': 'O', '31': 'O',
            'ITEM_DESC': 'B-DESCRIPTION',
            'QTY': 'B-QUANTITY',
            'TOTAL': 'B-TOTAL',
            'DATE': 'B-DATE',
            'PRICE': 'B-PRICE',
            'O': 'O', 'B': 'B', 'I': 'I'
        }
        self._logged_unmapped_tags = set()

    def load_unspsc_data(
        self,
        url: str = "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/unspsc-codes.csv",
        filepath: str = "data/data-unspsc-codes.csv"
    ) -> pd.DataFrame:
        """Load and process UNSPSC classification data from GitHub.
        
        Args:
            url: URL to download UNSPSC CSV data
            filepath: Local filepath to save/load UNSPSC data
            
        Returns:
            Processed DataFrame with searchable_text and class_name columns
        """
        try:
            # Try to load from local file first
            local_path = Path(filepath)
            if local_path.exists():
                logger.info(f"Loading UNSPSC data from local file: {filepath}")
                df = pd.read_csv(local_path)
            else:
                logger.info(f"Downloading UNSPSC data from: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save to local file
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                df = pd.read_csv(local_path)
            
            # Detect and map column names
            df = self._detect_column_mappings(df)
            
            # Process the UNSPSC data
            self.unspsc_data = self._process_unspsc_data(df)
            
            logger.info(f"Loaded and processed {len(self.unspsc_data)} UNSPSC records")
            logger.info(f"Unique classes available: {self.unspsc_data['class_name'].nunique()}")
            
            return self.unspsc_data

        except Exception as e:
            logger.error(f"Error loading UNSPSC data: {e}", exc_info=True)
            raise

    def _detect_column_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligently detect and standardize UNSPSC column names.
        
        Args:
            df: Raw UNSPSC DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Define mapping patterns for common UNSPSC column variations
        column_mappings = {
            'segment_code': ['segment code', 'segment_code', 'segmentcode'],
            'segment_name': ['segment name', 'segment_name', 'segmentname', 'segment title', 'segment_title'],
            'family_code': ['family code', 'family_code', 'familycode'],
            'family_name': ['family name', 'family_name', 'familyname', 'family title', 'family_title'],
            'class_code': ['class code', 'class_code', 'classcode'],
            'class_name': ['class name', 'class_name', 'classname', 'class title', 'class_title'],
            'commodity_code': ['commodity code', 'commodity_code', 'commoditycode', 'unspsc code', 'unspsc_code'],
            'commodity_name': ['commodity name', 'commodity_name', 'commodityname', 'commodity title', 'commodity_title', 'title', 'description']
        }
        
        # Create a case-insensitive column mapping
        df_columns_lower = [col.lower().strip() for col in df.columns]
        standardized_df = df.copy()
        
        for standard_name, variations in column_mappings.items():
            for variation in variations:
                if variation.lower() in df_columns_lower:
                    original_col = df.columns[df_columns_lower.index(variation.lower())]
                    if original_col not in standardized_df.columns:
                        continue
                    standardized_df = standardized_df.rename(columns={original_col: standard_name})
                    logger.info(f"Mapped column '{original_col}' to '{standard_name}'")
                    break
        
        return standardized_df

    def _process_unspsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process UNSPSC data to create searchable text and ensure class_name column.
        
        Args:
            df: UNSPSC DataFrame with detected column mappings
            
        Returns:
            Processed DataFrame with searchable_text and class_name columns
        """
        processed_df = df.copy()
        
        # Create searchable_text by concatenating relevant descriptive fields
        # Priority: commodity_name/title, class_name, family_name, segment_name
        text_components = []
        
        for _, row in processed_df.iterrows():
            components = []
            
            # Prioritize commodity name/title
            if 'commodity_name' in row and pd.notna(row['commodity_name']) and str(row['commodity_name']).strip():
                components.append(str(row['commodity_name']).strip())
            
            # Add class name if available and not already included
            if 'class_name' in row and pd.notna(row['class_name']) and str(row['class_name']).strip():
                class_text = str(row['class_name']).strip()
                if not components or class_text not in components[0]:
                    components.append(class_text)
            
            # Add family name if available
            if 'family_name' in row and pd.notna(row['family_name']) and str(row['family_name']).strip():
                family_text = str(row['family_name']).strip()
                if not any(family_text in comp for comp in components):
                    components.append(family_text)
            
            # Add segment name if available
            if 'segment_name' in row and pd.notna(row['segment_name']) and str(row['segment_name']).strip():
                segment_text = str(row['segment_name']).strip()
                if not any(segment_text in comp for comp in components):
                    components.append(segment_text)
            
            # Join components with space
            searchable_text = ' '.join(components) if components else ''
            text_components.append(searchable_text)
        
        processed_df['searchable_text'] = text_components
        
        # Ensure class_name column exists for classification target
        if 'class_name' not in processed_df.columns or processed_df['class_name'].isna().all():
            # Fallback hierarchy: commodity -> family -> segment
            if 'commodity_name' in processed_df.columns:
                processed_df['class_name'] = processed_df['commodity_name']
                logger.info("Using commodity_name as class_name for classification")
            elif 'family_name' in processed_df.columns:
                processed_df['class_name'] = processed_df['family_name']
                logger.info("Using family_name as class_name for classification")
            elif 'segment_name' in processed_df.columns:
                processed_df['class_name'] = processed_df['segment_name']
                logger.info("Using segment_name as class_name for classification")
            else:
                raise ValueError("No suitable column found for class_name. UNSPSC data must contain at least one hierarchical name column.")
        
        # Clean class_name - handle empty/null values
        processed_df['class_name'] = processed_df['class_name'].fillna('').astype(str).str.strip()
        
        # Remove rows with empty class_name or searchable_text
        initial_count = len(processed_df)
        processed_df = processed_df[
            (processed_df['class_name'] != '') & 
            (processed_df['searchable_text'] != '')
        ].copy()
        
        if len(processed_df) < initial_count:
            logger.warning(f"Removed {initial_count - len(processed_df)} rows with empty class_name or searchable_text")
        
        return processed_df

    def _load_json_annotations_with_images(
        self, dataset_path: str, dataset_name: str
    ) -> List[Dict]:
        """Helper to load JSON annotations and find corresponding images with resizing and bbox scaling.
        
        Args:
            dataset_path: Path to dataset directory
            dataset_name: Name of the dataset for logging
            
        Returns:
            List of processed samples with image_path, words, scaled boxes, ner_tags, and resized dimensions
        """
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            logger.error(f"Dataset directory {dataset_path} for {dataset_name} does not exist.")
            return []

        images_dir = dataset_dir / "images"
        annotations_dir = dataset_dir / "annotations"

        if not images_dir.exists() or not annotations_dir.exists():
            logger.error(f"Required subdirectories 'images' and 'annotations' not found in {dataset_path} for {dataset_name}.")
            return []

        processed_samples = []
        annotation_files = list(annotations_dir.glob("*.json"))

        logger.info(f"Found {len(annotation_files)} annotation files for {dataset_name}.")

        for ann_file in annotation_files:
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)

                # Find corresponding image file
                image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
                image_path = None
                for ext in image_extensions:
                    potential_img_path = images_dir / (ann_file.stem + ext)
                    if potential_img_path.exists():
                        image_path = potential_img_path
                        break

                if not image_path:
                    logger.warning(f"No corresponding image found for annotation {ann_file.name} in {dataset_name}. Skipping.")
                    continue

                # Extract words, boxes, and ner_tags
                words = annotation_data.get("words")
                incoming_boxes = annotation_data.get("boxes")
                ner_tags = annotation_data.get("ner_tags")

                if not words or not incoming_boxes or not ner_tags:
                    logger.warning(f"Skipping {ann_file.name} in {dataset_name}: missing 'words', 'boxes', or 'ner_tags'.")
                    continue

                # Basic consistency check
                if not (len(words) == len(incoming_boxes) == len(ner_tags)):
                    logger.warning(f"Skipping {ann_file.name} in {dataset_name}: inconsistent lengths of words, boxes, or ner_tags.")
                    continue

                # CRITICAL: Load and resize image, then scale bounding boxes
                try:
                    # Load original image
                    image = Image.open(image_path).convert("RGB")
                    original_width, original_height = image.size
                    
                    logger.debug(f"Original image dimensions for {ann_file.name}: {original_width}x{original_height}")
                    
                    # Calculate new dimensions maintaining aspect ratio
                    if original_width > original_height:
                        # Width is the longer side
                        new_width = min(self.MAX_IMAGE_SIDE, original_width)
                        new_height = int((new_width / original_width) * original_height)
                    else:
                        # Height is the longer side
                        new_height = min(self.MAX_IMAGE_SIDE, original_height)
                        new_width = int((new_height / original_height) * original_width)
                    
                    # Resize image using high-quality LANCZOS filter
                    if (new_width, new_height) != (original_width, original_height):
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                        logger.debug(f"Resized image for {ann_file.name}: {original_width}x{original_height} -> {new_width}x{new_height}")
                    
                    # Calculate scaling factors for bounding boxes
                    scale_x = new_width / original_width
                    scale_y = new_height / original_height
                    
                    # Scale bounding boxes proportionally
                    scaled_boxes = []
                    for i, box in enumerate(incoming_boxes):
                        if len(box) == 4 and all(isinstance(c, (int, float)) for c in box):
                            x0, y0, x1, y1 = box
                            
                            # Scale coordinates
                            scaled_x0 = int(x0 * scale_x)
                            scaled_y0 = int(y0 * scale_y)
                            scaled_x1 = int(x1 * scale_x)
                            scaled_y1 = int(y1 * scale_y)
                            
                            # Ensure proper box format (xmin <= xmax, ymin <= ymax)
                            if scaled_x0 > scaled_x1:
                                scaled_x0, scaled_x1 = scaled_x1, scaled_x0
                            if scaled_y0 > scaled_y1:
                                scaled_y0, scaled_y1 = scaled_y1, scaled_y0
                            
                            # Clamp to image boundaries
                            scaled_x0 = max(0, min(scaled_x0, new_width))
                            scaled_y0 = max(0, min(scaled_y0, new_height))
                            scaled_x1 = max(0, min(scaled_x1, new_width))
                            scaled_y1 = max(0, min(scaled_y1, new_height))
                            
                            scaled_boxes.append([scaled_x0, scaled_y0, scaled_x1, scaled_y1])
                        else:
                            logger.warning(f"Invalid box format at index {i} in {ann_file.name}: {box}. Using default [0,0,1,1].")
                            scaled_boxes.append([0, 0, 1, 1])
                    
                    # Save the resized image (optional - for debugging or caching)
                    # resized_image_path = image_path.parent / f"resized_{image_path.name}"
                    # image.save(resized_image_path)
                    
                except Exception as img_error:
                    logger.error(f"Error processing image {image_path} for {ann_file.name}: {img_error}")
                    continue

                processed_sample = {
                    "image_path": str(image_path),
                    "words": words,
                    "boxes": scaled_boxes,  # These are now scaled to the resized image
                    "ner_tags": ner_tags,
                    "relations": annotation_data.get("relations", []),
                    "dataset_source": dataset_name,
                    "annotation_data": annotation_data,
                    "resized_width": new_width,  # Store resized dimensions for normalization
                    "resized_height": new_height,
                    "original_width": original_width,  # Keep original dimensions for reference
                    "original_height": original_height,
                    "scale_factors": {"scale_x": scale_x, "scale_y": scale_y}
                }

                processed_samples.append(processed_sample)

            except Exception as e:
                logger.error(f"Error processing annotation file {ann_file.name} in {dataset_name}: {e}", exc_info=True)
                continue

        logger.info(f"Loaded {len(processed_samples)} samples from {dataset_name} with image resizing and bbox scaling.")
        return processed_samples

    def load_kaggle_invoice_dataset(self, dataset_path: str) -> List[Dict]:
        """Load Kaggle invoice dataset (images and JSON annotations) with resizing.
        
        Args:
            dataset_path: Path to Kaggle invoice dataset directory
            
        Returns:
            List of processed invoice samples with resized images and scaled bboxes
        """
        return self._load_json_annotations_with_images(dataset_path, "KAGGLE_INVOICE")

    def load_synthetic_invoice_dataset(self, dataset_path: str) -> List[Dict]:
        """Load synthetic invoice dataset (images and JSON annotations) with resizing.
        
        Args:
            dataset_path: Path to synthetic invoice dataset directory
            
        Returns:
            List of processed invoice samples with resized images and scaled bboxes
        """
        return self._load_json_annotations_with_images(dataset_path, "SYNTHETIC_INVOICE")

    def unify_dataset_format(self, samples: List[Dict]) -> List[Dict]:
        """Convert samples to unified format with NER tag standardization."""
        unified_samples = []
        for sample in samples:
            # Validate required fields
            if not all(key in sample for key in ["image_path", "words", "boxes", "ner_tags", "resized_width", "resized_height"]):
                logger.warning(f"Skipping malformed sample from {sample.get('dataset_source')}")
                continue

            # Extract data with validation
            words = sample["words"]
            boxes = sample["boxes"]
            ner_tags = sample["ner_tags"]
            resized_width = sample["resized_width"]
            resized_height = sample["resized_height"]
            
            # Validate array lengths
            if not (len(words) == len(boxes) == len(ner_tags)):
                logger.warning(f"Skipping sample: inconsistent array lengths")
                continue
                
            # Standardize NER tags
            cleaned_ner_tags = []
            for raw_tag in ner_tags:
                tag_str = str(raw_tag).strip()
                
                # Apply standardization map
                if tag_str in self.ner_tag_standardization_map:
                    cleaned_tag = self.ner_tag_standardization_map[tag_str]
                # Handle existing B-/I- prefixes
                elif tag_str.startswith(('B-', 'I-')):
                    cleaned_tag = tag_str
                # Fallback to 'O' for unknown tags
                else:
                    cleaned_tag = 'O'
                    # Log unknown tags once
                    if tag_str not in self._logged_unmapped_tags:
                        logger.warning(f"Unknown NER tag '{tag_str}' mapped to 'O'")
                        self._logged_unmapped_tags.add(tag_str)
                
                cleaned_ner_tags.append(cleaned_tag)

            # Normalize AND convert bounding boxes to integers
            normalized_boxes = []
            for box in boxes:
                if len(box) != 4:
                    logger.warning(f"Invalid box format: {box}")
                    normalized_boxes.append([0, 0, 1, 1])
                    continue
                    
                x0, y0, x1, y1 = box
                norm_x0 = max(0, min(1000, int(round((x0 / resized_width) * 1000))))
                norm_y0 = max(0, min(1000, int(round((y0 / resized_height) * 1000))))
                norm_x1 = max(0, min(1000, int(round((x1 / resized_width) * 1000))))
                norm_y1 = max(0, min(1000, int(round((y1 / resized_height) * 1000))))
                
                # Ensure valid box dimensions
                if norm_x1 <= norm_x0:
                    norm_x1 = norm_x0 + 1
                if norm_y1 <= norm_y0:
                    norm_y1 = norm_y0 + 1
                    
                normalized_boxes.append([norm_x0, norm_y0, norm_x1, norm_y1])

            # Create unified sample
            unified_sample = {
                "image_path": sample["image_path"],
                "tokens": words,
                "bboxes": normalized_boxes,
                "ner_tags": cleaned_ner_tags,
                "relations": sample.get("relations", []),
                "dataset_source": sample.get("dataset_source", "unknown")
            }
            unified_samples.append(unified_sample)
            
        logger.info(f"Unified {len(unified_samples)} samples with standardized NER tags")
        return unified_samples
    def prepare_unspsc_training_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
        """Prepare UNSPSC data for classification training.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (training_dataframe, label_to_idx_mapping, idx_to_label_mapping)
        """
        if self.unspsc_data is None:
            raise ValueError("UNSPSC data not loaded. Call load_unspsc_data() first.")
        
        # Create label mappings
        unique_classes = sorted(self.unspsc_data['class_name'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Prepare training DataFrame
        training_data = self.unspsc_data[['searchable_text', 'class_name']].copy()
        training_data['label'] = training_data['class_name'].map(self.label_to_idx)
        
        # Shuffle the data
        training_data = training_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        logger.info(f"Prepared UNSPSC training data: {len(training_data)} samples, {len(unique_classes)} classes")
        
        return training_data, self.label_to_idx, self.idx_to_label

    def create_training_split(
        self, samples: List[Dict], train_ratio: float = 0.8, random_state: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into training and validation sets.
        
        Args:
            samples: List of dataset samples
            train_ratio: Fraction of data to use for training
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_samples, validation_samples)
        """
        if len(samples) < 2:
            logger.warning("Not enough samples for splitting. Returning all samples as train, empty as validation.")
            return samples, []

        train_samples, val_samples = train_test_split(
            samples, test_size=(1 - train_ratio), random_state=random_state, shuffle=True
        )

        logger.info(f"Created training split: {len(train_samples)} train, {len(val_samples)} validation")
        return train_samples, val_samples

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about loaded datasets.
        
        Returns:
            Dictionary containing dataset statistics (JSON serializable)
        """
        stats = {
            "unspsc_data": None,
            "label_mappings": None,
            "image_processing": {
                "max_image_side": int(self.MAX_IMAGE_SIDE),
                "resize_filter": "LANCZOS",
                "bbox_normalization_range": "0-1000"
            }
        }
        
        if self.unspsc_data is not None:
            stats["unspsc_data"] = {
                "total_records": int(len(self.unspsc_data)),
                "unique_classes": int(self.unspsc_data['class_name'].nunique()),
                "has_searchable_text": bool('searchable_text' in self.unspsc_data.columns),
                "avg_searchable_text_length": float(self.unspsc_data['searchable_text'].str.len().mean()) if 'searchable_text' in self.unspsc_data.columns else 0.0,
                "columns": list(self.unspsc_data.columns),
            }
        
        if self.label_to_idx is not None:
            stats["label_mappings"] = {
                "num_classes": int(len(self.label_to_idx)),
                "sample_classes": list(self.label_to_idx.keys())[:10],  # First 10 classes as sample
            }
        
        return stats

    def save_processed_data(self, data: Any, filename: str) -> None:
        """Save processed data to disk.
        
        Args:
            data: Data to save (DataFrame, dict, list, etc.)
            filename: Filename to save to
        """
        filepath = self.data_dir / filename

        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
                logger.info(f"DataFrame saved to {filepath}")
            elif isinstance(data, (dict, list)):
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"JSON data saved to {filepath}")
            else:
                # For other types, use pickle
                import pickle
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
                logger.info(f"Pickled data saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}", exc_info=True)
            raise

    def load_processed_data(self, filename: str) -> Any:
        """Load processed data from disk.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Loaded data
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        try:
            if filename.endswith(".csv"):
                data = pd.read_csv(filepath)
                logger.info(f"CSV data loaded from {filepath}")
                return data
            elif filename.endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"JSON data loaded from {filepath}")
                return data
            else:
                import pickle
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Pickled data loaded from {filepath}")
                return data

        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    """Demonstration of the DataPreparationUtils functionality."""
    
    # Initialize the data preparation utility
    data_prep = DataPreparationUtils()
    
    try:
        # Load and process UNSPSC data
        logger.info("Loading UNSPSC data...")
        unspsc_df = data_prep.load_unspsc_data()
        
        # Prepare UNSPSC training data
        logger.info("Preparing UNSPSC training data...")
        training_data, label_to_idx, idx_to_label = data_prep.prepare_unspsc_training_data()
        
        logger.info(f"UNSPSC Training data shape: {training_data.shape}")
        logger.info(f"Sample classes: {list(label_to_idx.keys())[:5]}")
        logger.info(f"Sample training record: {training_data.iloc[0].to_dict()}")
        
        # Load invoice datasets
        logger.info("Loading invoice datasets with image resizing and bbox scaling...")
        kaggle_invoice_samples = data_prep.load_kaggle_invoice_dataset("data/kaggle_invoices")
        synthetic_invoice_samples = data_prep.load_synthetic_invoice_dataset("data/synthetic_invoices")
        
        # Combine and unify invoice datasets
        all_invoice_samples = kaggle_invoice_samples + synthetic_invoice_samples
        
        if all_invoice_samples:
            unified_samples = data_prep.unify_dataset_format(all_invoice_samples)
            train_samples, val_samples = data_prep.create_training_split(unified_samples)
            
            logger.info(f"Invoice dataset: {len(train_samples)} train, {len(val_samples)} validation samples")
            
            # Save processed data
            data_prep.save_processed_data(train_samples, "train_invoice_samples.json")
            data_prep.save_processed_data(val_samples, "val_invoice_samples.json")
            data_prep.save_processed_data(training_data, "unspsc_training_data.csv")
            data_prep.save_processed_data(label_to_idx, "unspsc_label_mappings.json")
        
        # Display statistics
        stats = data_prep.get_statistics()
        logger.info("Dataset Statistics:")
        logger.info(json.dumps(stats, indent=2))
        
        logger.info("Data preparation utilities ready for use.")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)