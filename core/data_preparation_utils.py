import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image # Keep if still using for invoice image loading
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreparationUtils:
    """
    Essential utilities for loading and processing datasets for IDP training.
    Now includes robust methods for handling UNSPSC classification data
    and invoice annotation datasets (Kaggle/Synthetic).
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.unspsc_data: Optional[Dict] = None # Store loaded UNSPSC data dict
        # Store label mappings for UNSPSC classification, populated by prepare_training_data
        self.unspsc_label_to_idx: Dict[str, int] = {}
        self.unspsc_idx_to_label: Dict[int, str] = {}
        # For semantic search fallback/enhancement
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.unspsc_vectors: Optional[Any] = None # Stores the vectorized UNSPSC descriptions

    def load_unspsc_data(
        self,
        filepath: str = "data/data-unspsc-codes.csv", # Changed to filepath as it's a local file
        url: Optional[str] = "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/unspsc-codes.csv",
    ) -> Dict:
        """
        Load and structure UNSPSC classification data from a CSV file (local first, then URL fallback).
        Expected columns: 'Segment Code', 'Segment Name', 'Family Code', 'Family Name',
                          'Class Code', 'Class Name', 'Commodity Code', 'Commodity Name'.
        """
        df = pd.DataFrame()
        if Path(filepath).exists():
            try:
                df = pd.read_csv(filepath)
                logger.info(f"Loaded UNSPSC data from local file: {filepath}")
            except Exception as e:
                logger.warning(f"Could not load local UNSPSC file {filepath}: {e}. Trying URL fallback.")

        if df.empty and url:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                from io import StringIO
                csv_content = StringIO(response.text)
                df = pd.read_csv(csv_content)
                logger.info(f"Loaded UNSPSC data from URL: {url}")
                # Optionally save to local file for future faster access
                df.to_csv(filepath, index=False)
                logger.info(f"Saved UNSPSC data to local file: {filepath}")
            except Exception as e:
                logger.error(f"Error loading UNSPSC data from URL {url}: {e}")
                return {}

        if df.empty:
            logger.error("No UNSPSC data could be loaded from local file or URL.")
            return {}

        # Clean column names (remove quotes and extra spaces)
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Ensure required columns exist (case-insensitive search and map to actual column names)
        required_cols_map = {
            "Segment Code": None, "Segment Name": None, "Family Code": None, 
            "Family Name": None, "Class Code": None, "Class Name": None, 
            "Commodity Code": None, "Commodity Name": None
        }
        
        for req_col in required_cols_map.keys():
            matching_col = None
            for col in df.columns:
                if req_col.lower() == col.lower():
                    matching_col = col
                    break
            required_cols_map[req_col] = matching_col
            if matching_col is None:
                logger.warning(f"Required column '{req_col}' not found in UNSPSC dataset. This may affect data quality.")
        
        # Fill NaN values with empty strings for consistency
        df_clean = df.fillna('')
        
        unspsc_mapping = {}
        segment_mapping = {}
        family_mapping = {}
        class_mapping = {}
        commodity_mapping = {}
        
        for idx, row in df_clean.iterrows():
            # Use column_mapping to safely access columns, defaulting to empty string if not found
            segment_code = str(row[required_cols_map["Segment Code"]]) if required_cols_map["Segment Code"] and required_cols_map["Segment Code"] in row else ""
            segment_name = str(row[required_cols_map["Segment Name"]]).strip() if required_cols_map["Segment Name"] and required_cols_map["Segment Name"] in row else ""
            family_code = str(row[required_cols_map["Family Code"]]) if required_cols_map["Family Code"] and required_cols_map["Family Code"] in row else ""
            family_name = str(row[required_cols_map["Family Name"]]).strip() if required_cols_map["Family Name"] and required_cols_map["Family Name"] in row else ""
            class_code = str(row[required_cols_map["Class Code"]]) if required_cols_map["Class Code"] and required_cols_map["Class Code"] in row else ""
            class_name = str(row[required_cols_map["Class Name"]]).strip() if required_cols_map["Class Name"] and required_cols_map["Class Name"] in row else ""
            commodity_code = str(row[required_cols_map["Commodity Code"]]) if required_cols_map["Commodity Code"] and required_cols_map["Commodity Code"] in row else ""
            commodity_name = str(row[required_cols_map["Commodity Name"]]).strip() if required_cols_map["Commodity Name"] and required_cols_map["Commodity Name"] in row else ""
            
            # Skip entries where the commodity name is missing for a complete hierarchical path
            if not commodity_name:
                continue
            
            # Create unique identifier based on codes (more robust than names)
            unspsc_id = f"{segment_code}{family_code}{class_code}{commodity_code}"
            
            # Create comprehensive mapping structure
            unspsc_entry = {
                "unspsc_id": unspsc_id, # Add ID to the entry itself
                "segment_code": segment_code,
                "segment_name": segment_name,
                "family_code": family_code,
                "family_name": family_name,
                "class_code": class_code,
                "class_name": class_name,
                "commodity_code": commodity_code,
                "commodity_name": commodity_name,
                "full_path": f"{segment_name} > {family_name} > {class_name} > {commodity_name}",
                "searchable_text": f"{segment_name} {family_name} {class_name} {commodity_name}".lower()
            }
            
            unspsc_mapping[unspsc_id] = unspsc_entry
            
            # Create reverse mappings for classification and lookup
            self._add_to_mapping(segment_mapping, segment_name, unspsc_id)
            self._add_to_mapping(family_mapping, family_name, unspsc_id)
            self._add_to_mapping(class_mapping, class_name, unspsc_id)
            self._add_to_mapping(commodity_mapping, commodity_name, unspsc_id)

        self.unspsc_data = {
            "unspsc_mapping": unspsc_mapping, # Full details mapped by unique ID
            "segment_mapping": segment_mapping,
            "family_mapping": family_mapping,
            "class_mapping": class_mapping,
            "commodity_mapping": commodity_mapping,
            "all_segments": list(segment_mapping.keys()),
            "all_families": list(family_mapping.keys()),
            "all_classes": list(class_mapping.keys()),
            "all_commodities": list(commodity_mapping.keys()),
            # Store the cleaned DataFrame directly for easier access later
            "processed_df": pd.DataFrame.from_records(list(unspsc_mapping.values()))
        }

        # Initialize vectorizer for semantic search (used by get_unspsc_by_description)
        self._initialize_vectorizer()

        logger.info(
            f"Loaded {len(unspsc_mapping)} UNSPSC entries across "
            f"{len(segment_mapping)} segments, {len(family_mapping)} families, "
            f"{len(class_mapping)} classes, and {len(commodity_mapping)} commodities"
        )
        
        return self.unspsc_data

    def _add_to_mapping(self, mapping_dict: Dict, key: str, value: str):
        """Helper method to add values to mapping dictionaries."""
        if key not in mapping_dict:
            mapping_dict[key] = []
        mapping_dict[key].append(value)

    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer for semantic search."""
        # Check if unspsc_data is loaded and contains data for vectorization
        if not self.unspsc_data or not self.unspsc_data.get("processed_df"):
            logger.warning("UNSPSC processed_df not available for vectorizer initialization.")
            return
        
        # Prepare texts for vectorization (using the combined searchable text from the processed_df)
        texts = self.unspsc_data["processed_df"]["searchable_text"].tolist()
        
        # Initialize vectorizer with optimal parameters
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3), # Capture phrases
            max_features=10000, # Limit features for efficiency
            min_df=1,
            max_df=0.95 # Ignore very common words
        )
        
        # Fit and transform the texts
        self.unspsc_vectors = self.vectorizer.fit_transform(texts)
        logger.info("UNSPSC vectorizer initialized for semantic search")

    def load_json_annotations_with_images(
        self, dataset_path: str, dataset_name: str
    ) -> List[Dict]:
        """Helper to load JSON annotations (LayoutLMv3 format) and find corresponding images."""
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            logger.warning(
                f"Dataset directory {dataset_path} for {dataset_name} does not exist."
            )
            return []

        images_dir = dataset_dir / "images"
        annotations_dir = dataset_dir / "annotations"

        if not images_dir.exists() or not annotations_dir.exists():
            logger.warning(
                f"Required subdirectories 'images' and 'annotations' not found in {dataset_path} for {dataset_name}. "
                "Ensure your dataset structure matches the expected format (e.g., 'dataset_name/images/' and 'dataset_name/annotations/')."
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

                # Find corresponding image file
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

                # Extract required fields for LayoutLMv3 training
                words = annotation_data.get("words")
                boxes = annotation_data.get("boxes")
                ner_tags = annotation_data.get("ner_tags")

                if not words or not boxes or not ner_tags:
                    logger.warning(
                        f"Skipping {ann_file.name} in {dataset_name}: missing 'words', 'boxes', or 'ner_tags'. These are critical for LayoutLMv3."
                    )
                    continue

                # Consistency check
                if not (len(words) == len(boxes) == len(ner_tags)):
                    logger.warning(
                        f"Skipping {ann_file.name} in {dataset_name}: inconsistent lengths of words, boxes, or ner_tags. All must match."
                    )
                    continue
                
                # Normalize bounding boxes to 0-1000 range if they are not already
                # (LayoutLMv3 expects 0-1000). This requires knowing original image dimensions.
                # Assuming `normalize_bbox` would be called elsewhere if needed, or that data is already normalized.

                processed_sample = {
                    "image_path": str(image_path),
                    "words": words,
                    "boxes": boxes,
                    "ner_tags": ner_tags,
                    "relations": annotation_data.get("relations", []), # Keep relations if available
                    "dataset_source": dataset_name,
                    # Optionally store more annotation data or derived labels here for LayoutLMv3 if needed
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
        """Load Kaggle invoice dataset (images and JSON annotations in LayoutLM format)."""
        return self.load_json_annotations_with_images(dataset_path, "KAGGLE_INVOICE")

    def load_synthetic_invoice_dataset(self, dataset_path: str) -> List[Dict]:
        """Load synthetic invoice dataset (images and JSON annotations in LayoutLM format)."""
        return self.load_json_annotations_with_images(
            dataset_path, "SYNTHETIC_INVOICE"
        )

    def unify_invoice_dataset_format(self, samples: List[Dict]) -> List[Dict]:
        """
        Ensures all invoice annotation samples (e.g., from Kaggle or Synthetic) conform
        to a unified format expected by the LayoutLMv3 trainer.
        This step is crucial for consistent input to the LayoutLMDataset.
        """
        unified_samples = []
        for sample in samples:
            image_path = sample.get("image_path")
            words = sample.get("words", [])
            boxes = sample.get("boxes", [])
            ner_tags = sample.get("ner_tags", [])

            # Basic validation
            if not (image_path and words and boxes and ner_tags and len(words) == len(boxes) == len(ner_tags)):
                logger.warning(
                    f"Skipping malformed invoice sample from {sample.get('dataset_source', 'unknown')}: "
                    "missing essential fields or inconsistent lengths."
                )
                continue

            # Ensure boxes are integers and within expected range (e.g., 0-1000 for LayoutLMv3)
            # This logic should ideally be applied during initial data loading or a dedicated normalization step.
            # Assuming boxes are already normalized by the `synthetic_data_generator` or external prep.
            cleaned_boxes = []
            for box in boxes:
                # LayoutLMv3 expects [x0, y0, x1, y1] for each word
                if len(box) == 4 and all(isinstance(coord, (int, float)) for coord in box):
                    cleaned_boxes.append([int(c) for c in box])
                else:
                    logger.warning(f"Invalid box format found: {box}. Replacing with [0,0,0,0].")
                    cleaned_boxes.append([0,0,0,0]) # Fallback for malformed boxes

            unified_sample = {
                "image_path": image_path,
                "words": words,
                "boxes": cleaned_boxes,
                "ner_tags": ner_tags,
                "relations": sample.get("relations", []), # Relations are optional but good to keep
                "dataset_source": sample.get("dataset_source", "unknown"),
            }
            unified_samples.append(unified_sample)
        logger.info(f"Unified {len(unified_samples)} invoice samples.")
        return unified_samples


    def prepare_training_data(
        self,
        data_df: pd.DataFrame,
        text_column: str = 'searchable_text', # Use searchable_text for model input
        label_column: str = 'class_name', # Target is UNSPSC Class Name
    ) -> pd.DataFrame:
        """
        Prepares UNSPSC data for item categorization training.
        Creates unique numerical labels and mappings based on the chosen label_column.
        Stores label_to_idx and idx_to_label internally.
        """
        if data_df.empty:
            logger.error("Input DataFrame is empty, cannot prepare training data.")
            return pd.DataFrame()

        # Create unique numerical labels for each target class name
        unique_labels = data_df[label_column].unique()
        self.unspsc_label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.unspsc_idx_to_label = {i: label for label, i in self.unspsc_label_to_idx.items()}

        data_df['label'] = data_df[label_column].map(self.unspsc_label_to_idx)

        # Shuffle the DataFrame for randomness during training
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Prepared {len(data_df)} training samples for {len(unique_labels)} unique classes.")
        logger.info(f"Sample distribution (top 10 target classes):\n{data_df[label_column].value_counts().head(10)}")

        return data_df[[text_column, 'label', label_column]]


    def get_unspsc_by_description(
        self, description: str, top_k: int = 1, min_similarity: float = 0.1
    ) -> Optional[List[Dict]]:
        """Find best matching UNSPSC entries based on item description using semantic search."""
        if not self.unspsc_data or not self.vectorizer or self.unspsc_vectors is None:
            logger.warning(
                "UNSPSC data not loaded or vectorizer not initialized for description lookup. Attempting to load."
            )
            # Try to load if not already
            if self.unspsc_data is None:
                self.load_unspsc_data()
            if self.unspsc_data is None or self.vectorizer is None or self.unspsc_vectors is None:
                logger.error("Failed to load UNSPSC data for description lookup. Cannot perform search.")
                return None

        try:
            # Clean and preprocess the description
            description_clean = self._clean_text(description)
            
            # Vectorize the input description
            desc_vector = self.vectorizer.transform([description_clean])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(desc_vector, self.unspsc_vectors).flatten()
            
            # Get top matches
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            # Get list of all UNSPSC entries to map back by index
            unspsc_entries_list = self.unspsc_data["processed_df"].to_dict(orient='records') # Use processed_df for direct access to generated searchable_text etc.
            
            for idx in top_indices:
                similarity_score = similarities[idx]
                if similarity_score >= min_similarity:
                    entry = unspsc_entries_list[idx].copy()
                    entry["similarity_score"] = float(similarity_score)
                    results.append(entry)
            
            return results if results else None
            
        except Exception as e:
            logger.error(f"Error in semantic UNSPSC search for '{description}': {e}")
            # Fallback to keyword-based search if semantic search fails
            return self._fallback_keyword_search(description, top_k)

    def _clean_text(self, text: str) -> str:
        """Clean text for better matching."""
        # Convert to lowercase and remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Replace multiple spaces with a single space and strip leading/trailing whitespace
        text = ' '.join(text.split()).strip()
        return text

    def _fallback_keyword_search(self, description: str, top_k: int = 1) -> Optional[List[Dict]]:
        """Fallback keyword-based search when semantic search fails or is not applicable."""
        if not self.unspsc_data or not self.unspsc_data.get("processed_df"):
            return None

        description_lower = self._clean_text(description)
        desc_words = set(description_lower.split())
        
        matches = []
        
        # Iterate through the processed DataFrame for keyword search
        for idx, unspsc_info in self.unspsc_data["processed_df"].iterrows():
            searchable_text = unspsc_info["searchable_text"]
            text_words = set(searchable_text.split())
            
            # Calculate word overlap score (Jaccard similarity)
            intersection = desc_words.intersection(text_words)
            union = desc_words.union(text_words)
            
            if union: # Avoid division by zero
                score = len(intersection) / len(union)
            else:
                score = 0.0 # No common words and empty combined set

            if score > 0: # Only add if there's some overlap
                match_entry = unspsc_info.to_dict() # Convert Series to dict
                match_entry["similarity_score"] = score
                matches.append(match_entry)
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:top_k] if matches else None

    def get_classification_by_level(self, level: str, value: str) -> List[str]:
        """Get all UNSPSC unique IDs that match a specific level name and value."""
        if not self.unspsc_data:
            logger.warning("UNSPSC data not loaded.")
            return []
        
        level_mapping_keys = {
            "segment": "segment_mapping",
            "family": "family_mapping", 
            "class": "class_mapping",
            "commodity": "commodity_mapping"
        }
        
        mapping_key = level_mapping_keys.get(level.lower())
        if not mapping_key:
            logger.warning(f"Invalid level: {level}. Valid levels are: {list(level_mapping_keys.keys())}")
            return []
        
        mapping = self.unspsc_data.get(mapping_key, {})
        return mapping.get(value, []) # Returns a list of unspsc_ids

    def create_training_split(
        self, samples: List[Dict], train_ratio: float = 0.8
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into training and validation sets."""
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
        """Save processed data to disk efficiently."""
        filepath = self.data_dir / filename

        try:
            if isinstance(data, pd.DataFrame):
                # Use efficient compression for large datasets
                if len(data) > 10000 and filepath.suffix != '.csv': # Use parquet for large DFs unless CSV is explicitly needed
                    data.to_parquet(filepath.with_suffix('.parquet'), index=False)
                    logger.info(f"Large dataset saved as Parquet to {filepath.with_suffix('.parquet')}")
                else:
                    data.to_csv(filepath, index=False)
                    logger.info(f"Data saved as CSV to {filepath}")
            elif isinstance(data, (dict, list)):
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"JSON data saved to {filepath}")
            else:
                import pickle
                with open(filepath.with_suffix('.pkl'), "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Pickled data saved to {filepath.with_suffix('.pkl')}")
                
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def load_processed_data(self, filename: str):
        """Load processed data from disk efficiently."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            # Try alternative extensions
            alternative_paths = [
                filepath.with_suffix('.parquet'),
                filepath.with_suffix('.pkl'),
                filepath.with_suffix('.json'),
                filepath.with_suffix('.csv') # Try CSV last if not explicitly asked
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    filepath = alt_path
                    break
            else:
                raise FileNotFoundError(f"File {filepath} not found")

        try:
            if filepath.suffix == '.csv':
                return pd.read_csv(filepath)
            elif filepath.suffix == '.parquet':
                return pd.read_parquet(filepath)
            elif filepath.suffix == '.json':
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif filepath.suffix == '.pkl':
                import pickle
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            else:
                logger.warning(f"Unknown file format for {filepath}. Attempting JSON load.")
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            raise

    def get_statistics(self) -> Dict:
        """Get statistics about the loaded UNSPSC data."""
        if not self.unspsc_data:
            return {"error": "No UNSPSC data loaded"}
        
        return {
            "total_entries": len(self.unspsc_data.get("unspsc_mapping", {})),
            "segments": len(self.unspsc_data.get("all_segments", [])),
            "families": len(self.unspsc_data.get("all_families", [])),
            "classes": len(self.unspsc_data.get("all_classes", [])),
            "commodities": len(self.unspsc_data.get("all_commodities", [])),
            "vectorizer_initialized": self.vectorizer is not None
        }

