import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreparationUtils:
    """
    Optimized utilities for UNSPSC classification and invoice annotation datasets.
    Focused on core functionality with efficient data handling.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core data storage
        self.unspsc_data: Optional[pd.DataFrame] = None
        self.unspsc_mapping: Dict[str, Dict] = {}
        
        # Label mappings for classification
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        
        # Semantic search components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.unspsc_vectors: Optional[np.ndarray] = None

    def load_unspsc_data(
        self,
        filepath: str = "data/data-unspsc-codes.csv",
        url: Optional[str] = "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/unspsc-codes.csv"
    ) -> pd.DataFrame:
        """Load and process UNSPSC classification data efficiently."""
        
        # Try local file first, then URL
        df = self._load_csv_source(filepath, url)
        if df.empty:
            logger.error("Failed to load UNSPSC data from any source")
            return pd.DataFrame()
        
        # Clean and validate data
        df = self._clean_unspsc_dataframe(df)
        if df.empty:
            return pd.DataFrame()
        
        # Process and structure data
        self.unspsc_data = self._process_unspsc_data(df)
        self._initialize_vectorizer()
        
        logger.info(f"Loaded {len(self.unspsc_data)} UNSPSC entries")
        return self.unspsc_data

    def _load_csv_source(self, filepath: str, url: Optional[str]) -> pd.DataFrame:
        """Load CSV from local file or URL with fallback."""
        # Try local file first
        if Path(filepath).exists():
            try:
                df = pd.read_csv(filepath)
                logger.info(f"Loaded from local file: {filepath}")
                return df
            except Exception as e:
                logger.warning(f"Local file failed: {e}")
        
        # Fallback to URL
        if url:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"Loaded from URL: {url}")
                
                # Save locally for future use
                try:
                    df.to_csv(filepath, index=False)
                    logger.info(f"Cached to: {filepath}")
                except Exception:
                    pass  # Non-critical if caching fails
                
                return df
            except Exception as e:
                logger.error(f"URL loading failed: {e}")
        
        return pd.DataFrame()

    def _clean_unspsc_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate UNSPSC DataFrame structure."""
        # Clean column names
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Expected columns (case-insensitive mapping)
        required_cols = {
            'segment_code': ['segment code', 'segmentcode'],
            'segment_name': ['segment name', 'segmentname'],
            'family_code': ['family code', 'familycode'],
            'family_name': ['family name', 'familyname'],
            'class_code': ['class code', 'classcode'],
            'class_name': ['class name', 'classname'],
            'commodity_code': ['commodity code', 'commoditycode'],
            'commodity_name': ['commodity name', 'commodityname']
        }
        
        # Map actual columns to standard names
        column_mapping = {}
        for standard_name, variations in required_cols.items():
            for col in df.columns:
                if col.lower() in variations:
                    column_mapping[col] = standard_name
                    break
        
        if len(column_mapping) < 6:  # Need at least class and commodity info
            logger.error("Insufficient UNSPSC columns found")
            return pd.DataFrame()
        
        # Rename columns and fill missing values
        df = df.rename(columns=column_mapping)
        df = df.fillna('')
        
        # Filter out rows with empty commodity names (essential for classification)
        df = df[df.get('commodity_name', '') != ''].copy()
        
        return df

    def _process_unspsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process UNSPSC data into standardized format."""
        processed_data = []
        
        for _, row in df.iterrows():
            # Create unique identifier
            unspsc_id = f"{row.get('segment_code', '')}{row.get('family_code', '')}{row.get('class_code', '')}{row.get('commodity_code', '')}"
            
            # Build searchable text
            text_parts = [
                str(row.get('segment_name', '')),
                str(row.get('family_name', '')),
                str(row.get('class_name', '')),
                str(row.get('commodity_name', ''))
            ]
            searchable_text = ' '.join(filter(None, text_parts)).lower()
            
            # Create full hierarchical path
            path_parts = [part for part in text_parts if part.strip()]
            full_path = ' > '.join(path_parts)
            
            entry = {
                'unspsc_id': unspsc_id,
                'segment_code': str(row.get('segment_code', '')),
                'segment_name': str(row.get('segment_name', '')),
                'family_code': str(row.get('family_code', '')),
                'family_name': str(row.get('family_name', '')),
                'class_code': str(row.get('class_code', '')),
                'class_name': str(row.get('class_name', '')),
                'commodity_code': str(row.get('commodity_code', '')),
                'commodity_name': str(row.get('commodity_name', '')),
                'full_path': full_path,
                'searchable_text': searchable_text
            }
            
            processed_data.append(entry)
            self.unspsc_mapping[unspsc_id] = entry
        
        return pd.DataFrame(processed_data)

    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer for semantic search."""
        if self.unspsc_data is None or self.unspsc_data.empty:
            logger.warning("No UNSPSC data available for vectorization")
            return
        
        texts = self.unspsc_data['searchable_text'].tolist()
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_features=10000,
            min_df=1,
            max_df=0.95
        )
        
        self.unspsc_vectors = self.vectorizer.fit_transform(texts)
        logger.info("Vectorizer initialized for semantic search")

    def find_unspsc_by_description(
        self, 
        description: str, 
        top_k: int = 5, 
        min_similarity: float = 0.1
    ) -> List[Dict]:
        """Find best matching UNSPSC entries using semantic search."""
        if not self._is_search_ready():
            return []
        
        try:
            # Clean input description
            clean_desc = self._clean_text(description)
            desc_vector = self.vectorizer.transform([clean_desc])
            
            # Calculate similarities
            similarities = cosine_similarity(desc_vector, self.unspsc_vectors).flatten()
            
            # Get top matches
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= min_similarity:
                    entry = self.unspsc_data.iloc[idx].to_dict()
                    entry['similarity_score'] = float(similarity)
                    results.append(entry)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._keyword_search_fallback(description, top_k)

    def _is_search_ready(self) -> bool:
        """Check if semantic search components are ready."""
        if self.vectorizer is None or self.unspsc_vectors is None:
            logger.warning("Search not initialized")
            return False
        return True

    def _clean_text(self, text: str) -> str:
        """Clean text for better matching."""
        # Remove non-alphanumeric characters, normalize spaces
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(text.split())

    def _keyword_search_fallback(self, description: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search."""
        if self.unspsc_data is None or self.unspsc_data.empty:
            return []
        
        desc_words = set(self._clean_text(description).split())
        matches = []
        
        for idx, row in self.unspsc_data.iterrows():
            text_words = set(row['searchable_text'].split())
            
            # Jaccard similarity
            intersection = desc_words.intersection(text_words)
            union = desc_words.union(text_words)
            
            if union:
                score = len(intersection) / len(union)
                if score > 0:
                    entry = row.to_dict()
                    entry['similarity_score'] = score
                    matches.append(entry)
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:top_k]

    def prepare_classification_data(
        self, 
        text_column: str = 'searchable_text',
        label_column: str = 'class_name'
    ) -> pd.DataFrame:
        """Prepare data for classification training."""
        if self.unspsc_data is None or self.unspsc_data.empty:
            logger.error("No UNSPSC data loaded")
            return pd.DataFrame()
        
        # Create label mappings
        unique_labels = self.unspsc_data[label_column].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Add numerical labels
        df = self.unspsc_data.copy()
        df['label'] = df[label_column].map(self.label_to_idx)
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Prepared {len(df)} samples for {len(unique_labels)} classes")
        return df[[text_column, 'label', label_column]]

    def load_invoice_annotations(self, dataset_path: str) -> List[Dict]:
        """Load invoice annotation dataset (JSON + images)."""
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return []
        
        images_dir = dataset_dir / "images"
        annotations_dir = dataset_dir / "annotations"
        
        if not (images_dir.exists() and annotations_dir.exists()):
            logger.error("Missing 'images' or 'annotations' directories")
            return []
        
        samples = []
        annotation_files = list(annotations_dir.glob("*.json"))
        
        for ann_file in annotation_files:
            try:
                sample = self._process_annotation_file(ann_file, images_dir)
                if sample:
                    samples.append(sample)
            except Exception as e:
                logger.error(f"Error processing {ann_file.name}: {e}")
        
        logger.info(f"Loaded {len(samples)} invoice annotation samples")
        return samples

    def _process_annotation_file(self, ann_file: Path, images_dir: Path) -> Optional[Dict]:
        """Process a single annotation file."""
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # Find corresponding image
        image_path = self._find_image_file(ann_file.stem, images_dir)
        if not image_path:
            logger.warning(f"No image found for {ann_file.name}")
            return None
        
        # Validate required fields
        required_fields = ['words', 'boxes', 'ner_tags']
        if not all(field in annotation for field in required_fields):
            logger.warning(f"Missing required fields in {ann_file.name}")
            return None
        
        words = annotation['words']
        boxes = annotation['boxes']
        ner_tags = annotation['ner_tags']
        
        # Validate consistency
        if not (len(words) == len(boxes) == len(ner_tags)):
            logger.warning(f"Inconsistent field lengths in {ann_file.name}")
            return None
        
        # Clean and validate boxes
        cleaned_boxes = self._clean_bounding_boxes(boxes)
        
        return {
            'image_path': str(image_path),
            'words': words,
            'boxes': cleaned_boxes,
            'ner_tags': ner_tags,
            'relations': annotation.get('relations', [])
        }

    def _find_image_file(self, base_name: str, images_dir: Path) -> Optional[Path]:
        """Find corresponding image file for annotation."""
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        for ext in extensions:
            image_path = images_dir / f"{base_name}{ext}"
            if image_path.exists():
                return image_path
        return None

    def _clean_bounding_boxes(self, boxes: List[List]) -> List[List[int]]:
        """Clean and validate bounding boxes."""
        cleaned = []
        for box in boxes:
            if len(box) == 4 and all(isinstance(coord, (int, float)) for coord in box):
                cleaned.append([int(coord) for coord in box])
            else:
                logger.warning(f"Invalid box format: {box}")
                cleaned.append([0, 0, 0, 0])  # Fallback
        return cleaned

    def create_train_val_split(
        self, 
        data: Union[List[Dict], pd.DataFrame], 
        train_ratio: float = 0.8
    ) -> Tuple[Union[List[Dict], pd.DataFrame], Union[List[Dict], pd.DataFrame]]:
        """Split data into training and validation sets."""
        if isinstance(data, list) and len(data) < 2:
            logger.warning("Insufficient samples for splitting")
            return data, []
        elif isinstance(data, pd.DataFrame) and len(data) < 2:
            logger.warning("Insufficient samples for splitting")
            return data, pd.DataFrame()
        
        train_data, val_data = train_test_split(
            data, 
            test_size=(1 - train_ratio), 
            random_state=42, 
            shuffle=True
        )
        
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data

    def save_data(self, data: Any, filename: str, format: str = 'auto'):
        """Save data efficiently with format auto-detection."""
        filepath = self.data_dir / filename
        
        try:
            if isinstance(data, pd.DataFrame):
                if format == 'auto':
                    format = 'parquet' if len(data) > 10000 else 'csv'
                
                if format == 'parquet':
                    data.to_parquet(filepath.with_suffix('.parquet'), index=False)
                else:
                    data.to_csv(filepath.with_suffix('.csv'), index=False)
                    
            elif isinstance(data, (dict, list)):
                with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                import pickle
                with open(filepath.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise

    def load_data(self, filename: str) -> Any:
        """Load data with automatic format detection."""
        filepath = self.data_dir / filename
        
        # Try different extensions if file doesn't exist
        if not filepath.exists():
            for ext in ['.parquet', '.csv', '.json', '.pkl']:
                alt_path = filepath.with_suffix(ext)
                if alt_path.exists():
                    filepath = alt_path
                    break
            else:
                raise FileNotFoundError(f"File not found: {filename}")
        
        try:
            if filepath.suffix == '.csv':
                return pd.read_csv(filepath)
            elif filepath.suffix == '.parquet':
                return pd.read_parquet(filepath)
            elif filepath.suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif filepath.suffix == '.pkl':
                import pickle
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
                
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about loaded data."""
        stats = {
            'unspsc_loaded': self.unspsc_data is not None,
            'search_ready': self._is_search_ready(),
            'label_mappings_ready': bool(self.label_to_idx)
        }
        
        if self.unspsc_data is not None:
            stats.update({
                'total_entries': len(self.unspsc_data),
                'unique_segments': self.unspsc_data['segment_name'].nunique(),
                'unique_families': self.unspsc_data['family_name'].nunique(),
                'unique_classes': self.unspsc_data['class_name'].nunique(),
                'unique_commodities': self.unspsc_data['commodity_name'].nunique()
            })
        
        return stats

    def reset(self):
        """Reset all loaded data and models."""
        self.unspsc_data = None
        self.unspsc_mapping.clear()
        self.label_to_idx.clear()
        self.idx_to_label.clear()
        self.vectorizer = None
        self.unspsc_vectors = None
        logger.info("Data preparation utils reset")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the utility
    data_prep = DataPreparationUtils()
    
    # Load UNSPSC data
    print("Loading UNSPSC data...")
    unspsc_df = data_prep.load_unspsc_data()
    
    if not unspsc_df.empty:
        print(f"Loaded {len(unspsc_df)} UNSPSC entries")
        
        # Test semantic search
        print("\nTesting semantic search...")
        results = data_prep.find_unspsc_by_description("office supplies paper", top_k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['commodity_name']} (Score: {result['similarity_score']:.3f})")
        
        # Prepare classification data
        print("\nPreparing classification data...")
        training_data = data_prep.prepare_classification_data()
        print(f"Training data shape: {training_data.shape}")
        
        # Show statistics
        print("\nData statistics:")
        stats = data_prep.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print("Failed to load UNSPSC data")