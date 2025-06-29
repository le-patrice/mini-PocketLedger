import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from io import StringIO
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreparationUtils:
    """
    Optimized data preparation utilities for UNSPSC dataset and invoice processing.
    Includes robust methods for loading, automatically mapping columns,
    processing to create searchable text, and preparing training data.
    """
    
    def __init__(self):
        self.unspsc_data: Optional[Dict[str, pd.DataFrame]] = None
        self.unspsc_label_to_idx: Dict[str, int] = {}
        self.unspsc_idx_to_label: Dict[int, str] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.unspsc_vectors = None
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing."""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string first
        text = str(text)
        
        # Basic text cleaning: lowercase, remove non-alphanumeric, normalize spaces
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _safe_str_convert(self, value) -> str:
        """Safely convert any value to string, handling NaNs and None."""
        if pd.isna(value) or value is None:
            return ""
        return str(value).strip()
    
    def load_unspsc_data(self, filepath_or_url: str) -> Dict[str, pd.DataFrame]:
        """
        Load UNSPSC data from a local file path or a URL.
        Automatically detects column names and processes the data for training.
        
        Args:
            filepath_or_url: Path to local CSV file or URL to download from
            
        Returns:
            Dictionary containing 'raw_df' and 'processed_df' DataFrames
        """
        try:
            # Load data based on input type
            if filepath_or_url.startswith(('http://', 'https://')):
                logger.info(f"Attempting to load UNSPSC data from URL: {filepath_or_url}")
                response = requests.get(filepath_or_url, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
            else:
                filepath = Path(filepath_or_url)
                if not filepath.exists():
                    raise FileNotFoundError(f"Local file not found: {filepath_or_url}")
                logger.info(f"Loading UNSPSC data from local file: {filepath_or_url}")
                df = pd.read_csv(filepath)
            
            logger.info(f"Original UNSPSC dataset shape: {df.shape}")
            logger.info(f"Original UNSPSC columns: {df.columns.tolist()}")
            
            # Handle empty dataset
            if df.empty:
                logger.warning("Loaded dataset is empty")
                return {'raw_df': pd.DataFrame(), 'processed_df': pd.DataFrame()}
            
            # Automated Column Mapping for Generalization
            column_mappings = self._detect_column_mappings(df)
            df_renamed = df.rename(columns=column_mappings)
            
            # Process the data after renaming columns to a standard format
            processed_df = self._process_unspsc_data(df_renamed)
            
            self.unspsc_data = {
                'raw_df': df_renamed,
                'processed_df': processed_df
            }
            
            logger.info(f"Successfully loaded and processed {len(processed_df)} UNSPSC records.")
            return self.unspsc_data
            
        except requests.RequestException as e:
            logger.error(f"Network error loading UNSPSC data from '{filepath_or_url}': {e}")
            return {'raw_df': pd.DataFrame(), 'processed_df': pd.DataFrame()}
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {'raw_df': pd.DataFrame(), 'processed_df': pd.DataFrame()}
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {filepath_or_url}")
            return {'raw_df': pd.DataFrame(), 'processed_df': pd.DataFrame()}
        except Exception as e:
            logger.error(f"Failed to load UNSPSC data from '{filepath_or_url}': {e}", exc_info=True)
            return {'raw_df': pd.DataFrame(), 'processed_df': pd.DataFrame()}
    
    def _detect_column_mappings(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detects and maps common UNSPSC column name variations
        to a standardized internal format.
        
        Args:
            df: Input DataFrame with potentially non-standard column names
            
        Returns:
            Dictionary mapping original column names to standard names
        """
        if df.empty:
            return {}
            
        columns = [col.lower().strip() for col in df.columns.tolist()]
        original_columns = df.columns.tolist()
        mappings = {}
        
        # Define common variations for each level
        variations_map = {
            'segment_code': ['segment code', 'segment_code', 'seg code', 'seg_code', 'segment_id', 'segmentcode'],
            'segment_name': ['segment name', 'segment_name', 'segment title', 'segment_title', 'segment', 'seg_desc', 'seg_name', 'segmentname'],
            'family_code': ['family code', 'family_code', 'fam code', 'fam_code', 'family_id', 'familycode'],
            'family_name': ['family name', 'family_name', 'family title', 'family_title', 'family', 'fam_desc', 'fam_name', 'familyname'],
            'class_code': ['class code', 'class_code', 'cls code', 'cls_code', 'class_id', 'classcode'],
            'class_name': ['class name', 'class_name', 'class title', 'class_title', 'class', 'cls_desc', 'cls_name', 'classname'],
            'commodity_code': ['commodity code', 'commodity_code', 'comm code', 'comm_code', 'commodity_id', 'unspsc code', 'unspsc_code', 'commoditycode'],
            'commodity_title': ['commodity name', 'commodity_name', 'commodity title', 'commodity_title', 'commodity', 'comm_desc', 'comm_name', 'description', 'commodityname', 'commoditytitle']
        }

        found_mappings = {}
        
        for i, original_col in enumerate(original_columns):
            col_lower = columns[i]
            
            for standard_name, variations_list in variations_map.items():
                if standard_name in found_mappings:
                    continue
                
                # Check for exact matches first, then partial matches
                for variation in variations_list:
                    if (col_lower == variation or 
                        (variation in col_lower and len(variation) > 3) or
                        (col_lower in variation and len(col_lower) > 3)):
                        
                        # Prefer more specific/longer matches
                        if (standard_name not in found_mappings or 
                            len(original_col) > len(found_mappings[standard_name])):
                            found_mappings[standard_name] = original_col
                            mappings[original_col] = standard_name
                        break

        logger.info(f"Detected column mappings: {mappings}")
        return mappings
    
    def _process_unspsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the UNSPSC DataFrame to create a 'searchable_text' column
        and ensures a 'class_name' column exists for classification targets.
        
        Args:
            df: Input DataFrame with standardized column names
            
        Returns:
            Processed DataFrame ready for training
        """
        if df.empty:
            return df.copy()
            
        processed_df = df.copy()
        
        # Define standard columns
        standard_cols = [
            'segment_name', 'family_name', 'class_name', 'commodity_title', 
            'segment_code', 'family_code', 'class_code', 'commodity_code'
        ]
        
        # Ensure all standard columns exist
        for col in standard_cols:
            if col not in processed_df.columns:
                processed_df[col] = ""
            processed_df[col] = processed_df[col].apply(self._safe_str_convert)
        
        # Create searchable text from available descriptive fields
        text_source_cols = [
            'commodity_title', 'class_name', 'family_name', 'segment_name',
            'commodity_code', 'class_code', 'family_code', 'segment_code'
        ]
        
        def create_searchable_text(row):
            components = []
            for col in text_source_cols:
                if col in row.index and row[col] and row[col].strip():
                    components.append(row[col].strip())
            return ' '.join(components)

        # Check if we have any usable text columns
        has_text_data = any(
            col in processed_df.columns and 
            not processed_df[col].eq('').all() 
            for col in text_source_cols
        )
        
        if has_text_data:
            processed_df['searchable_text'] = processed_df.apply(create_searchable_text, axis=1)
            processed_df['searchable_text'] = processed_df['searchable_text'].apply(self._clean_text)
        else:
            logger.warning("No suitable text columns found for creating 'searchable_text'.")
            processed_df['searchable_text'] = ""
        
        # Determine the best classification target column
        target_candidates = ['class_name', 'commodity_title', 'family_name', 'segment_name']
        
        resolved_class_col = None
        for candidate in target_candidates:
            if (candidate in processed_df.columns and 
                not processed_df[candidate].eq('').all() and
                processed_df[candidate].notna().sum() > 0):
                
                if candidate != 'class_name':
                    processed_df['class_name'] = processed_df[candidate]
                resolved_class_col = candidate
                logger.info(f"Using '{candidate}' as the primary 'class_name' for classification.")
                break
        
        if resolved_class_col is None:
            logger.warning("Could not determine a suitable 'class_name' column for classification.")
            processed_df['class_name'] = ""
        
        # Filter out invalid records
        initial_count = len(processed_df)
        processed_df = processed_df[
            (processed_df['searchable_text'].str.len() > 3) &
            (processed_df['class_name'].notna()) &
            (processed_df['class_name'].str.len() > 0)
        ].copy()
        
        filtered_count = initial_count - len(processed_df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} records with insufficient data.")
        
        logger.info(f"Processed UNSPSC dataset shape: {processed_df.shape}")
        if not processed_df.empty:
            logger.info(f"Sample searchable text: {processed_df['searchable_text'].head(3).tolist()}")
            logger.info(f"Sample class names: {processed_df['class_name'].head(3).tolist()}")
        
        return processed_df.reset_index(drop=True)
    
    def prepare_training_data(self, df: Optional[pd.DataFrame] = None, 
                            text_column: str = 'searchable_text', 
                            label_column: str = 'class_name') -> pd.DataFrame:
        """
        Prepares training data from the processed DataFrame.
        
        Args:
            df: DataFrame to prepare (uses self.unspsc_data if None)
            text_column: Name of the text feature column
            label_column: Name of the label column
            
        Returns:
            DataFrame ready for model training with numerical labels
        """
        # Use provided DataFrame or default to processed data
        if df is None:
            if self.unspsc_data is None or self.unspsc_data['processed_df'].empty:
                logger.error("No data available for training preparation.")
                return pd.DataFrame()
            df = self.unspsc_data['processed_df']
        
        if df.empty:
            logger.warning("Input DataFrame is empty.")
            return pd.DataFrame()
        
        # Validate required columns exist
        missing_cols = [col for col in [text_column, label_column] if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Filter valid records
        valid_df = df[
            (df[text_column].notna()) & 
            (df[label_column].notna()) &
            (df[text_column].str.len() > 5) &
            (df[label_column].str.len() > 0)
        ].copy()
        
        if valid_df.empty:
            logger.warning("No valid records found after filtering.")
            return pd.DataFrame()
        
        # Remove duplicates
        initial_count = len(valid_df)
        valid_df = valid_df.drop_duplicates(subset=[text_column, label_column])
        duplicate_count = initial_count - len(valid_df)
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate records.")
        
        # Create label mappings
        unique_labels = sorted(valid_df[label_column].unique())
        self.unspsc_label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.unspsc_idx_to_label = {idx: label for label, idx in self.unspsc_label_to_idx.items()}
        
        # Add numerical labels
        valid_df['label'] = valid_df[label_column].map(self.unspsc_label_to_idx)
        
        logger.info(f"Prepared {len(valid_df)} training samples with {len(unique_labels)} unique classes.")
        
        # Show class distribution
        class_counts = valid_df['label'].value_counts()
        logger.info(f"Class distribution (top 10): {class_counts.head(10).to_dict()}")
        
        return valid_df.reset_index(drop=True)

    def initialize_unspsc_vectorizer(self, df: Optional[pd.DataFrame] = None, 
                                   text_column: str = 'searchable_text') -> bool:
        """
        Initializes TF-IDF vectorizer and transforms UNSPSC descriptions.
        
        Args:
            df: DataFrame to vectorize (uses processed data if None)
            text_column: Column containing text to vectorize
            
        Returns:
            True if successful, False otherwise
        """
        if df is None:
            if self.unspsc_data is None or self.unspsc_data['processed_df'].empty:
                logger.warning("No data available for vectorizer initialization.")
                return False
            df = self.unspsc_data['processed_df']
        
        if df.empty or text_column not in df.columns:
            logger.warning(f"DataFrame is empty or missing '{text_column}' column.")
            return False
        
        try:
            # Filter out empty texts
            valid_texts = df[df[text_column].str.len() > 0][text_column]
            
            if valid_texts.empty:
                logger.warning("No valid texts found for vectorization.")
                return False
            
            self.vectorizer = TfidfVectorizer(
                stop_words='english', 
                ngram_range=(1, 2), 
                max_df=0.8, 
                min_df=2,  # Reduced from 5 to handle smaller datasets
                max_features=10000  # Limit features to prevent memory issues
            )
            
            self.unspsc_vectors = self.vectorizer.fit_transform(valid_texts)
            
            logger.info(f"UNSPSC vectorizer initialized with {len(self.vectorizer.vocabulary_)} features.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vectorizer: {e}")
            return False

    def get_unspsc_by_description(self, description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Finds the top_k best matching UNSPSC codes for a given description.
        
        Args:
            description: Text description to match
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries containing match information
        """
        if (self.vectorizer is None or self.unspsc_vectors is None or 
            self.unspsc_data is None):
            logger.warning("UNSPSC vectorizer not initialized.")
            return []

        cleaned_description = self._clean_text(description)
        if not cleaned_description:
            logger.warning("Empty description provided.")
            return []

        try:
            desc_vector = self.vectorizer.transform([cleaned_description])
            similarities = cosine_similarity(desc_vector, self.unspsc_vectors).flatten()

            top_indices = similarities.argsort()[-top_k:][::-1]
            results = []
            processed_df = self.unspsc_data['processed_df']
            
            for i in top_indices:
                score = similarities[i]
                if score > 0.01:  # Minimum similarity threshold
                    row = processed_df.iloc[i]
                    results.append({
                        "segment_name": row.get('segment_name', 'N/A'),
                        "family_name": row.get('family_name', 'N/A'),
                        "class_name": row.get('class_name', 'N/A'),
                        "commodity_title": row.get('commodity_title', 'N/A'),
                        "commodity_code": row.get('commodity_code', 'N/A'),
                        "similarity_score": float(score)  # Ensure JSON serializable
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Provides comprehensive statistics about the loaded UNSPSC data.
        
        Returns:
            Dictionary containing various statistics
        """
        if self.unspsc_data is None or self.unspsc_data.get('processed_df') is None:
            return {"error": "No UNSPSC data loaded."}

        df = self.unspsc_data['processed_df']
        
        if df.empty:
            return {"error": "UNSPSC data is empty."}

        try:
            stats = {
                "total_records": len(df),
                "columns_available": df.columns.tolist(),
                "unique_segments": df['segment_name'].nunique() if 'segment_name' in df.columns else 0,
                "unique_families": df['family_name'].nunique() if 'family_name' in df.columns else 0,
                "unique_classes": df['class_name'].nunique() if 'class_name' in df.columns else 0,
                "unique_commodities": df['commodity_title'].nunique() if 'commodity_title' in df.columns else 0,
                "has_searchable_text": 'searchable_text' in df.columns,
                "non_empty_searchable_text": (df['searchable_text'].str.len() > 0).sum() if 'searchable_text' in df.columns else 0,
                "average_text_length": df['searchable_text'].str.len().mean() if 'searchable_text' in df.columns else 0
            }
            
            # Add sample data if available
            if not df.empty and 'searchable_text' in df.columns:
                sample_size = min(3, len(df))
                stats["sample_searchable_text"] = df['searchable_text'].sample(sample_size).tolist()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {"error": f"Failed to generate statistics: {str(e)}"}

    def train_test_split_data(self, df: Optional[pd.DataFrame] = None, 
                            test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the prepared training data into train and test sets.
        
        Args:
            df: DataFrame to split (uses prepared training data if None)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if df is None:
            df = self.prepare_training_data()
        
        if df.empty:
            logger.warning("No data available for train/test split.")
            return pd.DataFrame(), pd.DataFrame()
        
        if 'label' not in df.columns:
            logger.error("DataFrame missing 'label' column required for stratified split.")
            return df, pd.DataFrame()
        
        try:
            # Perform stratified split to maintain class distribution
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=df['label']
            )
            
            logger.info(f"Split data into {len(train_df)} training and {len(test_df)} test samples.")
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
            
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split instead.")
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state
            )
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    """Example usage and testing function."""
    logger.info("--- DataPreparationUtils Standalone Test ---")

    data_prep = DataPreparationUtils()
    
    # Test with example file path - adjust as needed
    test_file = "data/data-unspsc-codes.csv"
    
    # Load data
    unspsc_data = data_prep.load_unspsc_data(test_file)
    
    if unspsc_data and not unspsc_data['processed_df'].empty:
        # Test data preparation
        training_data = data_prep.prepare_training_data()
        
        if not training_data.empty:
            logger.info(f"Successfully prepared {len(training_data)} records for training.")
            
            # Test train/test split
            train_df, test_df = data_prep.train_test_split_data(training_data)
            logger.info(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
            
            # Test similarity search
            if data_prep.initialize_unspsc_vectorizer():
                test_description = "office stationery supplies"
                matches = data_prep.get_unspsc_by_description(test_description, top_k=3)
                
                if matches:
                    logger.info(f"\nTop 3 UNSPSC matches for '{test_description}':")
                    for match in matches:
                        logger.info(f"  {match['commodity_title']} - Score: {match['similarity_score']:.3f}")
        
        # Show statistics
        stats = data_prep.get_statistics()
        logger.info(f"\nDataset Statistics:\n{json.dumps(stats, indent=2)}")
    else:
        logger.error("Failed to load UNSPSC data for testing.")


if __name__ == "__main__":
    main()