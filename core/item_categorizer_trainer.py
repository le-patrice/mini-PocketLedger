import os
import multiprocessing
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings
from datetime import datetime
import requests
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set multiprocessing and tokenizer settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass


class DataPreparationUtils:
    """Optimized data preparation utilities for UNSPSC dataset."""
    
    def __init__(self):
        self.unspsc_data = None
        self.unspsc_label_to_idx = {}
        self.unspsc_idx_to_label = {}
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def load_unspsc_data(self, filepath_or_url: str) -> Dict[str, pd.DataFrame]:
        """
        Load UNSPSC data from local file or URL with automatic column mapping.
        """
        try:
            # Try to load from URL first, then local file
            if filepath_or_url.startswith('http'):
                logger.info(f"Loading UNSPSC data from URL: {filepath_or_url}")
                response = requests.get(filepath_or_url)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
            else:
                logger.info(f"Loading UNSPSC data from local file: {filepath_or_url}")
                df = pd.read_csv(filepath_or_url)
            
            # Automatic column mapping for different CSV formats
            column_mappings = self._detect_column_mappings(df)
            df = df.rename(columns=column_mappings)
            
            # Process the data
            processed_df = self._process_unspsc_data(df)
            
            self.unspsc_data = {
                'raw_df': df,
                'processed_df': processed_df
            }
            
            logger.info(f"Successfully loaded {len(processed_df)} UNSPSC records")
            return self.unspsc_data
            
        except Exception as e:
            logger.error(f"Failed to load UNSPSC data: {e}")
            return {'raw_df': pd.DataFrame(), 'processed_df': pd.DataFrame()}
    
    def _detect_column_mappings(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect and map column names to standard format.
        """
        columns = df.columns.tolist()
        mappings = {}
        
        # Common column name variations
        segment_variations = ['segment', 'segment_code', 'segment code', 'seg', 'segment_name', 'segment name']
        family_variations = ['family', 'family_code', 'family code', 'fam', 'family_name', 'family name']
        class_variations = ['class', 'class_code', 'class code', 'cls', 'class_name', 'class name']
        commodity_variations = ['commodity', 'commodity_code', 'commodity code', 'comm', 'commodity_name', 'commodity name']
        title_variations = ['title', 'description', 'name', 'text', 'commodity_title', 'commodity title']
        
        # Find matching columns (case-insensitive)
        for col in columns:
            col_lower = col.lower().strip()
            
            if any(var in col_lower for var in segment_variations):
                if 'code' in col_lower:
                    mappings[col] = 'segment_code'
                else:
                    mappings[col] = 'segment_name'
            elif any(var in col_lower for var in family_variations):
                if 'code' in col_lower:
                    mappings[col] = 'family_code'
                else:
                    mappings[col] = 'family_name'
            elif any(var in col_lower for var in class_variations):
                if 'code' in col_lower:
                    mappings[col] = 'class_code'
                else:
                    mappings[col] = 'class_name'
            elif any(var in col_lower for var in commodity_variations):
                if 'code' in col_lower:
                    mappings[col] = 'commodity_code'
                elif any(var in col_lower for var in title_variations):
                    mappings[col] = 'commodity_title'
                else:
                    mappings[col] = 'commodity_name'
            elif any(var in col_lower for var in title_variations):
                mappings[col] = 'commodity_title'
        
        logger.info(f"Detected column mappings: {mappings}")
        return mappings
    
    def _process_unspsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process UNSPSC data to create searchable text and standardize format.
        """
        processed_df = df.copy()
        
        # Create searchable text from available columns
        text_columns = []
        if 'commodity_title' in processed_df.columns:
            text_columns.append('commodity_title')
        if 'commodity_name' in processed_df.columns:
            text_columns.append('commodity_name')
        if 'class_name' in processed_df.columns:
            text_columns.append('class_name')
        if 'family_name' in processed_df.columns:
            text_columns.append('family_name')
        
        if not text_columns:
            # Fallback: use first text column
            text_cols = processed_df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                text_columns = [text_cols[0]]
        
        # Create searchable text
        if text_columns:
            processed_df['searchable_text'] = processed_df[text_columns].fillna('').agg(' '.join, axis=1)
            processed_df['searchable_text'] = processed_df['searchable_text'].apply(self._clean_text)
        else:
            logger.error("No suitable text columns found for creating searchable text")
            processed_df['searchable_text'] = ""
        
        # Ensure we have class_name for classification target
        if 'class_name' not in processed_df.columns:
            if 'commodity_name' in processed_df.columns:
                processed_df['class_name'] = processed_df['commodity_name']
            elif 'commodity_title' in processed_df.columns:
                processed_df['class_name'] = processed_df['commodity_title']
            else:
                logger.warning("No suitable target column found, using first available text column")
                text_cols = processed_df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    processed_df['class_name'] = processed_df[text_cols[0]]
        
        # Remove empty records
        processed_df = processed_df[processed_df['searchable_text'].str.len() > 0]
        processed_df = processed_df[processed_df['class_name'].notna()]
        
        return processed_df.reset_index(drop=True)
    
    def prepare_training_data(self, df: pd.DataFrame, text_column: str = 'searchable_text', 
                            label_column: str = 'class_name') -> pd.DataFrame:
        """
        Prepare training data with label encoding.
        """
        # Filter valid records
        valid_df = df[(df[text_column].notna()) & (df[label_column].notna())].copy()
        valid_df = valid_df[valid_df[text_column].str.len() > 5]
        
        # Create label mappings
        unique_labels = sorted(valid_df[label_column].unique())
        self.unspsc_label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.unspsc_idx_to_label = {idx: label for label, idx in self.unspsc_label_to_idx.items()}
        
        # Add numerical labels
        valid_df['label'] = valid_df[label_column].map(self.unspsc_label_to_idx)
        
        logger.info(f"Prepared {len(valid_df)} training samples with {len(unique_labels)} unique classes")
        return valid_df


class ItemCategoryDataset(Dataset):
    """Custom Dataset for UNSPSC item classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ItemCategorizerTrainer:
    """
    Optimized trainer for UNSPSC item classification using HuggingFace Transformers.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.unspsc_data_df: Optional[pd.DataFrame] = None
        self.data_utils = DataPreparationUtils()
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the HuggingFace tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Tokenizer initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def prepare_unspsc_training_data(self, unspsc_filepath: str) -> pd.DataFrame:
        """Load and prepare UNSPSC data for training."""
        logger.info(f"Loading UNSPSC data from {unspsc_filepath}...")
        
        # Load data using the optimized data utils
        unspsc_loaded_data = self.data_utils.load_unspsc_data(unspsc_filepath)
        
        if not unspsc_loaded_data or unspsc_loaded_data.get("processed_df") is None or unspsc_loaded_data["processed_df"].empty:
            logger.error("Failed to load or process UNSPSC data")
            return pd.DataFrame()

        df_for_training = unspsc_loaded_data["processed_df"].copy()
        
        # Prepare training data with labels
        processed_df = self.data_utils.prepare_training_data(
            df_for_training,
            text_column='searchable_text',
            label_column='class_name'
        )
        
        # Transfer mappings
        self.label_to_idx = self.data_utils.unspsc_label_to_idx
        self.idx_to_label = self.data_utils.unspsc_idx_to_label
        self.unspsc_data_df = df_for_training

        if processed_df.empty:
            logger.error("No valid training data after preparation")
            return pd.DataFrame()

        # Clean training data
        cleaned_df = self._clean_training_data(processed_df)
        logger.info(f"Prepared {len(cleaned_df)} training samples for {len(self.label_to_idx)} classes")
        return cleaned_df

    def _clean_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate training data."""
        if df.empty:
            return pd.DataFrame()
            
        # Remove duplicates and short texts
        df = df.drop_duplicates(subset=["searchable_text", "label"])
        df = df[df["searchable_text"].str.len() >= 5]
        
        # Filter low-frequency classes
        class_counts = df["label"].value_counts()
        min_samples = 3
        valid_labels = class_counts[class_counts >= min_samples].index
        df = df[df["label"].isin(valid_labels)]
        
        # Re-encode labels if needed
        if df['label'].nunique() < len(self.label_to_idx):
            remaining_classes = df['class_name'].unique()
            self.label_to_idx = {cls: idx for idx, cls in enumerate(sorted(remaining_classes))}
            self.idx_to_label = {idx: cls for cls, idx in self.label_to_idx.items()}
            df['label'] = df['class_name'].map(self.label_to_idx)
            logger.info(f"Re-encoded labels: {len(self.label_to_idx)} classes")
        
        return df.reset_index(drop=True)

    def create_model(self, num_labels: int):
        """Create and configure the model."""
        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=self.idx_to_label,
                label2id=self.label_to_idx,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, config=config
            )
            
            if len(self.tokenizer) > self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))

            logger.info(f"Model created with {num_labels} labels")
            return self.model
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        mask = labels != -100
        true_labels = labels[mask]
        true_predictions = predictions[mask]
        
        if len(true_labels) == 0:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "f1_macro": f1_score(true_labels, true_predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(true_labels, true_predictions, average="weighted", zero_division=0),
        }

    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, epochs: int = 3) -> Optional[Trainer]:
        """Train the model with optimized parameters."""
        if df.empty or 'label' not in df.columns:
            logger.error("Invalid training data")
            return None

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["searchable_text"].tolist(),
            df["label"].tolist(),
            test_size=test_size,
            random_state=42,
            stratify=df["label"],
        )

        # Create datasets
        train_dataset = ItemCategoryDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = ItemCategoryDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        # Create model
        self.create_model(len(self.label_to_idx))

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            report_to="none",
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
            seed=42,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")
        return trainer

    def predict_category(self, text: str) -> Dict[str, Any]:
        """Predict UNSPSC category for given text."""
        if not all([self.model, self.tokenizer, self.unspsc_data_df is not None]):
            return {"error": "Model not properly loaded"}

        try:
            clean_text = self.data_utils._clean_text(text)
            inputs = self.tokenizer(
                clean_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_id = predictions.argmax().item()
            confidence = predictions.max().item()
            predicted_class = self.idx_to_label.get(predicted_id, "UNKNOWN")

            # Get hierarchical details
            details = {"Class Name": predicted_class}
            if self.unspsc_data_df is not None:
                matches = self.unspsc_data_df[self.unspsc_data_df['class_name'] == predicted_class]
                if not matches.empty:
                    first_match = matches.iloc[0]
                    details.update({
                        "Segment Name": first_match.get('segment_name', 'N/A'),
                        "Family Name": first_match.get('family_name', 'N/A'),
                        "Class Name": first_match.get('class_name', predicted_class),
                        "Commodity Name": first_match.get('commodity_name', 'N/A')
                    })

            return {
                "predicted_category": predicted_class,
                "confidence": float(confidence),
                **details
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e), "predicted_category": "UNKNOWN", "confidence": 0.0}

    def save_model(self, trainer: Trainer, path: str = "models/unspsc_item_classifier"):
        """Save the trained model and metadata."""
        model_path = Path(path)
        model_path.mkdir(exist_ok=True, parents=True)

        try:
            trainer.save_model(str(model_path))
            self.tokenizer.save_pretrained(str(model_path))

            mappings = {
                "label_to_idx": self.label_to_idx,
                "idx_to_label": self.idx_to_label,
                "model_name": self.model_name,
                "max_length": self.max_length,
                "num_labels": len(self.label_to_idx),
                "training_timestamp": datetime.now().isoformat(),
                "unspsc_data_df_json": self.unspsc_data_df.to_json(orient='records') if self.unspsc_data_df is not None else None
            }

            with open(model_path / "label_mappings.json", "w", encoding="utf-8") as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)

            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    @classmethod
    def load_model(cls, path: str) -> 'ItemCategorizerTrainer':
        """Load a trained model."""
        model_path = Path(path)
        mappings_file = model_path / "label_mappings.json"

        with open(mappings_file, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        trainer = cls(
            model_name=mappings.get("model_name", "distilbert-base-uncased"),
            max_length=mappings.get("max_length", 512)
        )

        trainer.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        trainer.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        trainer.label_to_idx = mappings.get("label_to_idx", {})
        trainer.idx_to_label = {int(k): v for k, v in mappings.get("idx_to_label", {}).items()}

        if mappings.get("unspsc_data_df_json"):
            trainer.unspsc_data_df = pd.read_json(StringIO(mappings["unspsc_data_df_json"]), orient='records')

        logger.info(f"Model loaded from {model_path}")
        return trainer


def train_unspsc_item_classifier():
    """Main training function with the new dataset URL."""
    model_save_path = "models/unspsc_item_classifier"

    # Check for existing model
    if (Path(model_save_path) / "pytorch_model.bin").exists():
        logger.info(f"Found existing model at {model_save_path}")
        try:
            trainer = ItemCategorizerTrainer.load_model(model_save_path)
            dummy_hf_trainer = Trainer(model=trainer.model, args=TrainingArguments(output_dir="./tmp"))
            return dummy_hf_trainer, trainer
        except Exception as e:
            logger.error(f"Failed to load existing model: {e}")

    try:
        trainer = ItemCategorizerTrainer(model_name="distilbert-base-uncased")

        # Use the new dataset URL
        unspsc_url = "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/unspsc-codes.csv"
        df_prepared = trainer.prepare_unspsc_training_data(unspsc_url)

        if df_prepared.empty:
            logger.error("No training data available")
            return None, None

        # Train model
        trained_trainer = trainer.train_model(df_prepared, epochs=3)

        if trained_trainer is None:
            logger.error("Training failed")
            return None, None

        # Save model
        trainer.save_model(trained_trainer, model_save_path)
        logger.info(f"Training completed! Model saved with {len(trainer.label_to_idx)} classes.")

        return trained_trainer, trainer

    except Exception as e:
        logger.critical(f"Training failed: {e}")
        return None, None


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True, parents=True)
    
    logger.info("--- Starting UNSPSC Item Categorizer Training ---")
    hf_trainer, custom_trainer = train_unspsc_item_classifier()

    if hf_trainer and custom_trainer:
        logger.info("Model ready for use!")
        
        # Test prediction
        test_text = "office stationery and pens"
        result = custom_trainer.predict_category(test_text)
        logger.info(f"Test prediction for '{test_text}':")
        logger.info(f"Category: {result['predicted_category']} (Confidence: {result['confidence']:.3f})")
        
        for key, value in result.items():
            if key not in ['predicted_category', 'confidence', 'error']:
                logger.info(f"  {key}: {value}")
    else:
        logger.error("Training failed!")