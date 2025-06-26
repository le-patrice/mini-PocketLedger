import os
# IMPORTANT: Disable tokenizers parallelism to prevent multiprocessing issues with FastAI
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastai.text.all import *
from fastai.text.all import text_classifier_learner
from fastai.learner import Learner, load_learner
from fastai.text.models import AWD_LSTM
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import ColReader, RandomSplitter
from fastai.text.data import TextBlock
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
import numpy as np
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import logging
import matplotlib.pyplot as plt
from data_preparation_utils import DataPreparationUtils # Ensure this import is correct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PSCClassifierTrainer:
    """FastAI-based trainer for PSC text classification using pre-trained transformers."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.psc_to_idx = {}
        self.idx_to_psc = {}
        self.data_utils = DataPreparationUtils()

    def prepare_psc_training_data(
        self, 
        psc_data: Dict, 
        additional_descriptions: Optional[List[Dict[str, str]]] = None, 
        augment_data: bool = True
    ) -> pd.DataFrame:
        """
        Prepare comprehensive training data from PSC mapping.
        Ensures all text fields are correctly handled as strings.
        """
        training_samples = []
        
        if not psc_data or 'psc_mapping' not in psc_data:
            raise ValueError("Invalid PSC data provided: 'psc_mapping' key is missing or empty.")
        
        psc_mapping = psc_data['psc_mapping']
        
        # Generate training samples from PSC mapping
        for psc_code, psc_info in psc_mapping.items():
            # Ensure all retrieved values are strings, even if they are None or other types
            psc_code_str = str(psc_info.get('psc', '')) 
            short_name_str = str(psc_info.get('shortName', ''))
            spend_category_str = str(psc_info.get('spendCategoryTitle', ''))
            portfolio_group_str = str(psc_info.get('portfolioGroup', ''))
            long_name_str = str(psc_info.get('longName', ''))
            
            # Handle lists of strings (examplePhrases, synonyms)
            example_phrases_list = [str(p) for p in psc_info.get('examplePhrases', []) if p is not None]
            synonyms_list = [str(s) for s in psc_info.get('synonyms', []) if s is not None]

            # Combine various text fields for a comprehensive base description
            base_description_parts = []
            if short_name_str: base_description_parts.append(short_name_str)
            if spend_category_str and spend_category_str != short_name_str: base_description_parts.append(spend_category_str)
            if portfolio_group_str and portfolio_group_str not in base_description_parts: base_description_parts.append(portfolio_group_str)
            if long_name_str and long_name_str not in base_description_parts: base_description_parts.append(long_name_str)
            
            # Add synonyms and example phrases as individual samples or combined
            if synonyms_list:
                base_description_parts.extend(synonyms_list)
            if example_phrases_list:
                base_description_parts.extend(example_phrases_list)

            # Create base training sample by joining unique parts
            # Filter out empty strings before joining, then strip final result
            unique_base_parts = list(dict.fromkeys([p.strip() for p in base_description_parts if p.strip()]))
            base_text = ". ".join(unique_base_parts)

            if base_text:
                training_samples.append({
                    'text': base_text,
                    'psc': psc_code_str,
                    'source': 'combined_base'
                })
            
            # Generate augmented samples if enabled
            if augment_data:
                augmented_samples = self._generate_augmented_samples(
                    psc_code=psc_code_str,
                    short_name=short_name_str,
                    category=spend_category_str,
                    portfolio=portfolio_group_str,
                    long_name=long_name_str,
                    synonyms=synonyms_list,
                    example_phrases=example_phrases_list
                )
                for aug_sample in augmented_samples:
                    if isinstance(aug_sample.get('text'), str) and aug_sample['text'].strip():
                        training_samples.append(aug_sample)
                    else:
                        logger.warning(f"Skipping empty/invalid augmented sample for PSC {psc_code_str}: {aug_sample}")
        
        # Add additional descriptions if provided
        if additional_descriptions:
            for desc_item in additional_descriptions:
                if isinstance(desc_item, dict) and 'text' in desc_item and 'psc' in desc_item:
                    text_to_add = str(desc_item['text']).strip()
                    psc_to_add = str(desc_item['psc']).strip()
                    if text_to_add:
                        training_samples.append({
                            'text': text_to_add,
                            'psc': psc_to_add,
                            'source': 'additional_provided'
                        })
                else:
                    logger.warning(f"Skipping invalid additional description item: {desc_item}")
        
        df = pd.DataFrame(training_samples)
        
        # Ensure the 'text' column contains only strings and handle NaNs or non-strings if any slip through
        df['text'] = df['text'].fillna('').astype(str)
        df['psc'] = df['psc'].fillna('').astype(str)

        # Clean and validate data
        df = self._clean_training_data(df)
        
        logger.info(f"Generated {len(df)} training samples for {df['psc'].nunique()} PSC classes.")
        return df

    def _generate_augmented_samples(
        self, 
        psc_code: str, 
        short_name: str, 
        category: str, 
        portfolio: str,
        long_name: str,
        synonyms: List[str],
        example_phrases: List[str]
    ) -> List[Dict]:
        """Generate augmented training samples for better coverage."""
        samples = []
        
        business_templates = [
            "Purchase of {item}", "Procurement of {item}", "Supply of {item}",
            "Acquisition of {item}", "{item} services", "{item} and related services",
            "Professional {item}", "{item} equipment", "{item} supplies",
            "Consulting on {item}", "Maintenance for {item}", "Repair of {item}",
            "Installation of {item}", "Lease of {item}", "Contract for {item}",
            "Development of {item}", "Support for {item}", "Software for {item}",
            "Hardware for {item}", "Advisory on {item}"
        ]
        
        base_terms = set([
            self._clean_text_for_augmentation(short_name),
            self._clean_text_for_augmentation(category),
            self._clean_text_for_augmentation(portfolio),
            self._clean_text_for_augmentation(long_name)
        ])
        base_terms.update([self._clean_text_for_augmentation(s) for s in synonyms])
        base_terms.update([self._clean_text_for_augmentation(p) for p in example_phrases])
        
        base_terms = {term for term in base_terms if term.strip()}

        for term in base_terms:
            lower_term = term.lower()
            for template in business_templates:
                augmented_text = template.format(item=lower_term).strip()
                if augmented_text:
                    samples.append({
                        'text': augmented_text,
                        'psc': psc_code,
                        'source': 'augmented_template'
                    })
        
        return samples

    def _clean_text_for_augmentation(self, text: Any) -> str:
        """
        Clean text for augmentation purposes, ensuring input is string and handles None.
        """
        text_str = str(text).strip() if text is not None else ""
        cleaned = re.sub(r'[^\w\s-]', '', text_str)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _clean_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate training data."""
        df = df[df['text'].str.strip() != '']
        df = df.drop_duplicates(subset=['text', 'psc'])
        df = df[df['text'].str.len() >= 3]
        
        psc_counts = df['psc'].value_counts()
        valid_pscs = psc_counts[psc_counts >= 2].index 
        df = df[df['psc'].isin(valid_pscs)]
        
        if df.empty:
            logger.warning("No valid training data remaining after cleaning and filtering.")
            return pd.DataFrame(columns=['text', 'psc', 'source'])
            
        logger.info(f"Cleaned data: {len(df)} samples remaining for training.")
        return df.reset_index(drop=True)

    def prepare_data(
        self, 
        psc_data: Dict, 
        additional_descriptions: Optional[List[Dict[str, str]]] = None
    ) -> pd.DataFrame:
        """Main data preparation method."""
        df = self.prepare_psc_training_data(psc_data, additional_descriptions)
        
        if df.empty:
            logger.error("DataFrame is empty after data preparation. Cannot create label mappings.")
            return df

        unique_pscs = sorted(df['psc'].unique())
        self.psc_to_idx = {psc: idx for idx, psc in enumerate(unique_pscs)}
        self.idx_to_psc = {idx: psc for psc, idx in self.psc_to_idx.items()}
        
        df['label'] = df['psc'].map(self.psc_to_idx)
        
        logger.info(f"Prepared {len(df)} training samples for {len(unique_pscs)} PSC classes.")
        logger.info(f"Sample distribution (top 10):\n{df['psc'].value_counts().head(10)}")
        
        return df

    def create_dataloaders(
        self, 
        df: pd.DataFrame, 
        valid_pct: float = 0.2, 
        bs: int = 16
    ) -> DataLoaders:
        """Create FastAI text dataloaders with proper tokenization."""
        
        if len(df) < 10:
            logger.warning(f"Insufficient data for robust train/val split: only {len(df)} samples available. Validation might be skipped or poor.")
            valid_pct = 0.0

        text_block = TextBlock.from_df(
            'text',
            seq_len=128,
            tok=self.tokenizer,
            is_lm=False
        )
        
        category_block = CategoryBlock(vocab=list(self.idx_to_psc.values()))
        
        dblock = DataBlock(
            blocks=(text_block, category_block),
            get_x=ColReader('text'),
            get_y=ColReader('psc'),
            splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
            item_tfms=[],
            batch_tfms=[]
        )
        
        try:
            # Setting num_workers=0 is crucial for compatibility with HuggingFace tokenizers
            dls = dblock.dataloaders(df, bs=bs, num_workers=0) 
            logger.info(f"Dataloaders created successfully with batch size {bs}.")
            return dls
        except Exception as e:
            logger.error(f"Error creating dataloaders with batch size {bs}: {e}")
            if bs > 1:
                new_bs = max(1, bs // 2)
                logger.info(f"Trying with smaller batch size: {new_bs}...")
                try:
                    dls = dblock.dataloaders(df, bs=new_bs, num_workers=0)
                    logger.info(f"Dataloaders created successfully with smaller batch size {new_bs}.")
                    return dls
                except Exception as e2:
                    logger.error(f"Failed again with smaller batch size {new_bs}: {e2}")
                    raise RuntimeError("Failed to create dataloaders even with smaller batch size.") from e2
            else:
                raise RuntimeError("Failed to create dataloaders with batch size 1.") from e

    def create_learner(self, dls: DataLoaders) -> Learner:
        """Create FastAI text learner with pre-trained transformer."""
        
        def accuracy_func(inp, targ):
            return (inp.argmax(dim=-1) == targ).float().mean()
        
        def f1_score_func(inp, targ):
            preds = inp.argmax(dim=-1)
            preds_np = preds.cpu().numpy()
            targ_np = targ.cpu().numpy()
            return f1_score(targ_np, preds_np, average='weighted', zero_division=0)
        
        metrics = [accuracy_func, f1_score_func]
        
        learner = text_classifier_learner(
            dls,
            arch=AWD_LSTM,
            metrics=metrics,
            drop_mult=0.5,
            path='.'
        )
        
        logger.info("FastAI text learner created.")
        return learner

    def train_model(
        self, 
        df: pd.DataFrame, 
        epochs: int = 5, 
        lr: float = 1e-3
    ) -> Learner:
        """Complete training pipeline with FastAI optimizations."""
        
        logger.info("Creating dataloaders for PSC classifier training...")
        dls = self.create_dataloaders(df)
        
        logger.info("Sample batch (first 3 items):")
        try:
            dls.show_batch(max_n=3)
        except Exception as e:
            logger.warning(f"Could not show batch: {e}")
        
        logger.info("Creating text learner for PSC classifier...")
        learner = self.create_learner(dls)
        
        logger.info("Finding optimal learning rate...")
        try:
            lr_finder_results = learner.lr_find()
            suggested_lr = lr_finder_results.lr_min
            if suggested_lr > 0 and suggested_lr < lr * 10:
                lr = suggested_lr
                logger.info(f"Using suggested learning rate: {lr:.2e}")
            else:
                logger.warning(f"Suggested LR ({suggested_lr:.2e}) is extreme, sticking to default LR: {lr:.2e}")
        except Exception as e:
            logger.error(f"LR finder failed: {e}, using default LR: {lr:.2e}")
        
        logger.info(f"Training PSC classifier model for {epochs} epochs with lr={lr:.2e}...")
        try:
            learner.fit_one_cycle(epochs, lr)
        except Exception as e:
            logger.error(f"Training failed with fit_one_cycle: {e}. Attempting standard fit.")
            try:
                learner.fit(epochs, lr)
            except Exception as e2:
                logger.error(f"Training failed even with standard fit: {e2}")
                raise
        
        try:
            learner.recorder.plot_loss()
            plt.show()
        except Exception as e:
            logger.warning(f"Could not plot loss: {e}")
        
        return learner

    def evaluate_model(self, learner: Learner, df: Optional[pd.DataFrame] = None):
        """Evaluate model performance on the validation set."""
        logger.info("\nEvaluating PSC classifier model performance...")
        
        if df is None or df.empty:
            logger.info("Using learner's internal validation set for evaluation.")
            try:
                val_preds, val_targets = learner.get_preds(ds_idx=1) 
                
                val_preds_np = val_preds.argmax(dim=1).cpu().numpy()
                val_targets_np = val_targets.cpu().numpy()
                
                val_accuracy = accuracy_score(val_targets_np, val_preds_np)
                val_f1 = f1_score(val_targets_np, val_preds_np, average='weighted', zero_division=0)
                
                logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
                logger.info(f"Validation F1 Score (Weighted): {val_f1:.4f}")

                unique_labels_in_val = np.unique(val_targets_np)
                target_names_val = [
                    self.idx_to_psc[int(i)] 
                    for i in unique_labels_in_val 
                    if int(i) in self.idx_to_psc
                ]
                
                logger.info("\nValidation Set Classification Report:")
                report = classification_report(val_targets_np, val_preds_np, 
                                               target_names=target_names_val, zero_division=0)
                logger.info(report)

            except Exception as e:
                logger.error(f"Error calculating validation metrics: {e}")
        else:
            logger.info("Evaluating on provided DataFrame (treated as test set)...")
            predictions = []
            actuals = []
            
            df['label'] = df['psc'].map(self.psc_to_idx)
            test_df_valid = df[df['label'].notna()]

            if test_df_valid.empty:
                logger.warning("Provided DataFrame is empty or contains no valid PSCs after mapping. Skipping test evaluation.")
                return

            for _, row in test_df_valid.iterrows():
                try:
                    pred_class_str, _, _ = learner.predict(str(row['text']))
                    
                    pred_idx_for_report = self.psc_to_idx.get(pred_class_str)
                    
                    if pred_idx_for_report is not None:
                        predictions.append(pred_idx_for_report)
                        actuals.append(int(row['label']))
                    else:
                        logger.warning(f"Skipping prediction for text '{row['text'][:50]}...': Predicted PSC '{pred_class_str}' not in training vocabulary.")
                except Exception as e:
                    logger.warning(f"Prediction failed for text: '{row['text'][:50]}...': {e}")
                    continue
            
            if predictions and actuals:
                unique_actual_labels = np.unique(actuals)
                target_names_test = [self.idx_to_psc[int(i)] for i in unique_actual_labels if int(i) in self.idx_to_psc]
                
                logger.info("\nTest Set Classification Report:")
                report = classification_report(actuals, predictions, target_names=target_names_test, zero_division=0)
                logger.info(report)
            else:
                logger.warning("No valid predictions or actuals for test set evaluation.")


    def save_model(self, learner: Learner, path: str = "models/psc_classifier"):
        """Save the trained model and mappings."""
        model_path = Path(path)
        model_path.mkdir(exist_ok=True, parents=True)
        
        try:
            learner.export(model_path / "learner.pkl")
            
            with open(model_path / "label_mappings.json", "w", encoding="utf-8") as f:
                json.dump({
                    "psc_to_idx": self.psc_to_idx,
                    "idx_to_psc": {str(k): v for k, v in self.idx_to_psc.items()} 
                }, f, indent=2, ensure_ascii=False)
            
            tokenizer_info = {
                "model_name": self.model_name,
                "vocab_size": len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else None,
                "model_max_length": self.tokenizer.model_max_length,
                "add_prefix_space": self.tokenizer.add_prefix_space,
                "cls_token": str(self.tokenizer.cls_token) if self.tokenizer.cls_token else None,
                "sep_token": str(self.tokenizer.sep_token) if self.tokenizer.sep_token else None,
                "unk_token": str(self.tokenizer.unk_token) if self.tokenizer.unk_token else None,
                "pad_token": str(self.tokenizer.pad_token) if self.tokenizer.pad_token else None,
                "mask_token": str(self.tokenizer.mask_token) if self.tokenizer.mask_token else None
            }
            with open(model_path / "tokenizer_info.json", "w", encoding="utf-8") as f:
                json.dump(tokenizer_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"PSC classifier model and mappings saved to {model_path}.")
            
        except Exception as e:
            logger.error(f"Error saving PSC classifier model: {e}")
            raise

    def predict_psc(self, learner: Learner, text: str, psc_data: Dict) -> Dict:
        """Predict PSC for a given text description."""
        try:
            text_input = str(text) 
            pred_class_str, pred_idx_tensor, probs_tensor = learner.predict(text_input)
            
            predicted_psc_code = str(pred_class_str)
            confidence = float(torch.max(probs_tensor))
            
            psc_mapping = psc_data.get('psc_mapping', {})
            psc_details = psc_mapping.get(predicted_psc_code, {})
            
            return {
                "predicted_psc": predicted_psc_code,
                "confidence": confidence,
                "shortName": psc_details.get("shortName", "N/A"),
                "spendCategoryTitle": psc_details.get("spendCategoryTitle", "N/A"),
                "portfolioGroup": psc_details.get("portfolioGroup", "N/A"),
                "probabilities": {
                    self.idx_to_psc.get(i, f"class_{i}"): float(prob) 
                    for i, prob in enumerate(probs_tensor.tolist())
                }
            }
        except Exception as e:
            logger.error(f"PSC prediction failed for text '{text}': {e}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "predicted_psc": "UNKNOWN",
                "confidence": 0.0,
                "shortName": "Unknown",
                "spendCategoryTitle": "Unknown",
                "portfolioGroup": "Unknown",
                "probabilities": {}
            }

    @classmethod
    def load_trained_model(cls, path: str):
        """Load a previously trained model."""
        model_path = Path(path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist. Ensure the model has been trained and saved.")
        
        learner = load_learner(model_path / "learner.pkl")
        
        with open(model_path / "label_mappings.json", "r", encoding="utf-8") as f:
            mappings = json.load(f)
        
        trainer = cls(model_name="distilbert-base-uncased")
        trainer.psc_to_idx = mappings["psc_to_idx"]
        trainer.idx_to_psc = {int(k): v for k, v in mappings["idx_to_psc"].items()}
        
        logger.info(f"PSC classifier model loaded from {model_path}.")
        return learner, trainer


def train_psc_classifier():
    """Main function to train PSC classifier."""
    trainer = PSCClassifierTrainer()
    
    logger.info("Loading PSC data from DataPreparationUtils...")
    psc_data = trainer.data_utils.load_psc_data()
    
    if not psc_data:
        logger.error("Failed to load PSC data from DataPreparationUtils. Cannot proceed with PSC classifier training.")
        return None, None
    
    logger.info("Preparing training data for PSC classifier from PSC definitions...")
    try:
        df = trainer.prepare_data(psc_data)
        
        if df.empty:
            logger.error("Prepared DataFrame for PSC classifier training is empty. Check PSC data and augmentation logic.")
            return None, None
        
        if len(df['psc'].unique()) < 2:
            logger.error(f"Only {len(df['psc'].unique())} unique PSC classes found. Need at least 2 classes for classification training.")
            return None, None

        if len(df) < 10:
            logger.warning(f"Very few training samples ({len(df)}) available. Training might not be effective.")
        
    except Exception as e:
        logger.error(f"Error preparing data for PSC classifier: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    logger.info("Starting PSC classifier training...")
    try:
        learner = trainer.train_model(df, epochs=5, lr=1e-3)
    except Exception as e:
        logger.error(f"PSC classifier training failed during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    logger.info("Evaluating PSC classifier model on validation set...")
    trainer.evaluate_model(learner)
    
    logger.info("Saving PSC classifier model and mappings...")
    trainer.save_model(learner, "models/psc_classifier")
    
    logger.info("\nTesting PSC predictions with sample descriptions:")
    test_cases = [
        "Professional office chair with ergonomic design",
        "Computer hardware and software installation",
        "Office supplies and stationery for administrative tasks",
        "Consulting services for business analysis and strategy",
        "Purchase of new server equipment",
        "IT support and maintenance",
        "Rental of office space",
        "Legal advisory services for contracts",
        "Medical supplies for first aid",
        "Waste disposal services",
        "Building maintenance and repair"
    ]
    
    for test_text in test_cases:
        result = trainer.predict_psc(learner, test_text, psc_data)
        logger.info(f"Text: {test_text}")
        logger.info(f"Prediction: {result['predicted_psc']} (Confidence: {result['confidence']:.3f})")
        logger.info(f"  Description: {result['shortName']}")
        logger.info(f"  Category: {result['spendCategoryTitle']}")
        logger.info(f"  Portfolio: {result['portfolioGroup']}")
        logger.info("-" * 50)
    
    return learner, trainer


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True, parents=True)
    
    try:
        trained_learner, trainer = train_psc_classifier()
        if trained_learner and trainer:
            logger.info("PSC classifier training completed successfully!")
        else:
            logger.error("PSC classifier training process did not complete successfully.")
    except Exception as e:
        logger.critical(f"PSC training script failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()

