import os
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
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism to prevent multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PSCDataset(Dataset):
    """Custom Dataset for PSC text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PSCClassifierTrainer:
    """Enhanced PSC classifier using pure HuggingFace Transformers."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.psc_to_idx = {}
        self.idx_to_psc = {}
        
        # Initialize tokenizer
        self._initialize_tokenizer()
        
        # Import data utils
        try:
            from data_preparation_utils import DataPreparationUtils
            self.data_utils = DataPreparationUtils()
        except ImportError:
            logger.warning("DataPreparationUtils not found. Please ensure it's available.")
            self.data_utils = None
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer with proper configuration."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                add_prefix_space=False
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Tokenizer initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise
    
    def prepare_psc_training_data(
        self, 
        psc_data: Dict, 
        additional_descriptions: Optional[List[Dict[str, str]]] = None, 
        augment_data: bool = True
    ) -> pd.DataFrame:
        """Prepare comprehensive training data from PSC mapping."""
        training_samples = []
        
        if not psc_data or 'psc_mapping' not in psc_data:
            raise ValueError("Invalid PSC data provided: 'psc_mapping' key is missing or empty.")
        
        psc_mapping = psc_data['psc_mapping']
        
        for psc_code, psc_info in psc_mapping.items():
            # Convert all values to strings
            psc_code_str = str(psc_info.get('psc', ''))
            short_name_str = str(psc_info.get('shortName', ''))
            spend_category_str = str(psc_info.get('spendCategoryTitle', ''))
            portfolio_group_str = str(psc_info.get('portfolioGroup', ''))
            long_name_str = str(psc_info.get('longName', ''))
            includes_str = str(psc_info.get('includes', ''))
            excludes_str = str(psc_info.get('excludes', ''))
            
            # Handle lists
            example_phrases_list = [str(p) for p in psc_info.get('examplePhrases', []) if p is not None]
            synonyms_list = [str(s) for s in psc_info.get('synonyms', []) if s is not None]
            naics_list = [str(n) for n in psc_info.get('NAICS', []) if n is not None]
            
            # Create base description
            base_description_parts = []
            if short_name_str and short_name_str != 'None':
                base_description_parts.append(short_name_str)
            if spend_category_str and spend_category_str != 'None' and spend_category_str != short_name_str:
                base_description_parts.append(spend_category_str)
            if portfolio_group_str and portfolio_group_str != 'None' and portfolio_group_str not in base_description_parts:
                base_description_parts.append(portfolio_group_str)
            if long_name_str and long_name_str != 'None' and long_name_str not in base_description_parts:
                base_description_parts.append(long_name_str)
            if includes_str and includes_str != 'None':
                base_description_parts.append(f"Includes: {includes_str}")
            
            # Add synonyms and example phrases
            if synonyms_list:
                base_description_parts.extend(synonyms_list)
            if example_phrases_list:
                base_description_parts.extend(example_phrases_list)
            
            # Create base training sample
            unique_base_parts = list(dict.fromkeys([p.strip() for p in base_description_parts if p.strip() and p.strip() != 'None']))
            base_text = ". ".join(unique_base_parts)
            
            if base_text:
                training_samples.append({
                    'text': base_text,
                    'psc': psc_code_str,
                    'source': 'combined_base'
                })
            
            # Generate augmented samples
            if augment_data:
                augmented_samples = self._generate_augmented_samples(
                    psc_code=psc_code_str,
                    short_name=short_name_str,
                    category=spend_category_str,
                    portfolio=portfolio_group_str,
                    long_name=long_name_str,
                    synonyms=synonyms_list,
                    example_phrases=example_phrases_list,
                    includes=includes_str
                )
                training_samples.extend(augmented_samples)
        
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
        
        df = pd.DataFrame(training_samples)
        
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
        example_phrases: List[str],
        includes: str = ""
    ) -> List[Dict]:
        """Generate augmented training samples with enhanced business context."""
        samples = []
        
        # Enhanced business templates with more variety
        business_templates = [
            "Purchase of {item}", "Procurement of {item}", "Supply of {item}",
            "Acquisition of {item}", "{item} services", "{item} and related services",
            "Professional {item}", "{item} equipment", "{item} supplies",
            "Consulting on {item}", "Maintenance for {item}", "Repair of {item}",
            "Installation of {item}", "Lease of {item}", "Contract for {item}",
            "Development of {item}", "Support for {item}", "Software for {item}",
            "Hardware for {item}", "Advisory on {item}", "Training on {item}",
            "Management of {item}", "Operation of {item}", "Upgrade of {item}",
            "Rental of {item}", "Outsourcing of {item}", "Implementation of {item}",
            "{item} solutions", "Custom {item}", "Standard {item}",
            "{item} integration", "{item} deployment", "{item} configuration"
        ]
        
        # Context-specific templates
        context_templates = [
            "Government {item} requirements", "Federal {item} contract",
            "Department needs {item}", "Agency procurement of {item}",
            "Office {item} purchase", "Facility {item} services",
            "Administrative {item}", "Operational {item}"
        ]
        
        base_terms = set()
        for term in [short_name, category, portfolio, long_name]:
            if term and term != 'None':
                cleaned_term = self._clean_text_for_augmentation(term)
                if cleaned_term:
                    base_terms.add(cleaned_term)
        
        # Add includes information
        if includes and includes != 'None':
            includes_terms = self._extract_terms_from_includes(includes)
            base_terms.update(includes_terms)
        
        for term_list in [synonyms, example_phrases]:
            for term in term_list:
                if term and term != 'None':
                    cleaned_term = self._clean_text_for_augmentation(term)
                    if cleaned_term:
                        base_terms.add(cleaned_term)
        
        # Generate augmented samples
        for term in base_terms:
            if len(term) < 3:  # Skip very short terms
                continue
                
            lower_term = term.lower()
            
            # Use business templates
            for template in business_templates[:15]:  # Limit templates
                augmented_text = template.format(item=lower_term).strip()
                if augmented_text:
                    samples.append({
                        'text': augmented_text,
                        'psc': psc_code,
                        'source': 'augmented_business'
                    })
            
            # Use context templates (fewer to avoid over-augmentation)
            for template in context_templates[:5]:
                augmented_text = template.format(item=lower_term).strip()
                if augmented_text:
                    samples.append({
                        'text': augmented_text,
                        'psc': psc_code,
                        'source': 'augmented_context'
                    })
        
        return samples
    
    def _extract_terms_from_includes(self, includes_text: str) -> List[str]:
        """Extract meaningful terms from includes field."""
        if not includes_text or includes_text == 'None':
            return []
        
        # Split by common separators
        terms = re.split(r'[;,.]', includes_text)
        cleaned_terms = []
        
        for term in terms:
            cleaned = self._clean_text_for_augmentation(term)
            if cleaned and len(cleaned) >= 3:
                cleaned_terms.append(cleaned)
        
        return cleaned_terms
    
    def _clean_text_for_augmentation(self, text: Any) -> str:
        """Clean text for augmentation."""
        if text is None or text == 'None':
            return ""
        
        text_str = str(text).strip()
        # Remove extra punctuation but keep essential characters
        cleaned = re.sub(r'[^\w\s\-&]', '', text_str)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _clean_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate training data with enhanced filtering."""
        # Remove empty texts
        df = df[df['text'].str.strip() != '']
        df = df[df['text'].str.len() >= 5]  # Increased minimum length
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text', 'psc'])
        
        # Filter PSCs with sufficient samples
        psc_counts = df['psc'].value_counts()
        min_samples_per_class = 10  # Increased minimum for better training
        valid_pscs = psc_counts[psc_counts >= min_samples_per_class].index
        df = df[df['psc'].isin(valid_pscs)]
        
        # Remove extremely long texts that might cause memory issues
        df = df[df['text'].str.len() <= 1000]
        
        if df.empty:
            logger.warning("No valid training data remaining after cleaning.")
            return pd.DataFrame(columns=['text', 'psc', 'source'])
        
        logger.info(f"Cleaned data: {len(df)} samples for {df['psc'].nunique()} PSC classes.")
        return df.reset_index(drop=True)
    
    def prepare_data(
        self, 
        psc_data: Dict, 
        additional_descriptions: Optional[List[Dict[str, str]]] = None
    ) -> pd.DataFrame:
        """Main data preparation method."""
        df = self.prepare_psc_training_data(psc_data, additional_descriptions)
        
        if df.empty:
            logger.error("DataFrame is empty after data preparation.")
            return df
        
        # Encode labels
        unique_pscs = sorted(df['psc'].unique())
        self.label_encoder.fit(unique_pscs)
        
        self.psc_to_idx = {psc: idx for idx, psc in enumerate(unique_pscs)}
        self.idx_to_psc = {idx: psc for psc, idx in self.psc_to_idx.items()}
        
        df['label'] = self.label_encoder.transform(df['psc'])
        
        logger.info(f"Prepared {len(df)} training samples for {len(unique_pscs)} PSC classes.")
        
        # Show distribution
        psc_distribution = df['psc'].value_counts()
        logger.info(f"Sample distribution (top 10):\n{psc_distribution.head(10)}")
        
        return df
    
    def create_model(self, num_labels: int):
        """Create and configure the model with optimal settings."""
        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                finetuning_task="text-classification",
                id2label=self.idx_to_psc,
                label2id=self.psc_to_idx,
                hidden_dropout_prob=0.3,  # Increased dropout for regularization
                attention_probs_dropout_prob=0.3
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config
            )
            
            logger.info(f"Model created with {num_labels} labels.")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro
        }
    
    def _get_training_args_kwargs(self, train_dataset_len: int, epochs: int) -> dict:
        """Get training arguments with version compatibility."""
        # Try to determine which parameter name to use
        try:
            # Test if eval_strategy is supported (newer versions)
            test_args = TrainingArguments(output_dir='./test', eval_strategy="no")
            eval_param = "eval_strategy"
        except TypeError:
            # Fall back to evaluation_strategy (older versions)
            eval_param = "evaluation_strategy"
        
        kwargs = {
            'output_dir': './results',
            'num_train_epochs': epochs,
            'per_device_train_batch_size': 8,  # Reduced for stability
            'per_device_eval_batch_size': 16,
            'gradient_accumulation_steps': 2,  # Added for effective larger batch size
            'warmup_steps': min(500, train_dataset_len // 10),  # Dynamic warmup
            'weight_decay': 0.01,
            'learning_rate': 2e-5,  # Optimal learning rate for DistilBERT
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'logging_dir': './logs',
            'logging_steps': 100,
            eval_param: "steps",  # Use the correct parameter name
            'eval_steps': min(500, train_dataset_len // 4),  # Dynamic eval steps
            'save_strategy': "steps",
            'save_steps': min(500, train_dataset_len // 4),
            'load_best_model_at_end': True,
            'metric_for_best_model': "f1_weighted",  # Changed to weighted F1
            'greater_is_better': True,
            'report_to': None,  # Disable wandb
            'dataloader_num_workers': 0,  # Disable multiprocessing for stability
            'fp16': torch.cuda.is_available(),  # Use mixed precision if CUDA available
            'seed': 42,
            'data_seed': 42,
            'remove_unused_columns': False
        }
        
        return kwargs
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, epochs: int = 3) -> Trainer:
        """Train the PSC classifier model with optimized parameters."""
        
        # Split data with stratification
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=test_size,
            random_state=42,
            stratify=df['label']
        )
        
        # Create datasets
        train_dataset = PSCDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = PSCDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Create model
        num_labels = df['label'].nunique()
        self.create_model(num_labels)
        
        # Get training arguments with version compatibility
        training_args_kwargs = self._get_training_args_kwargs(len(train_dataset), epochs)
        training_args = TrainingArguments(**training_args_kwargs)
        
        # Create trainer with enhanced configuration
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("Starting model training...")
        trainer.train()
        
        logger.info("Training completed!")
        return trainer
    
    def evaluate_model(self, trainer: Trainer, df: pd.DataFrame = None):
        """Evaluate model performance with detailed metrics."""
        logger.info("Evaluating model performance...")
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        
        logger.info("Validation Results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Additional evaluation if test data provided
        if df is not None:
            logger.info("\nTesting on sample predictions...")
            test_cases = df.sample(min(10, len(df)))['text'].tolist()
            
            for text in test_cases:
                result = self.predict_psc(text)
                logger.info(f"Text: {text[:100]}...")
                logger.info(f"Prediction: {result['predicted_psc']} (Confidence: {result['confidence']:.3f})")
                logger.info("-" * 50)
    
    def predict_psc(self, text: str, top_k: int = 1) -> Dict:
        """Predict PSC for given text with enhanced prediction logic."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                if top_k == 1:
                    predicted_class_id = predictions.argmax().item()
                    confidence = predictions.max().item()
                    predicted_psc = self.idx_to_psc[predicted_class_id]
                    
                    return {
                        "predicted_psc": predicted_psc,
                        "confidence": confidence,
                        "text": text
                    }
                else:
                    # Return top-k predictions
                    top_predictions = torch.topk(predictions, top_k)
                    results = []
                    
                    for i in range(top_k):
                        class_id = top_predictions.indices[0][i].item()
                        conf = top_predictions.values[0][i].item()
                        psc = self.idx_to_psc[class_id]
                        results.append({
                            "predicted_psc": psc,
                            "confidence": conf
                        })
                    
                    return {
                        "predictions": results,
                        "text": text
                    }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "error": str(e),
                "predicted_psc": "UNKNOWN",
                "confidence": 0.0,
                "text": text
            }
    
    def save_model(self, trainer: Trainer, path: str = "models/psc_classifier"):
        """Save the trained model with enhanced metadata."""
        model_path = Path(path)
        model_path.mkdir(exist_ok=True, parents=True)
        
        try:
            # Save model and tokenizer
            trainer.save_model(str(model_path))
            self.tokenizer.save_pretrained(str(model_path))
            
            # Save comprehensive mappings and metadata
            mappings = {
                "psc_to_idx": self.psc_to_idx,
                "idx_to_psc": self.idx_to_psc,
                "model_name": self.model_name,
                "max_length": self.max_length,
                "num_labels": len(self.psc_to_idx),
                "training_timestamp": pd.Timestamp.now().isoformat(),
                "model_version": "1.0"
            }
            
            with open(model_path / "label_mappings.json", "w", encoding="utf-8") as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    @classmethod
    def load_model(cls, path: str):
        """Load a trained model with enhanced error handling."""
        model_path = Path(path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        
        # Load mappings
        mappings_file = model_path / "label_mappings.json"
        if not mappings_file.exists():
            raise FileNotFoundError(f"Label mappings file not found at {mappings_file}")
        
        with open(mappings_file, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        
        # Initialize trainer
        trainer = cls(
            model_name=mappings["model_name"],
            max_length=mappings["max_length"]
        )
        
        # Load model and tokenizer
        trainer.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        trainer.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Set mappings
        trainer.psc_to_idx = mappings["psc_to_idx"]
        trainer.idx_to_psc = {int(k): v for k, v in mappings["idx_to_psc"].items()}
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model supports {mappings.get('num_labels', 'unknown')} PSC classes")
        return trainer


def train_psc_classifier():
    """Main training function with enhanced error handling and logging."""
    try:
        # Initialize trainer
        trainer = PSCClassifierTrainer(model_name="distilbert-base-uncased")
        
        # Load PSC data
        logger.info("Loading PSC data...")
        if trainer.data_utils is None:
            logger.error("DataPreparationUtils not available. Cannot load PSC data.")
            return None, None
        
        psc_data = trainer.data_utils.load_psc_data()
        
        if not psc_data:
            logger.error("Failed to load PSC data.")
            return None, None
        
        logger.info(f"Loaded {len(psc_data.get('psc_mapping', {}))} PSC codes across {len(set([v.get('spendCategoryParent', '') for v in psc_data.get('psc_mapping', {}).values()]))} categories")
        
        # Prepare data
        logger.info("Preparing training data...")
        df = trainer.prepare_data(psc_data)
        
        if df.empty:
            logger.error("No training data available.")
            return None, None
        
        # Train model
        logger.info("Training model...")
        trained_model = trainer.train_model(df, epochs=3)
        
        # Evaluate model
        trainer.evaluate_model(trained_model, df)
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model(trained_model)
        
        # Test predictions with enhanced examples
        logger.info("\nTesting predictions...")
        test_cases = [
            "Professional office chair with ergonomic design",
            "Computer hardware and software installation services",
            "Office supplies and stationery for administrative tasks",
            "Consulting services for business analysis and strategy development",
            "Purchase of new server equipment for data center",
            "GUNS, THROUGH 30MM machine guns and pistol brushes",
            "Maintenance and repair services for office equipment",
            "Training and educational services for staff development"
        ]
        
        for test_text in test_cases:
            result = trainer.predict_psc(test_text)
            logger.info(f"Text: {test_text}")
            logger.info(f"Prediction: {result['predicted_psc']} (Confidence: {result['confidence']:.3f})")
            logger.info("-" * 50)
        
        return trained_model, trainer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Create models directory
    Path("models").mkdir(exist_ok=True, parents=True)
    
    # Train classifier
    logger.info("Starting PSC classifier training...")
    model, trainer = train_psc_classifier()
    
    if model and trainer:
        logger.info("PSC classifier training completed successfully!")
        logger.info(f"Model saved with {len(trainer.psc_to_idx)} PSC classes")
    else:
        logger.error("PSC classifier training failed.")