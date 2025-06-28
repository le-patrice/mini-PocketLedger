import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import requests
import json
import re
from pathlib import Path
import logging
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic UNSPSC training data from the four hierarchical fields."""
    
    def __init__(self):
        self.templates = [
            "{commodity_name}",
            "{commodity_name} for {family_name}",
            "{class_name} - {commodity_name}",
            "{segment_name} {family_name} {commodity_name}",
            "{commodity_name} in {class_name}",
            "{family_name} {commodity_name}",
            "Professional {commodity_name}",
            "Industrial {commodity_name}",
            "{commodity_name} and related {class_name}",
            "{segment_name} category {commodity_name}"
        ]
        
    def generate_synthetic_text(self, row):
        """Generate synthetic text from UNSPSC hierarchy."""
        template = random.choice(self.templates)
        
        # Clean and prepare field values
        segment = self._clean_field(row.get('Segment Name', ''))
        family = self._clean_field(row.get('Family Name', ''))
        class_name = self._clean_field(row.get('Class Name', ''))
        commodity = self._clean_field(row.get('Commodity Name', ''))
        
        # Generate text using template
        synthetic_text = template.format(
            segment_name=segment.lower(),
            family_name=family.lower(),
            class_name=class_name.lower(),
            commodity_name=commodity.lower()
        )
        
        # Add variations
        variations = [
            synthetic_text,
            f"supply of {synthetic_text}",
            f"purchase {synthetic_text}",
            f"procurement {synthetic_text}",
            synthetic_text.replace(' and ', ' & '),
        ]
        
        return random.choice(variations)
    
    def _clean_field(self, field):
        """Clean field values."""
        if pd.isna(field) or field == '':
            return 'general'
        return re.sub(r'[^\w\s-]', '', str(field)).strip()

class UNSPSCDataset(Dataset):
    """Optimized PyTorch Dataset for UNSPSC classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

class OptimizedUNSPSCClassifier:
    """Efficient UNSPSC Classifier with synthetic data generation."""
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_mappings = {}
        self.synthetic_generator = SyntheticDataGenerator()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_unspsc_data(self, url="https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/unspsc-codes.csv"):
        """Load and prepare UNSPSC data efficiently."""
        logger.info("Loading UNSPSC dataset...")
        
        try:
            df = pd.read_csv(url)
            logger.info(f"Loaded {len(df)} UNSPSC records")
            
            # Required columns for UNSPSC hierarchy
            required_cols = ['Segment Name', 'Family Name', 'Class Name', 'Commodity Name']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                return None
            
            # Clean data
            df = df.dropna(subset=required_cols).reset_index(drop=True)
            
            # Use Commodity Name as target for classification
            df['target'] = df['Commodity Name']
            
            # Filter classes with minimum samples
            class_counts = df['target'].value_counts()
            valid_classes = class_counts[class_counts >= 2].index
            df = df[df['target'].isin(valid_classes)].reset_index(drop=True)
            
            logger.info(f"Processed {len(df)} samples with {df['target'].nunique()} classes")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def generate_training_data(self, df, samples_per_class=10):
        """Generate synthetic training data."""
        logger.info("Generating synthetic training data...")
        
        synthetic_data = []
        
        for target_class in df['target'].unique():
            class_df = df[df['target'] == target_class]
            
            # Generate multiple synthetic samples per class
            for _ in range(min(samples_per_class, len(class_df) * 3)):
                row = class_df.sample(1).iloc[0]
                synthetic_text = self.synthetic_generator.generate_synthetic_text(row)
                synthetic_data.append({
                    'text': synthetic_text,
                    'target': target_class,
                    'segment': row['Segment Name'],
                    'family': row['Family Name'],
                    'class': row['Class Name'],
                    'commodity': row['Commodity Name']
                })
        
        synthetic_df = pd.DataFrame(synthetic_data)
        logger.info(f"Generated {len(synthetic_df)} synthetic samples")
        return synthetic_df
    
    def prepare_training_data(self, df):
        """Prepare data for training with label encoding."""
        # Create label mappings
        unique_labels = sorted(df['target'].unique())
        self.label_mappings = {
            'label_to_idx': {label: idx for idx, label in enumerate(unique_labels)},
            'idx_to_label': {idx: label for idx, label in enumerate(unique_labels)}
        }
        
        # Encode labels
        df['encoded_label'] = df['target'].map(self.label_mappings['label_to_idx'])
        
        return df
    
    def create_model(self, num_labels):
        """Create optimized classification model."""
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        )
        
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }
    
    def train(self, df, test_size=0.2, epochs=3):
        """Train the classifier efficiently."""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["text"].tolist(),
            df["encoded_label"].tolist(),
            test_size=test_size,
            random_state=42,
            stratify=df["encoded_label"],
        )
        
        logger.info(f"Training: {len(train_texts)}, Validation: {len(val_texts)}")
        
        # Create datasets
        train_dataset = UNSPSCDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = UNSPSCDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Create model
        num_labels = len(self.label_mappings['label_to_idx'])
        self.create_model(num_labels)
        
        # Training arguments - optimized for efficiency
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=1,
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=3e-5,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            logging_steps=100,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
            seed=42,
            report_to="none",
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
    
    def predict(self, text):
        """Predict UNSPSC category for input text."""
        if self.model is None:
            return {"error": "Model not trained"}
        
        # Clean and tokenize input
        clean_text = re.sub(r'[^\w\s-]', '', str(text).lower()).strip()
        
        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class_id = predictions.argmax().item()
        confidence = predictions.max().item()
        predicted_class = self.label_mappings['idx_to_label'].get(predicted_class_id, "UNKNOWN")
        
        return {
            "predicted_category": predicted_class,
            "confidence": float(confidence),
            "predicted_class_id": predicted_class_id
        }
    
    def save_model(self, trainer, path="models/unspsc_classifier"):
        """Save trained model and metadata."""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        
        # Save metadata
        metadata = {
            "label_mappings": self.label_mappings,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_labels": len(self.label_mappings['label_to_idx']),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")

def main():
    """Main execution function."""
    # Initialize classifier
    classifier = OptimizedUNSPSCClassifier()
    
    # Load UNSPSC data
    df = classifier.load_unspsc_data()
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Generate synthetic training data
    synthetic_df = classifier.generate_training_data(df, samples_per_class=15)
    
    # Prepare data for training
    training_df = classifier.prepare_training_data(synthetic_df)
    
    # Train model
    trainer = classifier.train(training_df, epochs=3)
    
    # Evaluate
    eval_results = trainer.evaluate()
    logger.info("Evaluation Results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save model
    classifier.save_model(trainer)
    
    # Test predictions
    test_cases = [
        "office supplies and paper products",
        "computer software and applications",
        "construction materials and equipment",
        "medical devices and equipment",
        "food and beverage products"
    ]
    
    logger.info("\nTesting predictions:")
    for text in test_cases:
        result = classifier.predict(text)
        if "error" not in result:
            logger.info(f"Text: '{text}'")
            logger.info(f"  Predicted: {result['predicted_category']}")
            logger.info(f"  Confidence: {result['confidence']:.3f}")
            logger.info("-" * 50)

if __name__ == "__main__":
    main()