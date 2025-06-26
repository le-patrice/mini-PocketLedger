# Enhanced IDP Solution - Updated Project Guide Sections

## Project Tree Structure

```
enhanced_idp_solution/
├── data/
│   ├── raw/
│   │   ├── funsd/
│   │   ├── sroie/
│   │   └── kaggle_invoices/
│   ├── processed/
│   │   ├── train_samples.json
│   │   ├── val_samples.json
│   │   └── psc_data.json
│   └── synthetic/
│       ├── images/
│       ├── annotations/
│       └── dataset_manifest.json
├── src/
│   ├── data_preparation_utils.py          # Core dataset loading & PSC integration
│   ├── synthetic_data_generator.py        # PSC-enhanced synthetic data generation
│   ├── models/
│   │   ├── layoutlm_classifier.py
│   │   └── psc_classifier.py
│   └── training/
│       ├── train_layoutlm.py
│       └── train_psc_classifier.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── outputs/
│   ├── models/
│   └── predictions/
└── requirements.txt
```

## Data Acquisition & Preparation Guide

### Public Datasets

#### 1. CORD Dataset (via Hugging Face)

```python
from data_preparation_utils import DataPreparationUtils

# Initialize data preparation utilities
data_prep = DataPreparationUtils()

# Load CORD dataset directly from Hugging Face
cord_train = data_prep.load_cord_dataset("train")
cord_val = data_prep.load_cord_dataset("validation")
```

#### 2. FUNSD+ Dataset

**Download Instructions:**

- Visit: https://guillaumejaume.github.io/FUNSD/
- Download the complete FUNSD dataset
- Extract to `data/raw/funsd/`

```python
# Load FUNSD with original annotations
funsd_samples = data_prep.load_funsd_dataset("data/raw/funsd")
```

#### 3. SROIE Dataset

**Download Instructions:**

- Visit: https://rrc.cvc.uab.es/?ch=13&com=downloads
- Register and download SROIE Task 2 dataset
- Extract to `data/raw/sroie/`

```python
# Load SROIE with bounding box annotations
sroie_samples = data_prep.load_sroie_dataset("data/raw/sroie")
```

#### 4. Kaggle High-Quality Invoice Dataset

**Download Instructions:**

- Visit: https://www.kaggle.com/datasets/afoley4/high-quality-invoice-images-for-ocr
- Download and extract to `data/raw/kaggle_invoices/`

### PSC Data Integration

The system automatically loads PSC (Product and Service Code) data from the official GitHub repository:

```python
# Load PSC classification data
psc_data = data_prep.load_psc_data()

# PSC data structure includes:
# - psc_mapping: Direct PSC code to details mapping
# - category_mapping: Spend categories to PSC codes
# - portfolio_mapping: Portfolio groups to PSC codes
```

### Unified Dataset Preparation

```python
# Combine all datasets with unified format
all_samples = cord_samples + funsd_samples + sroie_samples
unified_samples = data_prep.unify_dataset_format(all_samples)

# Create training/validation split
train_samples, val_samples = data_prep.create_training_split(unified_samples, train_ratio=0.8)

# Save processed data
data_prep.save_processed_data(train_samples, "train_samples.json")
data_prep.save_processed_data(val_samples, "val_samples.json")
```

## Synthetic Data Generation Guide

### Basic Synthetic Data Generation

```python
from synthetic_data_generator import SyntheticInvoiceGenerator

# Initialize generator with PSC integration
generator = SyntheticInvoiceGenerator(output_dir="data/synthetic")

# Generate batch of synthetic invoices
generator.generate_batch(
    num_samples=100,
    layout_types=["standard", "compact", "receipt"]
)

# Create dataset manifest for tracking
manifest = generator.generate_dataset_manifest()
```

### PSC-Enhanced Features

The synthetic data generator provides:

1. **Intelligent PSC Integration:**

   - Automatic PSC code assignment to line items
   - Document-level PSC classification based on spending patterns
   - Realistic item descriptions generated from PSC shortNames

2. **Diverse Invoice Layouts:**

   - Standard business invoices (612x792)
   - Compact formats (595x842)
   - Receipt-style layouts (400x600)

3. **Comprehensive Annotations:**
   - LayoutLMv2-compatible bounding boxes
   - Named Entity Recognition labels
   - PSC metadata for each line item
   - Document-level spending categorization

### Advanced PSC Analysis

```python
# Generate targeted data for specific PSC categories
from collections import defaultdict

# Filter by portfolio group
generator.psc_data["portfolio_mapping"]["Information Technology"]

# Generate category-specific datasets
it_focused_data = generator.generate_invoice_data()
# Modify generator to focus on specific PSC categories as needed
```

## Integration & Expansion Guide

### Training Data Pipeline

1. **Combined Dataset Creation:**

```python
# Merge real and synthetic data
real_data = data_prep.load_processed_data("train_samples.json")
synthetic_manifest = json.load(open("data/synthetic/dataset_manifest.json"))

# Combine for enhanced training
combined_dataset = real_data + synthetic_manifest["samples"]
```

2. **PSC Classification Training:**

```python
# Prepare PSC training data
psc_training_data = data_prep.prepare_psc_training_data(
    psc_data["psc_mapping"],
    descriptions_from_invoices
)
```

### Next Steps for Model Training

1. **LayoutLMv2 Fine-tuning:**

   - Use unified dataset format for information extraction
   - Incorporate PSC classification as auxiliary task
   - Leverage synthetic data for data augmentation

2. **PSC Classification Model:**

   - Train dedicated classifier for PSC assignment
   - Use real + synthetic descriptions for robust training
   - Implement document-level PSC prediction

3. **Multi-task Learning:**
   - Combine information extraction with PSC classification
   - Use document-level PSC as additional supervision signal
   - Implement end-to-end trainable pipeline

### Quality Assurance

- **Synthetic Data Validation:** Each generated sample includes comprehensive PSC metadata for verification
- **Dataset Balance:** Monitor PSC category distribution through dataset manifest
- **Annotation Quality:** Precise bounding box calculations with field-type labeling

This enhanced pipeline provides a complete foundation for training robust IDP models with integrated PSC classification capabilities, suitable for production deployment.
