# Enhanced IDP Solution - Project Structure & Setup Guide

## 📁 Project Directory Structure

```
enhanced_idp_solution/
├── 📄 main.py                          # Main entry point
├── 📄 data_preparation_utils.py        # Data utilities
├── 📄 layoutlmv3_trainer.py           # LayoutLMv2 training
├── 📄 psc_classifier_trainer.py       # PSC classification training
├── 📄 idp_inference_engine.py         # Complete inference pipeline
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
│
├── 📁 data/                           # Dataset storage
│   ├── 📁 funsd/                     # FUNSD dataset
│   ├── 📁 sroie/                     # SROIE dataset
│   ├── 📁 cord/                      # CORD dataset
│   ├── 📁 kaggle_invoices/           # Kaggle invoice dataset
│   ├── 📁 synthetic/                 # Generated synthetic data
│   └── 📄 pscs.json                  # PSC classification data
│
├── 📁 models/                        # Trained model storage
│   ├── 📁 layoutlmv2_invoice_extractor/
│   │   ├── 📄 config.json
│   │   ├── 📄 pytorch_model.bin
│   │   └── 📄 tokenizer files...
│   └── 📁 psc_classifier/
│       ├── 📄 config.json
│       ├── 📄 pytorch_model.bin
│       ├── 📄 label_mappings.json
│       └── 📄 tokenizer files...
│
├── 📁 output/                        # Processing results
│   ├── 📄 processed_document_1.json
│   ├── 📄 batch_results.json
│   └── 📄 processing_logs.txt
│
├── 📁 samples/                       # Sample invoice images
│   ├── 📄 sample_invoice_1.jpg
│   ├── 📄 sample_invoice_2.png
│   └── 📄 test_receipts/
│
└── 📁 notebooks/                     # Jupyter notebooks (optional)
    ├── 📄 data_exploration.ipynb
    ├── 📄 model_evaluation.ipynb
    └── 📄 synthetic_data_generation.ipynb
```

## 🚀 Quick Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv idp_env
source idp_env/bin/activate  # On Windows: idp_env\Scripts\activate

# Install dependencies
pip install fastai torch transformers
pip install opencv-python pillow easyocr
pip install pandas numpy scikit-learn
pip install requests pathlib
```

### 2. Initialize Project Structure

```bash
# Create directories
mkdir -p data/{funsd,sroie,cord,kaggle_invoices,synthetic}
mkdir -p models/{layoutlmv2_invoice_extractor,psc_classifier}
mkdir -p output samples notebooks

# Download PSC data
curl -o data/pscs.json https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/pscs.json
```

### 3. Training Pipeline

```bash
# Train models (in order)
python psc_classifier_trainer.py      # Train PSC classifier first
python layoutlmv3_trainer.py         # Train LayoutLMv2 model

# Verify training
python main.py --check-models
```

### 4. Usage Examples

```bash
# Interactive mode
python main.py

# Process single document
python -c "
from idp_inference_engine import create_inference_engine
engine = create_inference_engine()
result = engine.process_document('path/to/invoice.jpg')
print(result)
"
```

## 📊 Data Acquisition Guide

### Public Datasets Download Links:

**FUNSD (Form Understanding in Noisy Scanned Documents)**

```bash
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip -d data/funsd/
```

**SROIE (Scanned Receipts OCR and Information Extraction)**

```bash
# Download from: https://rrc.cvc.uab.es/?ch=13&com=downloads
# Requires registration
```

**CORD (Consolidated Receipt Dataset)**

```bash
git clone https://github.com/clovaai/cord.git data/cord/
```

**Kaggle High-Quality Invoice Images**

```bash
kaggle datasets download -d valakhorasani/high-quality-invoice-images-for-ocr
unzip high-quality-invoice-images-for-ocr.zip -d data/kaggle_invoices/
```

### Synthetic Data Generation Strategy:

**Template-Based Generation:**

- Create invoice templates with variable fields
- Use Faker library for realistic data generation
- Integrate PSC codes for balanced classification
- Generate variations in layout, fonts, and styling

**Implementation:**

```python
from faker import Faker
import random
from PIL import Image, ImageDraw, ImageFont

def generate_synthetic_invoice(psc_mapping):
    fake = Faker()

    # Select random PSC items
    psc_items = random.sample(list(psc_mapping.keys()), k=random.randint(1, 5))

    invoice_data = {
        'invoice_number': f"INV-{fake.random_int(10000, 99999)}",
        'date': fake.date_this_year().strftime('%Y-%m-%d'),
        'vendor_name': fake.company(),
        'line_items': []
    }

    for psc in psc_items:
        item = {
            'description': psc_mapping[psc]['shortName'],
            'quantity': random.randint(1, 10),
            'unit_price': round(random.uniform(10, 1000), 2),
            'psc': psc
        }
        invoice_data['line_items'].append(item)

    return invoice_data
```

## 🔧 Integration & Expansion Guide

### Backend Integration:

```python
# Flask/FastAPI integration example
from flask import Flask, request, jsonify
from idp_inference_engine import create_inference_engine

app = Flask(__name__)
engine = create_inference_engine()

@app.route('/process-invoice', methods=['POST'])
def process_invoice_api():
    file = request.files['invoice']
    file.save('temp_invoice.jpg')

    result = engine.process_document('temp_invoice.jpg')
    return jsonify(result)
```

### Production Deployment:

- **Docker containerization** for consistent deployment
- **GPU acceleration** for faster inference
- **Model versioning** with MLflow or similar
- **Monitoring & logging** for production systems
- **Horizontal scaling** with load balancers

### Performance Optimization:

- **Model quantization** for reduced memory usage
- **Batch processing** for multiple documents
- **Caching mechanisms** for repeated PSC lookups
- **Asynchronous processing** for large volumes

### Extension Points:

- **Multi-language support** with additional OCR models
- **Document type detection** for automatic workflow routing
- **Confidence scoring** for quality assessment
- **Human-in-the-loop** validation for low-confidence results
- **Export formats** (Excel, CSV, XML) beyond JSON

## 🎯 Usage Patterns

### Development Workflow:

1. **Data Collection** → Gather diverse invoice samples
2. **Model Training** → Fine-tune on domain-specific data
3. **Validation** → Test on held-out validation set
4. **Integration** → Connect to existing business systems
5. **Monitoring** → Track performance in production

### Production Deployment:

1. **Model Serving** → Deploy via REST API or gRPC
2. **Queue Processing** → Handle high-volume batch jobs
3. **Results Storage** → Store processed data in database
4. **Audit Trail** → Maintain processing history
5. **Continuous Learning** → Retrain with new data
