#!/usr/bin/env python3
"""
Main entry point for the Enhanced IDP Solution
Demonstrates the complete invoice processing pipeline
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from core.idp_inference_engine import IDPInferenceEngine, create_inference_engine
from core.data_preparation_utils import DataPreparationUtils # Ensure this import is correct


def main():
    """Main function to demonstrate IDP pipeline usage."""
    print("🚀 Enhanced Intelligent Document Processing (IDP) Solution")
    print("=" * 60)

    # Initialize the inference engine
    print("1. Initializing IDP Inference Engine...")
    try:
        engine = create_inference_engine()
        print("✅ Inference engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        print("💡 Make sure models are trained and saved in the 'models' directory by running:")
        print("   - python core/layoutlmv3_trainer.py")
        print("   - python core/psc_classifier_trainer.py")
        return

    # Initialize data preparation utilities
    print("\n2. Initializing Data Preparation Utilities...")
    data_utils = DataPreparationUtils()
    print("✅ Data utilities initialized successfully!")

    # Load PSC data
    print("\n3. Loading PSC classification data...")
    # This psc_data contains 'psc_mapping', 'category_mapping', 'portfolio_mapping'
    psc_data = data_utils.load_psc_data()

    if psc_data and psc_data.get("psc_mapping"):
        print(
            f"✅ Loaded {len(psc_data['psc_mapping'])} PSC codes across {len(psc_data['category_mapping'])} categories"
        )
    else:
        print("⚠️  Failed to load PSC data from remote. Using limited fallback for PSC classification.")
        # The IDPInferenceEngine itself handles a basic internal PSC fallback if psc_data is empty,
        # but loading an explicit fallback here ensures consistency for data_utils.get_psc_by_description
        psc_data = {
            "psc_mapping": {
                "7510": {"psc": "7510", "shortName": "Office Supplies", "spendCategoryTitle": "General Supplies", "portfolioGroup": "Administrative"},
                "7520": {"psc": "7520", "shortName": "Computer Equipment", "spendCategoryTitle": "IT Hardware", "portfolioGroup": "Technology"},
                "7530": {"psc": "7530", "shortName": "Furniture", "spendCategoryTitle": "Office Furniture", "portfolioGroup": "Administrative"},
            },
            "category_mapping": {}, # Simplified, but ensures structure
            "portfolio_mapping": {},
            "all_pscs": [], "all_categories": [], "all_portfolios": []
        }
        # Force data_utils to use this fallback if it didn't load from URL
        data_utils.psc_data = psc_data


    # Process documents
    print("\n4. Document Processing Options:")
    print("   a) Process single document")
    print("   b) Process multiple documents (batch)")
    print("   c) Load and prepare training datasets")
    print("   d) Run a full demo with sample data")
    print("   e) Show PSC classification examples (interactive)")

    choice = input("\nSelect option (a/b/c/d/e): ").lower().strip()

    if choice == "a":
        process_single_document(engine, psc_data, data_utils) # Pass full psc_data
    elif choice == "b":
        process_multiple_documents(engine, psc_data, data_utils) # Pass full psc_data
    elif choice == "c":
        load_training_datasets(data_utils, psc_data)
    elif choice == "d":
        run_demo(engine, psc_data, data_utils) # Pass full psc_data
    elif choice == "e":
        show_psc_examples(data_utils, psc_data)
    else:
        print("Invalid choice. Running demo by default...")
        run_demo(engine, psc_data, data_utils) # Pass full psc_data


def process_single_document(
    engine: IDPInferenceEngine, psc_data: dict, data_utils: DataPreparationUtils
):
    """Process a single document."""
    print("\n📄 Single Document Processing")
    print("-" * 30)

    image_path = input("Enter path to invoice image (e.g., data/kaggle_invoices/images/batch1-0001.jpg): ").strip()

    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return

    print(f"Processing: {image_path}")
    try:
        # Pass the full psc_data dictionary to the engine
        result = engine.process_document(image_path, psc_data)

        # Display results
        print("\n✅ Processing completed!")
        print("\n📊 Extracted Information:")
        print(f"Invoice Number: {result['document_info'].get('invoice_number', 'N/A')}")
        print(f"Date: {result['document_info'].get('date', 'N/A')}")
        print(f"Due Date: {result['document_info'].get('due_date', 'N/A')}") # Added
        print(f"Vendor: {result['document_info'].get('vendor_name', 'N/A')}")
        print(f"Vendor Address: {result['document_info'].get('vendor_address', 'N/A')}") # Added
        print(f"Customer: {result['document_info'].get('customer_name', 'N/A')}") # Added
        print(f"Customer Address: {result['document_info'].get('customer_address', 'N/A')}") # Added
        print(f"Subtotal: {result['document_info'].get('subtotal', 'N/A')}") # Added
        print(f"Tax Amount: {result['document_info'].get('tax_amount', 'N/A')}") # Added
        print(f"Discount Amount: {result['document_info'].get('discount_amount', 'N/A')}") # Added
        print(f"Total Amount: {result['document_info'].get('total_amount', 'N/A')}")
        print(f"Currency: {result['document_info'].get('currency', 'N/A')}") # Added
        print(f"Line Items: {len(result['line_items'])}")

        # Show line items with PSC classification (already done by engine)
        for i, item in enumerate(result["line_items"], 1):
            print(f"\nLine Item {i}:")
            print(f"  Description: {item.get('item_description', 'N/A')}")
            print(f"  Quantity: {item.get('quantity', 'N/A')}")
            print(f"  Unit Price: {item.get('unit_price', 'N/A')}")
            print(f"  Line Total: {item.get('line_total', 'N/A')}") # Added

            # Display PSC classification result from the engine's output
            psc_class = item.get("psc_classification", {})
            print(f"  PSC: {psc_class.get('psc', 'N/A')} - {psc_class.get('shortName', 'N/A')}")
            print(f"  Category: {psc_class.get('spendCategoryTitle', 'N/A')}")
            print(f"  Portfolio: {psc_class.get('portfolioGroup', 'N/A')}")
            print(f"  Confidence: {psc_class.get('confidence', 0.0):.2%}")


        # Save results
        output_file = "processed_invoice.json"
        data_utils.save_processed_data(result, output_file)
        print(f"\n💾 Results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


def process_multiple_documents(
    engine: IDPInferenceEngine, psc_data: dict, data_utils: DataPreparationUtils
):
    """Process multiple documents in batch."""
    print("\n📄 Batch Document Processing")
    print("-" * 30)

    folder_path = input("Enter path to folder containing invoice images (e.g., data/kaggle_invoices/images/): ").strip()
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print("❌ No image files found in the specified folder.")
        return

    print(f"Found {len(image_files)} image files.")

    try:
        # Pass the full psc_data dictionary to the engine for batch processing
        results = engine.batch_process([str(f) for f in image_files], psc_data=psc_data)

        print(f"\n✅ Batch processing completed!")
        print(
            f"📊 Successfully processed: {len([r for r in results if 'error' not in r])}/{len(results)} documents."
        )

        # Save batch results using data_utils
        data_utils.save_processed_data(results, "batch_results.json")
        print("💾 All batch results aggregated and saved as 'batch_results.json' in the 'output' directory.")

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()


def load_training_datasets(data_utils: DataPreparationUtils, psc_data: dict):
    """Load and process training datasets."""
    print("\n📚 Training Dataset Loading")
    print("-" * 30)

    print("Available datasets:")
    print("   1) CORD Dataset (via Hugging Face)")
    print("   2) FUNSD Dataset (local)")
    print("   3) SROIE Dataset (local)")
    print("   4) Kaggle Invoices Dataset (local)") # Added
    print("   5) Synthetic Invoices Dataset (local)") # Added
    print("   6) Load all available datasets (combines all above)") # Updated numbering

    choice = input("\nSelect dataset(s) to load (1/2/3/4/5/6): ").strip()

    all_samples = []

    if choice in ["1", "6"]:
        print("\n📊 Loading CORD dataset...")
        try:
            cord_train = data_utils.load_cord_dataset("train")
            cord_test = data_utils.load_cord_dataset("test")
            all_samples.extend(cord_train + cord_test)
            print(
                f"✅ Loaded {len(cord_train)} training + {len(cord_test)} test samples from CORD."
            )
        except Exception as e:
            print(f"❌ Failed to load CORD dataset: {e}")

    if choice in ["2", "6"]:
        funsd_path = input(
            "Enter path to FUNSD dataset directory (e.g., data/funsd or press Enter to skip): "
        ).strip()
        if funsd_path and Path(funsd_path).exists():
            print("\n📊 Loading FUNSD dataset...")
            try:
                funsd_samples = data_utils.load_funsd_dataset(funsd_path)
                all_samples.extend(funsd_samples)
                print(f"✅ Loaded {len(funsd_samples)} samples from FUNSD.")
            except Exception as e:
                print(f"❌ Failed to load FUNSD dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping FUNSD dataset as path not provided or not found.")


    if choice in ["3", "6"]:
        sroie_path = input(
            "Enter path to SROIE dataset directory (e.g., data/sroie or press Enter to skip): "
        ).strip()
        if sroie_path and Path(sroie_path).exists():
            print("\n📊 Loading SROIE dataset...")
            try:
                sroie_samples = data_utils.load_sroie_dataset(sroie_path)
                all_samples.extend(sroie_samples)
                print(f"✅ Loaded {len(sroie_samples)} samples from SROIE.")
            except Exception as e:
                print(f"❌ Failed to load SROIE dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping SROIE dataset as path not provided or not found.")

    # Added Kaggle Invoices Dataset loading
    if choice in ["4", "6"]:
        kaggle_invoices_path = "data/kaggle_invoices" # Assuming default location
        if Path(kaggle_invoices_path).exists():
            print(f"\n📊 Loading Kaggle Invoices dataset from {kaggle_invoices_path}...")
            try:
                kaggle_samples = data_utils.load_kaggle_invoice_dataset(kaggle_invoices_path)
                all_samples.extend(kaggle_samples)
                print(f"✅ Loaded {len(kaggle_samples)} samples from Kaggle Invoices.")
            except Exception as e:
                print(f"❌ Failed to load Kaggle Invoices dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Skipping Kaggle Invoices dataset: directory {kaggle_invoices_path} not found.")

    # Added Synthetic Invoices Dataset loading
    if choice in ["5", "6"]:
        synthetic_invoices_path = "data/synthetic_invoices" # Assuming default location
        if Path(synthetic_invoices_path).exists():
            print(f"\n📊 Loading Synthetic Invoices dataset from {synthetic_invoices_path}...")
            try:
                synthetic_samples = data_utils.load_synthetic_invoice_dataset(synthetic_invoices_path)
                all_samples.extend(synthetic_samples)
                print(f"✅ Loaded {len(synthetic_samples)} samples from Synthetic Invoices.")
            except Exception as e:
                print(f"❌ Failed to load Synthetic Invoices dataset: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Skipping Synthetic Invoices dataset: directory {synthetic_invoices_path} not found.")

    if all_samples:
        print(f"\n📊 Total samples loaded: {len(all_samples)}")

        # Unify dataset format
        print("🔄 Unifying dataset format...")
        # This will process the raw samples into the format expected by LayoutLMDataset
        unified_samples = data_utils.unify_dataset_format(all_samples)

        if not unified_samples:
            print("❌ No valid samples remaining after unifying dataset format.")
            return

        # Create training split
        print("📊 Creating training/validation split...")
        train_samples, val_samples = data_utils.create_training_split(unified_samples)

        # Save processed datasets
        print("💾 Saving processed datasets to 'data' directory...")
        data_utils.save_processed_data(train_samples, "unified_train_samples.json")
        data_utils.save_processed_data(val_samples, "unified_val_samples.json")
        data_utils.save_processed_data(unified_samples, "unified_all_samples.json")

        print("✅ Datasets processed and saved successfully!")
        print(f"   - Training samples saved: {len(train_samples)}")
        print(f"   - Validation samples saved: {len(val_samples)}")

        print("\n💡 You can now train the LayoutLMv2 model using:")
        print("   python core/layoutlmv3_trainer.py")
        print("   (Ensure 'data/unified_train_samples.json' and 'data/unified_val_samples.json' are used by the trainer)")


    else:
        print("❌ No datasets were loaded.")


def show_psc_examples(data_utils: DataPreparationUtils, psc_data: dict):
    """Show PSC classification examples and statistics."""
    print("\n🔍 PSC Classification Examples (using keyword matching for demo)")
    print("-" * 35)

    if not psc_data or not psc_data.get('psc_mapping'):
        print("❌ No PSC data available. Cannot show examples.")
        return

    # Show statistics
    print(f"📊 PSC Statistics:")
    print(f"   Total PSC codes: {len(psc_data.get('all_pscs', []))}")
    print(f"   Spend categories: {len(psc_data.get('all_categories', []))}")
    print(f"   Portfolio groups: {len(psc_data.get('all_portfolios', []))}")

    # Show sample categories
    print(f"\n📋 Sample Spend Categories (top 5):")
    for i, category in enumerate(psc_data.get("all_categories", [])[:5], 1):
        psc_codes = psc_data.get("category_mapping", {}).get(category, [])
        print(f"   {i}. {category} ({len(psc_codes)} PSC codes)")

    # Show sample portfolios
    print(f"\n📁 Sample Portfolio Groups (top 5):")
    for i, portfolio in enumerate(psc_data.get("all_portfolios", [])[:5], 1):
        psc_codes = psc_data.get("portfolio_mapping", {}).get(portfolio, [])
        print(f"   {i}. {portfolio} ({len(psc_codes)} PSC codes)")

    # Interactive PSC lookup
    print(f"\n🔍 Interactive PSC Classification (using simple keyword matching):")
    while True:
        description = input("Enter item description (or 'quit' to exit): ").strip()
        if description.lower() in ["quit", "exit", "q"]:
            break

        if description:
            # This uses the simple keyword matcher from data_utils
            psc_result = data_utils.get_psc_by_description(description)
            if psc_result:
                print(f"  Best match found:")
                print(f"    PSC Code: {psc_result.get('psc', 'N/A')}")
                print(f"    Short Name: {psc_result.get('shortName', 'N/A')}")
                print(f"    Category: {psc_result.get('spendCategoryTitle', 'N/A')}")
                print(f"    Portfolio: {psc_result.get('portfolioGroup', 'N/A')}")
            else:
                print(f"  No matching PSC found for: '{description}'")


def run_demo(
    engine: IDPInferenceEngine, psc_data: dict, data_utils: DataPreparationUtils
):
    """Run demonstration with sample data."""
    print("\n🎯 Demo Mode - Sample Invoice Processing & PSC Classification")
    print("-" * 40)

    # Note: For a true demo, you'd process a real image.
    # This section simulates parts of the process and demonstrates PSC.
    print("This demo focuses on demonstrating PSC classification capabilities.")
    print("For full invoice processing, use option 'a' or 'b' from the main menu.")

    # Demonstrate PSC classification using the *engine's* classify_psc method
    # which uses the trained PSC model (or its fallback)
    print(f"\n🔍 PSC Classification Examples (using IDP Inference Engine):")
    test_items = [
        "Professional office chair with ergonomic design",
        "Laptop Computer Dell",
        "Printer Paper A4",
        "Conference Table Oak",
        "Cloud computing services",
        "IT consulting services",
        "Facility maintenance"
    ]

    for item_desc in test_items:
        # The engine.classify_psc uses the trained model if loaded, else falls back to keyword.
        psc_result = engine.classify_psc(item_desc, psc_data)

        print(f"  '{item_desc}'")
        print(f"    → PSC: {psc_result.get('psc', 'N/A')} - {psc_result.get('shortName', 'N/A')}")
        print(f"    → Category: {psc_result.get('spendCategoryTitle', 'N/A')}")
        print(f"    → Portfolio: {psc_result.get('portfolioGroup', 'N/A')}")
        print(f"    → Confidence: {psc_result.get('confidence', 0.0):.2%}")
        print()

    # Generate a sample structured output (simulated for demo purposes)
    # This structure mirrors what engine.process_document would return.
    demo_result = {
        "document_info": {
            "invoice_number": "DEMO-2024-001",
            "date": "2024-06-23",
            "vendor_name": "Demo Solutions Corp.",
            "total_amount": "$1999.99",
            "currency": "$",
        },
        "line_items": [
            {
                "item_description": "Office Chair Professional",
                "quantity": "2",
                "unit_price": "299.99",
                "line_total": "599.98",
                # Directly call engine.classify_psc for consistency
                "psc_classification": engine.classify_psc("Office Chair Professional", psc_data),
            },
            {
                "item_description": "Laptop Computer Dell",
                "quantity": "1",
                "unit_price": "1299.99",
                "line_total": "1299.99",
                # Directly call engine.classify_psc for consistency
                "psc_classification": engine.classify_psc("Laptop Computer Dell", psc_data),
            },
        ],
        "processing_metadata": {
            "processed_at": datetime.now().isoformat(),
            "extraction_method": "Demo Simulation",
            "total_items_extracted": 2,
        },
    }

    print("📋 Final Simulated JSON Output Structure:")
    print(json.dumps(demo_result, indent=2, ensure_ascii=False)) # ensure_ascii for non-ASCII chars

    # Save demo results using data_utils
    data_utils.save_processed_data(demo_result, "demo_output.json")
    print(f"\n💾 Demo results saved to: demo_output.json")


def show_training_status():
    """Check if models are trained and available."""
    print("\n🔍 Model Training Status:")
    print("-" * 25)

    layoutlm_path = Path("models/layoutlmv2_invoice_extractor")
    psc_path = Path("models/psc_classifier")

    if layoutlm_path.exists():
        print("✅ LayoutLMv2 model: Available")
    else:
        print("❌ LayoutLMv2 model: Not found")
        print("   Run: python core/layoutlmv3_trainer.py")

    if psc_path.exists():
        print("✅ PSC Classifier model: Available")
    else:
        print("❌ PSC Classifier model: Not found")
        print("   Run: python core/psc_classifier_trainer.py")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check-models":
            show_training_status()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Enhanced IDP Solution - Usage:")
            print("  python main.py                 # Interactive mode")
            print("  python main.py --check-models  # Check model availability")
            print("  python main.py --help          # Show this help")
            sys.exit(0)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error in main execution: {e}")
        print("Please check your installation and model files.")
        import traceback
        traceback.print_exc()

