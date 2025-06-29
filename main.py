#!/usr/bin/env python3
"""
Main entry point for the Enhanced IDP Solution
Demonstrates the complete invoice processing pipeline
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from core.idp_inference_engine import IDPInferenceEngine
from core.data_preparation_utils import DataPreparationUtils


def main():
    """Main function to demonstrate IDP pipeline usage."""
    print("🚀 Enhanced Intelligent Document Processing (IDP) Solution")
    print("=" * 60)

    # Initialize the inference engine
    print("1. Initializing IDP Inference Engine...")
    try:
        engine = IDPInferenceEngine()
        print("✅ Inference engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        print(
            "💡 Make sure models are trained and saved in the 'models' directory by running:"
        )
        print("   - python core/layoutlmv3_trainer.py")
        print("   - python core/item_categorizer_trainer.py")
        return

    # Initialize data preparation utilities
    print("\n2. Initializing Data Preparation Utilities...")
    data_utils = DataPreparationUtils()
    print("✅ Data utilities initialized successfully!")

    # Process documents
    print("\n3. Document Processing Options:")
    print("   a) Process single document")
    print("   b) Process multiple documents (batch)")
    print("   c) Load and prepare training datasets")

    choice = input("\nSelect option (a/b/c): ").lower().strip()

    if choice == "a":
        process_single_document(engine, data_utils)
    elif choice == "b":
        process_multiple_documents(engine, data_utils)
    elif choice == "c":
        load_training_datasets(data_utils)
    else:
        print("Invalid choice. Please select a, b, or c.")


def process_single_document(engine: IDPInferenceEngine, data_utils: DataPreparationUtils):
    """Process a single document."""
    print("\n📄 Single Document Processing")
    print("-" * 30)

    image_path = input(
        "Enter path to invoice image (e.g., data/kaggle_invoices/images/batch1-0001.jpg): "
    ).strip()

    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return

    print(f"Processing: {image_path}")
    try:
        # Engine handles PSC classification internally
        result = engine.process_document(image_path)

        # Display results
        print("\n✅ Processing completed!")
        print("\n📊 Extracted Information:")
        print(f"Invoice Number: {result['document_info'].get('invoice_number', 'N/A')}")
        print(f"Date: {result['document_info'].get('date', 'N/A')}")
        print(f"Due Date: {result['document_info'].get('due_date', 'N/A')}")
        print(f"Vendor: {result['document_info'].get('vendor_name', 'N/A')}")
        print(f"Vendor Address: {result['document_info'].get('vendor_address', 'N/A')}")
        print(f"Customer: {result['document_info'].get('customer_name', 'N/A')}")
        print(f"Customer Address: {result['document_info'].get('customer_address', 'N/A')}")
        print(f"Subtotal: {result['document_info'].get('subtotal', 'N/A')}")
        print(f"Tax Amount: {result['document_info'].get('tax_amount', 'N/A')}")
        print(f"Discount Amount: {result['document_info'].get('discount_amount', 'N/A')}")
        print(f"Total Amount: {result['document_info'].get('total_amount', 'N/A')}")
        print(f"Currency: {result['document_info'].get('currency', 'N/A')}")
        print(f"Line Items: {len(result['line_items'])}")

        # Show line items with PSC classification (output from engine)
        for i, item in enumerate(result["line_items"], 1):
            print(f"\nLine Item {i}:")
            print(f"  Description: {item.get('item_description', 'N/A')}")
            print(f"  Quantity: {item.get('quantity', 'N/A')}")
            print(f"  Unit Price: {item.get('unit_price', 'N/A')}")
            print(f"  Line Total: {item.get('line_total', 'N/A')}")

            # Display PSC classification result from the engine's output
            psc_class = item.get("psc_classification", {})
            print(
                f"  PSC: {psc_class.get('psc', 'N/A')} - {psc_class.get('shortName', 'N/A')}"
            )
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


def process_multiple_documents(engine: IDPInferenceEngine, data_utils: DataPreparationUtils):
    """Process multiple documents in batch."""
    print("\n📄 Batch Document Processing")
    print("-" * 30)

    folder_path = input(
        "Enter path to folder containing invoice images (e.g., data/kaggle_invoices/images/): "
    ).strip()
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
        # Engine handles PSC classification internally
        results = engine.batch_process([str(f) for f in image_files])

        print(f"\n✅ Batch processing completed!")
        print(
            f"📊 Successfully processed: {len([r for r in results if 'error' not in r])}/{len(results)} documents."
        )

        # Save batch results using data_utils
        data_utils.save_processed_data(results, "batch_results.json")
        print(
            "💾 All batch results aggregated and saved as 'batch_results.json' in the 'output' directory."
        )

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()


def load_training_datasets(data_utils: DataPreparationUtils):
    """Load and process training datasets."""
    print("\n📚 Training Dataset Loading")
    print("-" * 30)

    print("Available datasets:")
    print("   1) Kaggle Invoices Dataset")
    print("   2) Synthetic Invoices Dataset")
    print("   3) Load all available datasets")

    choice = input("\nSelect dataset(s) to load (1/2/3): ").strip()

    all_samples = []

    if choice in ["1", "3"]:
        kaggle_invoices_path = "data/kaggle_invoices"
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

    if choice in ["2", "3"]:
        synthetic_invoices_path = "data/synthetic_invoices"
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

        print("\n💡 You can now train the LayoutLMv3 model using:")
        print("   python core/layoutlmv3_trainer.py")
        print("   (Ensure 'data/unified_train_samples.json' and 'data/unified_val_samples.json' are used by the trainer)")

    else:
        print("❌ No datasets were loaded.")


def show_training_status():
    """Check if models are trained and available."""
    print("\n🔍 Model Training Status:")
    print("-" * 25)

    layoutlm_path = Path("models/layoutlmv3_invoice_extractor")
    psc_path = Path("models/unspsc_item_classifier")

    if layoutlm_path.exists():
        print("✅ LayoutLMv3 model: Available")
    else:
        print("❌ LayoutLMv3 model: Not found")
        print("   Run: python core/layoutlmv3_trainer.py")

    if psc_path.exists():
        print("✅ Item Categorizer model: Available")
    else:
        print("❌ Item Categorizer model: Not found")
        print("   Run: python core/item_categorizer_trainer.py")


def auto_process_samples(engine: IDPInferenceEngine, data_utils: DataPreparationUtils):
    """Automated processing of sample images."""
    print("\n🤖 Automated Sample Processing Mode")
    print("-" * 35)

    # Check if models are loaded
    try:
        # Check if engine has required models loaded
        if not hasattr(engine, 'layoutlm_model') or engine.layoutlm_model is None:
            print("❌ LayoutLMv3 model not loaded.")
            print("   Run: python core/layoutlmv3_trainer.py")
            return False

        if not hasattr(engine, 'item_categorizer') or engine.item_categorizer is None:
            print("❌ Item Categorizer model not loaded.")
            print("   Run: python core/item_categorizer_trainer.py")
            return False

        print("✅ Both models successfully loaded!")

    except Exception as e:
        print(f"❌ Error checking model status: {e}")
        return False

    # Define sample image folders to process
    sample_folders = [
        "data/kaggle_invoices/images",
        "data/synthetic_invoices/images"
    ]

    all_results = []
    processed_count = 0

    for folder_path in sample_folders:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"⚠️  Sample folder not found: {folder_path}")
            continue

        # Find image files
        image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"⚠️  No image files found in: {folder_path}")
            continue

        # Limit to first 5 files per folder for demo
        sample_files = image_files[:5]
        print(f"📊 Processing {len(sample_files)} sample files from {folder_path}...")

        try:
            # Process batch
            results = engine.batch_process([str(f) for f in sample_files])
            all_results.extend(results)
            processed_count += len([r for r in results if 'error' not in r])

        except Exception as e:
            print(f"❌ Failed to process samples from {folder_path}: {e}")

    if all_results:
        print(f"\n✅ Automated processing completed!")
        print(f"📊 Successfully processed: {processed_count}/{len(all_results)} documents.")

        # Save results
        output_file = "auto_processed_samples.json"
        data_utils.save_processed_data(all_results, output_file)
        print(f"💾 Results saved to: {output_file}")

        return True
    else:
        print("❌ No samples were processed.")
        return False


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check-models":
            show_training_status()
            sys.exit(0)
        elif sys.argv[1] == "--auto-process-samples":
            # Automated execution mode
            print("🚀 Enhanced IDP Solution - Automated Sample Processing")
            print("=" * 60)

            try:
                # Initialize engine
                print("1. Initializing IDP Inference Engine...")
                engine = IDPInferenceEngine()
                print("✅ Inference engine initialized successfully!")

                # Initialize data utils
                print("\n2. Initializing Data Preparation Utilities...")
                data_utils = DataPreparationUtils()
                print("✅ Data utilities initialized successfully!")

                # Run automated processing
                print("\n3. Starting automated sample processing...")
                success = auto_process_samples(engine, data_utils)

                if success:
                    print("\n🎉 Automated processing completed successfully!")
                    sys.exit(0)
                else:
                    print("\n❌ Automated processing failed.")
                    sys.exit(1)

            except Exception as e:
                print(f"❌ Failed to initialize inference engine: {e}")
                print("💡 Make sure models are trained and saved in the 'models' directory by running:")
                print("   - python core/layoutlmv3_trainer.py")
                print("   - python core/item_categorizer_trainer.py")
                sys.exit(1)

        elif sys.argv[1] == "--help":
            print("Enhanced IDP Solution - Usage:")
            print("  python main.py                        # Interactive mode")
            print("  python main.py --check-models         # Check model availability")
            print("  python main.py --auto-process-samples # Automated sample processing")
            print("  python main.py --help                 # Show this help")
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