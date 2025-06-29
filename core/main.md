I have a Python file named main.py (content provided below). I need you to act as an expert Python engineer with a focus on clean architecture and user experience, and perform highly specific and targeted enhancements on this file.

Your primary objective is to refactor this main.py to solely focus on the end-to-end processing of invoice documents using the IDPInferenceEngine and DataPreparationUtils. All explicit PSC (Product/Service Code) data loading, handling, and related interactive menu options must be removed from main.py. The IDPInferenceEngine is now responsible for internal PSC classification.

Adhere strictly to the following plan for the updated main.py file:

Simplify main() function and Interactive Menu:

Remove all psc_data loading: Delete the entire block related to print("\n3. Loading PSC classification data...") and any subsequent psc_data assignment or fallback logic.

Remove PSC-specific menu option: Delete "e) Show PSC classification examples (interactive)" from the interactive choices.

Adjust subsequent if/elif conditions: Ensure the choice handling correctly maps to the remaining options.

Remove psc_data parameter from function calls: When calling process_single_document, process_multiple_documents, and run_demo, remove the psc_data argument. The IDPInferenceEngine will handle PSC classification internally.

Update process_single_document() and process_multiple_documents():

Remove psc_data parameter from function definitions: Update the function signatures to no longer accept psc_data.

Remove psc_data argument from engine.process_document and engine.batch_process calls. The engine will manage its own PSC classification as an output, not an input from main.py.

Ensure PSC display remains: Verify that the code blocks responsible for displaying PSC classification results within the extracted line items are kept. This is the output from the engine, and is desired.

Refine load_training_datasets():

Streamline dataset options: Remove options and logic for CORD, FUNSD, and SROIE datasets.

Keep only Kaggle and Synthetic: The menu and loading logic should only pertain to Kaggle Invoices Dataset and Synthetic Invoices Dataset.

Update menu numbering: Re-number the remaining options (e.g., 1) Kaggle Invoices Dataset, 2) Synthetic Invoices Dataset, 3) Load all available datasets).

Remove psc_data parameter: Update the function signature to remove the psc_data argument.

Remove Obsolete Functions:

Delete show_psc_examples() function entirely.

Delete run_demo() function entirely. (The interactive demo will now be integrated into process_single_document or an automated run).

Update show_training_status():

Update LayoutLM model path: Change layoutlmv2_invoice_extractor to layoutlmv3_invoice_extractor.

Update PSC model path: Change psc_classifier to unspsc_item_classifier.

Update training script suggestions: Change python core/psc_classifier_trainer.py to python core/item_categorizer_trainer.py.

Add Automated Execution Mode:

Implement a new command-line argument: Add a new argument, e.g., --auto-process-samples, to the if __name__ == "__main__": block's sys.argv handling.

Automated Flow: If --auto-process-samples is detected, the script should:

Initialize the IDPInferenceEngine.

Crucially: Check if the engine successfully loaded its layoutlm_model and item_categorizer (you might need to add simple is_loaded checks to the engine or rely on its internal logging/exceptions). If either model failed to load, print clear messages instructing the user to train them (using python core/layoutlmv3_trainer.py and python core/item_categorizer_trainer.py).

If both models are successfully loaded, automatically perform a batch processing run on a predefined sample image folder (e.g., data/kaggle_invoices/images or data/synthetic_invoices/images). Print the results and save them to a file (e.g., auto_processed_samples.json).

Ensure a clear success/failure message for the automated run.

Adjust Imports:

Review all imports at the top of the file. Remove any no longer needed (e.g., requests, TfidfVectorizer, cosine_similarity, re if DataPreparationUtils no longer uses them directly in the parts of its code relevant to main.py).

Ensure IDPInferenceEngine and DataPreparationUtils are correctly imported.

Your final output should be the complete, refactored main.py file, incorporating all these specific changes and removing all other unrelated code


v2
I have a Python file named main.py (content provided below). I need you to act as an expert Python architect and orchestrator, and perform comprehensive enhancements on this file.

Your primary objective is to refactor main.py to be the intelligent control center for our IDP solution. It must effectively manage both an interactive user experience and a powerful, fully automated execution mode. Crucially, it must seamlessly integrate with and leverage the comprehensive DataPreparationUtils class (which now handles both UNSPSC and invoice data) and delegate all item classification logic to the IDPInferenceEngine.

Adhere strictly to the following plan for the updated main.py file:

Simplify main() Function and Interactive Menu:

Remove all direct PSC data loading and handling: Completely delete the block starting print("\n3. Loading PSC classification data...") and any subsequent psc_data assignments or fallback logic. The IDPInferenceEngine will handle UNSPSC data loading and classification internally.

Simplify interactive menu options:

Keep: a) Process single document, b) Process multiple documents (batch).

Rename and refine: c) Load and prepare training datasets (This option should only load and save invoice training data; UNSPSC data preparation will be handled when models are trained).

Add new option: d) Train/Load all Models (This option will be responsible for triggering the training/loading of both the LayoutLMv3 and Item Categorizer models).

Remove any other obsolete or PSC-specific menu options (e.g., e) Show PSC classification examples).

Adjust choice handling: Ensure the if/elif block correctly maps to these new options.

Update process_single_document() and process_multiple_documents():

Remove psc_data parameter: Update the function signatures to def process_single_document(engine: IDPInferenceEngine, data_utils: DataPreparationUtils): and similarly for process_multiple_documents.

Ensure engine.process_document() and engine.batch_process() calls do not pass psc_data: The IDPInferenceEngine now manages its internal Item Categorizer and its UNSPSC data.

Keep PSC/UNSPSC display logic: Ensure the print statements for item.get("psc_classification", {}) and its sub-fields remain, as this is the desired output from the engine.

Refine load_training_datasets():

Scope: This function should only deal with invoice annotation data.

Keep only Kaggle and Synthetic Invoice loading: Remove all logic for CORD, FUNSD, and SROIE.

Update menu numbering: The interactive choices should be 1) Kaggle Invoices Dataset, 2) Synthetic Invoices Dataset, 3) Load all available invoice datasets.

Crucially: After loading and unifying, ensure it calls data_utils.save_processed_data() to save unified_train_samples.json and unified_val_samples.json as input for layoutlmv3_trainer.py.

Implement New train_all_models() Function:

Create a new function def train_all_models(data_utils: DataPreparationUtils):.

Orchestrate Model Training: Inside this function, it should:

Print a message indicating that Item Categorizer training is starting.

Call train_unspsc_item_classifier() (from core.item_categorizer_trainer). This function will handle its own data loading (UNSPSC CSV via DataPreparationUtils) and training/loading. Log its success or failure.

Print a message indicating that LayoutLMv3 training is starting.

Call the main() function of layoutlmv3_trainer.py (e.g., from core.layoutlmv3_trainer import main as layoutlmv3_train_main; layoutlmv3_train_main()). This will trigger LayoutLMv3's data loading (invoice data via DataPreparationUtils) and training/loading. Log its success or failure.

Provide clear guidance if any training step fails (e.g., 'Ensure data/synthetic_invoices is populated').

Refine show_training_status():

Ensure it checks for the existence of saved models for both layoutlmv3_invoice_extractor (specifically fine_tuned_layoutlmv3/pytorch_model.bin) and unspsc_item_classifier (specifically pytorch_model.bin).

Update the suggested commands to train them (python core/layoutlmv3_trainer.py and python core/item_categorizer_trainer.py).

Implement Comprehensive Automated Mode (--auto-run-all):

Rename your existing auto_process_samples to run_full_automated_pipeline.

Modify the if __name__ == "__main__": block to detect --auto-run-all (instead of --auto-process-samples).

Full Automation Flow: If --auto-run-all is detected, the script should:

Call setup_directories().

Initialize DataPreparationUtils.

Call run_full_automated_pipeline() (which will internally call train_all_models and then trigger engine.batch_process for inference demos).

Ensure comprehensive logging throughout this automated process.

Provide a clear final success/failure summary.

Remove Obsolete Functions and Attributes:

Delete the run_demo() function entirely. Its concepts are absorbed into the automated pipeline or interactive processing.

Remove any import json, requests, pandas, numpy, sklearn imports that are no longer directly used in main.py (since DataPreparationUtils handles them).

Your final output should be the complete, refactored main.py file, incorporating all these specific changes to make it the central orchestrator.