I have a Python file named data_preparation_utils.py (content provided below). I need you to act as an expert Python engineer with a deep understanding of data science pipelines, and perform targeted, comprehensive enhancements on this file.

Your overarching goal is to transform this DataPreparationUtils class into the single, ultimate module for preparing all data required by our Intelligent Document Processing (IDP) project's machine learning models. This includes data for both UNSPSC hierarchical classification and LayoutLMv3-based invoice information extraction.

You must strictly adhere to the following plan to achieve optimal data preparation, robust error handling, and perfect alignment with our project's needs:

Full Transition from PSC to UNSPSC Data Handling:

Rename and Refactor load_psc_data:

Rename the method load_psc_data to load_unspsc_data.

Update its default url to "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/unspsc-codes.csv" and its default filepath to "data/data-unspsc-codes.csv".

Completely replace the internal logic of this method (including _load_csv_source and _clean_unspsc_dataframe) with the robust, generalized UNSPSC loading and processing methods we previously developed:

Integrate the _detect_column_mappings method: It must intelligently identify and standardize column names (e.g., 'Segment Code', 'Family Name', 'Class Name', 'Commodity Name/Title') from the raw UNSPSC CSV.

Integrate the _process_unspsc_data method: It must create a searchable_text column by concatenating relevant descriptive fields (prioritizing commodity_name/title, then class_name, family_name, segment_name). It must also robustly ensure a class_name column exists and is used as the primary classification target (falling back to commodity, family, or segment if class_name is empty/missing).

Update self.psc_data to self.unspsc_data and ensure it stores the processed DataFrame (containing searchable_text, class_name, and all original hierarchical columns). Remove self.unspsc_mapping and self.psc_mapping/category_mapping/portfolio_mapping as these are legacy PSC structures.

Remove PSC-specific Method: Delete the get_psc_by_description method entirely, as its functionality is superseded by the ItemCategorizerTrainer's model-based prediction.

Reinstate and Enhance Core Invoice Annotation Loaders for LayoutLMv3:

Keep and ensure the robustness of the following methods for loading and unifying invoice annotation datasets:

_load_json_annotations_with_images (as a helper)

load_kaggle_invoice_dataset

load_synthetic_invoice_dataset

remove any method dealing with the (load_funsd_dataset,load_sroie_dataset)
since we are to use only the synthtic_invoice_dataset and kaggle_invoice_dataset in the unified_dataset
unify_dataset_format: This method is crucial. Ensure it thoroughly validates and cleans words, boxes, and ner_tags, and normalizes the bounding boxes to [xmin, ymin, xmax, ymax] integer format as required by LayoutLMv3 processors. All samples must have image_path, words, boxes, ner_tags, and relations (even if empty list).

Perfect prepare_classification_data for UNSPSC Training:

Rename prepare_classification_data to prepare_unspsc_training_data to reflect its specific purpose.

Ensure this method solely focuses on preparing data from self.unspsc_data (the processed UNSPSC DataFrame) using its searchable_text and class_name columns.

It should correctly create and store self.label_to_idx and self.idx_to_label mappings for the UNSPSC classes.

Make sure the returned DataFrame for training contains only the searchable_text feature, the numerical label, and the original string class_name (for debugging/reference) and that it's properly shuffled.

JSON Serialization Fix in get_statistics:

Re-apply the fix to the get_statistics method. All numerical values derived from pandas/NumPy (e.g., nunique(), len(), sum()) must be explicitly cast to standard Python int or float before being included in the dictionary returned by this method, to prevent TypeError: Object of type int64 is not JSON serializable.

Remove or Isolate Semantic Search (TF-IDF):

The TF-IDF related methods (_initialize_vectorizer, find_unspsc_by_description, _is_search_ready, _keyword_search_fallback, and their corresponding self.vectorizer, self.unspsc_vectors attributes) are primarily for inference-time semantic search, not for preparing data for training our transformer models.

Remove all these TF-IDF related methods and attributes from DataPreparationUtils to ensure this class remains lean and focused solely on data preparation for training. If semantic search is needed, it should be implemented in the IDPInferenceEngine or a separate dedicated utility.

General Refinements:

Maintain consistent and informative logging (logger.info, logger.warning, logger.error with exc_info=True for exceptions).

Ensure type hints are used consistently.

Review all docstrings for clarity and accuracy.

Ensure all necessary imports are at the top.

Update the if __name__ == "__main__": block to demonstrate loading both UNSPSC data and at least one type of invoice annotation data, and how the prepared data would look.

Your final output should be the complete, updated data_preparation_utils.py file only, implementing all these changes."


