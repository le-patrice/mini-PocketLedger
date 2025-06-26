#!/usr/bin/env python3
"""
Ultimate Perfect LayoutLMv2 Annotation Generator - State-of-the-art OCR-Ground Truth Alignment
Generates highly accurate annotations by focusing on robust fuzzy matching, normalization,
and comprehensive entity tagging. This version reads CSV data from an external file.
"""
#!pip install pandas easyocr opencv-python fuzzywuzzy python-levenshtein Pillow numpy

import io # Still needed for pandas.read_csv even for files, as it uses underlying IO functionality.
import pandas as pd
import json
import logging
from pathlib import Path
import easyocr
import cv2
import re
from typing import List, Dict, Any, Tuple, Optional
from fuzzywuzzy import fuzz, process
import numpy as np
import unicodedata
from dateutil import parser as date_parser
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configurable field mapping - maps JSON paths to LayoutLMv2 entity labels
FIELD_MAPPING = {
    "invoice.invoice_number": "INVOICE_NUM",
    "invoice.invoice_date": "DATE",
    "invoice.due_date": "DUE_DATE",
    "invoice.seller_name": "VENDOR",
    "invoice.seller_address": "VENDOR_ADDRESS",
    "invoice.client_name": "CUSTOMER_NAME",
    "invoice.client_address": "CUSTOMER_ADDRESS",
    "invoice.tax_id": "TAX_ID", # Added assuming your GT JSON will include this for consistency
    
    "subtotal.total": "TOTAL_AMT",
    "subtotal.tax": "TAX",
    "subtotal.discount": "DISCOUNT",
    
    "items.description": "ITEM_DESC",
    "items.quantity": "QTY",
    "items.total_price": "LINE_TOTAL",
    
    "payment_instructions.due_date": "DUE_DATE", # Can be redundant if already in invoice.due_date
    "payment_instructions.bank_name": "BANK_NAME",
    "payment_instructions.account_number": "ACCOUNT_NUM",
    "payment_instructions.iban": "IBAN", # Added assuming your GT JSON will include this for consistency
    "payment_instructions.payment_method": "PAYMENT_METHOD" 
}

# Fields considered critical for strict matching, ordered by importance for tagging priority
# Priority will be given to these fields during alignment, and stricter matching thresholds applied.
CRITICAL_FIELDS = {
    "INVOICE_NUM", "DATE", "TOTAL_AMT", "TAX", "DISCOUNT",
    "VENDOR", "CUSTOMER_NAME", "VENDOR_ADDRESS", "CUSTOMER_ADDRESS",
    "QTY", "LINE_TOTAL", "TAX_ID", "IBAN", "ACCOUNT_NUM", "BANK_NAME", "PAYMENT_METHOD"
}
NUMERIC_FIELDS = {"TOTAL_AMT", "TAX", "DISCOUNT", "QTY", "LINE_TOTAL"}
DATE_FIELDS = {"DATE", "DUE_DATE"}

class TextNormalizer:
    """Utility for normalizing text, numbers, and dates for robust comparison."""
    @staticmethod
    def normalize_number(text: str) -> str:
        """
        Normalizes numerical strings by removing common currency symbols, commas, and spaces.
        Handles both dot and comma as decimal separators.
        """
        if not text: return ""
        text = str(text).strip()
        
        # Remove common currency symbols and non-numeric characters except digits, '.', ','
        text = re.sub(r'[^\d.,]+', '', text) 
        
        # Determine decimal separator and format consistently to dot
        if '.' in text and ',' in text:
            # If both are present, assume comma is thousands separator (e.g., 1,234.56 -> 1234.56)
            text = text.replace(',', '') 
        elif ',' in text:
            # If only comma, assume it's the decimal separator (e.g., 123,45 -> 123.45)
            text = text.replace(',', '.')
            
        return text

    @staticmethod
    def normalize_date(text: str) -> str:
        """Normalizes date strings to a consistent %m/%d/%Y format using dateutil.parser."""
        if not text: return ""
        try:
            # fuzzy=True allows for more lenient parsing
            parsed_date = date_parser.parse(str(text), fuzzy=True)
            return parsed_date.strftime("%m/%d/%Y")
        except date_parser.ParserError:
            logger.debug(f"Could not parse date with dateutil: '{text}'. Falling back to simple clean.")
            return re.sub(r'[^\d/\-.]', '', str(text)).strip()
        except Exception as e:
            logger.debug(f"Unexpected error normalizing date '{text}': {e}")
            return str(text).strip()
    
    @staticmethod
    def normalize_text(text: str, field_type: str = "text") -> str:
        """
        Normalizes general text, applying specific normalization for numbers and dates.
        Removes extra whitespace and converts to lowercase.
        """
        if not text: return ""
        text = unicodedata.normalize('NFKD', str(text).strip())
        
        if field_type == "number": return TextNormalizer.normalize_number(text)
        if field_type == "date": return TextNormalizer.normalize_date(text)
        
        return re.sub(r'\s+', ' ', text.lower())

class AnnotationGenerator:
    """Enhanced annotation generator with improved alignment capabilities."""
    
    def __init__(self):
        """Initialize the annotation generator."""
        self.ocr_reader = None
        self.normalizer = TextNormalizer()
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed_ocr': 0,
            'failed_json': 0,
            'failed_alignment': 0,
            'skipped_missing_image': 0
        }
        
    def initialize_ocr(self):
        """Initializes EasyOCR reader, trying GPU first then falling back to CPU."""
        if self.ocr_reader is None:
            logger.info("Initializing EasyOCR reader...")
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=True)
                logger.info("EasyOCR reader initialized with GPU.")
            except Exception as e:
                logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR reader initialized with CPU.")
                
    def get_word_level_ocr(self, image_path: Path) -> List[Tuple[str, List[int], float]]:
        """
        Performs OCR on the image and returns word-level results with confidence scores and cleaned text.
        
        Returns:
            List of (word_text, [x_min, y_min, x_max, y_max], confidence)
        """
        try:
            if self.ocr_reader is None:
                self.initialize_ocr()
                
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                self.stats['skipped_missing_image'] += 1
                return []
                
            results = self.ocr_reader.readtext(image, detail=1, paragraph=False)
            
            word_level_data = []
            for (bbox, text, confidence) in results:
                # Apply a confidence threshold to filter out weak OCR detections
                if confidence > 0.3 and text.strip():
                    # Extract bounding box coordinates
                    x_coords = [int(p[0]) for p in bbox]
                    y_coords = [int(p[1]) for p in bbox]
                    x_min, y_min = min(x_coords), min(y_coords)
                    x_max, y_max = max(x_coords), max(y_coords)
                    
                    clean_text = text.strip()
                    if clean_text:
                        word_level_data.append((clean_text, [x_min, y_min, x_max, y_max], confidence))
                        
            logger.debug(f"OCR extracted {len(word_level_data)} words from {image_path.name}")
            return word_level_data
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}", exc_info=True)
            self.stats['failed_ocr'] += 1
            return []
    
    def extract_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Extracts value from nested dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
            return current
        except Exception as e:
            logger.debug(f"Error extracting nested value for path '{path}': {e}")
            return None
    
    def flatten_ground_truth(self, ground_truth: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Flattens ground truth JSON into (value, label) tuples for easier processing.
        Values are extracted and their corresponding LayoutLMv2 entity labels are assigned.
        """
        entities = []
        
        # Process non-array fields using the global FIELD_MAPPING
        for json_path, entity_label in FIELD_MAPPING.items():
            if not json_path.startswith('items.'):
                value = self.extract_nested_value(ground_truth, json_path)
                # Ensure value is not None and not just empty whitespace
                if value is not None and str(value).strip():
                    # Special handling for addresses: replace newline with space for matching
                    if entity_label in ["VENDOR_ADDRESS", "CUSTOMER_ADDRESS"]:
                        formatted_value = str(value).replace('\n', ' ').strip()
                        entities.append((formatted_value, entity_label))
                    else:
                        entities.append((str(value).strip(), entity_label))
        
        # Process items array
        items = ground_truth.get('items', [])
        for item_idx, item in enumerate(items):
            if isinstance(item, dict):
                for field in ['description', 'quantity', 'total_price']:
                    if field in item and item[field] is not None and str(item[field]).strip():
                        entity_label = FIELD_MAPPING.get(f"items.{field}")
                        if entity_label:
                            # For descriptions, replace newlines with spaces for matching
                            if field == 'description':
                                formatted_value = str(item[field]).replace('\n', ' ').strip()
                                entities.append((formatted_value, entity_label))
                            else:
                                entities.append((str(item[field]).strip(), entity_label))
        
        logger.debug(f"Extracted {len(entities)} ground truth entities for flattening.")
        return entities
    
    def find_best_match(self, gt_text: str, label: str, ocr_data: List[Tuple[str, List[int], float]], used_indices: set) -> List[int]:
        """
        Finds the best matching OCR words for a ground truth text.
        Employs multiple strategies (exact, sequence, fuzzy, token set) with normalization and
        considers whether OCR words have already been used.
        """
        if not gt_text or not ocr_data:
            return []
        
        is_critical = label in CRITICAL_FIELDS
        field_type = "number" if label in NUMERIC_FIELDS else "date" if label in DATE_FIELDS else "text"
        gt_norm = self.normalizer.normalize_text(gt_text, field_type)
        gt_words_norm_list = [self.normalizer.normalize_text(w, field_type) for w in gt_text.split() if w]
        
        # Caching normalized OCR words for efficiency in loops
        normalized_ocr_words = [self.normalizer.normalize_text(ocr_word, field_type) for ocr_word, _, _ in ocr_data]

        best_score_overall = -1
        best_indices_overall = []

        # Strategy 1: Exact match for single-word ground truths or if multi-word, check for exact sequence
        if len(gt_words_norm_list) == 1 and len(gt_text.split()) == 1: # Ensure it's truly a single word GT
            for i, ocr_norm_word in enumerate(normalized_ocr_words):
                if i in used_indices: continue
                if gt_norm == ocr_norm_word:
                    logger.debug(f"STRATEGY 1 (Exact Single Word) match for '{gt_text}' ({label}): '{ocr_data[i][0]}' at index {i}")
                    return [i]
        elif len(gt_words_norm_list) > 1: # Multi-word GT, try strict sequence
            for start in range(len(ocr_data) - len(gt_words_norm_list) + 1):
                current_ocr_indices = list(range(start, start + len(gt_words_norm_list)))
                if any(idx in used_indices for idx in current_ocr_indices):
                    continue
                
                # Check normalized words directly
                ocr_sub_sequence_norm = normalized_ocr_words[start:start+len(gt_words_norm_list)]
                if gt_words_norm_list == ocr_sub_sequence_norm:
                     logger.debug(f"STRATEGY 1 (Exact Multi-Word Sequence) match for '{gt_text}' ({label}): '{' '.join([ocr_data[i][0] for i in current_ocr_indices])}' at {current_ocr_indices}")
                     return current_ocr_indices

        # Strategy 2: Token Set Ratio for Flexible Phrase Matching (Addresses, Descriptions, Names)
        # This is robust to word reordering, missing minor words, or extra words.
        # Iterate over possible OCR word windows to find the best token set match
        # Max window size can be capped to allow for some OCR noise (extra words)
        max_window_size = min(len(gt_words_norm_list) * 2 + 5, len(ocr_data)) 
        
        for window_size in range(1, max_window_size + 1):
            for start in range(len(ocr_data) - window_size + 1):
                current_ocr_indices = list(range(start, start + window_size))
                if any(idx in used_indices for idx in current_ocr_indices):
                    continue
                
                ocr_segment_norm = " ".join(normalized_ocr_words[start : start + window_size])
                
                score = fuzz.token_set_ratio(gt_norm, ocr_segment_norm)
                
                # Prioritize if score is high and it covers the necessary words well
                if score > best_score_overall: # Always keep the highest score
                    best_score_overall = score
                    best_indices_overall = current_ocr_indices
                    logger.debug(f"STRATETERY 2 (Token Set) candidate for '{gt_text}' ({label}): '{' '.join([ocr_data[i][0] for i in current_ocr_indices])}' with score {score} at {current_ocr_indices}")

        # Final decision based on the best overall score from token set ratio
        # Set minimum threshold for a valid match
        final_threshold = 90 if is_critical else 75 # Adjust as needed based on empirical results
        
        if best_score_overall >= final_threshold and best_indices_overall:
            logger.debug(f"Best Token Set Match for '{gt_text}' ({label}): indices {best_indices_overall} (Score: {best_score_overall})")
            return best_indices_overall
            
        logger.debug(f"No good match found for ground truth: '{gt_text[:50]}...' (Label: {label})")
        return []
    
    def align_ocr_with_ground_truth(self, ocr_data: List[Tuple[str, List[int], float]], 
                                   ground_truth: Dict[str, Any]) -> List[str]:
        """
        Aligns OCR results with ground truth entities to generate BIO-tagged labels.
        Prioritizes critical fields and longer entities to ensure optimal tagging.
        """
        num_ocr_words = len(ocr_data)
        ner_tags = ["O"] * num_ocr_words
        
        gt_entities = self.flatten_ground_truth(ground_truth)
        
        # Sort entities: Critical fields first, then by length (longer first)
        # This order is crucial for correct BIO tagging, ensuring multi-word entities
        # and important fields claim their words before shorter, less critical ones.
        gt_entities.sort(key=lambda x: (x[1] not in CRITICAL_FIELDS, -len(x[0])))
        
        used_indices = set()
        
        for gt_text, entity_label in gt_entities:
            match_indices = self.find_best_match(gt_text, entity_label, ocr_data, used_indices)
            
            # Filter out indices that have already been tagged by a higher priority entity
            available_indices = [idx for idx in match_indices if idx not in used_indices]
            
            if available_indices:
                available_indices.sort() # Ensure indices are sorted for B-I-O consistency
                
                for i, idx in enumerate(available_indices):
                    # Only tag if the slot is still 'O'. This prevents overwriting by lower-priority matches
                    if ner_tags[idx] == "O": 
                        prefix = "B-" if i == 0 else "I-"
                        ner_tags[idx] = f"{prefix}{entity_label}"
                        used_indices.add(idx)
                    else:
                        logger.debug(f"OCR word {idx} ('{ocr_data[idx][0]}') already tagged as '{ner_tags[idx]}'. "
                                     f"Skipping re-tagging for '{entity_label}' ('{gt_text}').")
                
                logger.debug(f"Tagged '{gt_text[:30]}...' ({entity_label}) at indices {available_indices}")
            else:
                logger.debug(f"No match for GT: '{gt_text[:30]}...' (Label: {entity_label}).")
        
        tagged_count = sum(1 for tag in ner_tags if tag != "O")
        logger.info(f"Alignment: Tagged {tagged_count}/{num_ocr_words} OCR words ({tagged_count/num_ocr_words*100:.1f}% coverage)")
        
        return ner_tags
    
    def extract_invoice_metadata(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts metadata for the invoice, ensuring total_amount is correctly parsed.
        Prioritizes the 'subtotal.total' from JSON, falls back to summing line items.
        """
        invoice_data = ground_truth.get("invoice", {})
        subtotal_data = ground_truth.get("subtotal", {})
        payment_data = ground_truth.get("payment_instructions", {})
        
        total_amount = 0.0
        # Strategy 1: Try to get total from subtotal.total directly from the ground truth JSON
        total_str_from_gt = subtotal_data.get("total", "")
        if total_str_from_gt:
            try:
                # IMPORTANT FIX: Ensure normalize_number always returns a string parseable by float, or handle ""
                normalized_total_str = self.normalizer.normalize_number(total_str_from_gt)
                total_amount = float(normalized_total_str) if normalized_total_str else 0.0
                logger.debug(f"Extracted total from subtotal.total in GT: {total_amount}")
            except ValueError:
                logger.warning(f"Could not parse total_amount from 'subtotal.total' in GT: '{total_str_from_gt}'.")
        
        # Strategy 2: Fallback to calculating from line items if subtotal.total is missing or invalid
        if total_amount == 0.0 or np.isnan(total_amount):
            calculated_total = 0.0
            for item in ground_truth.get("items", []):
                if isinstance(item, dict) and 'total_price' in item and item['total_price'] is not None:
                    try:
                        # IMPORTANT FIX: Same handling for line item total_price
                        normalized_price_str = self.normalizer.normalize_number(str(item['total_price']))
                        price = float(normalized_price_str) if normalized_price_str else 0.0
                        calculated_total += price
                    except ValueError:
                        logger.warning(f"Could not parse line item total_price for calculation: '{item.get('total_price')}'")
            
            if calculated_total > 0:
                total_amount = calculated_total
                logger.info(f"Used calculated total from line items as fallback: ${total_amount:.2f}")

        # IMPORTANT FIX: Apply same robust conversion for tax_amount and discount_amount
        tax_amount = 0.0
        normalized_tax_str = self.normalizer.normalize_number(subtotal_data.get("tax", "0.0"))
        try:
            tax_amount = float(normalized_tax_str) if normalized_tax_str else 0.0
        except ValueError:
            logger.warning(f"Could not parse tax_amount: '{normalized_tax_str}'")

        discount_amount = 0.0
        normalized_discount_str = self.normalizer.normalize_number(subtotal_data.get("discount", "0.0"))
        try:
            discount_amount = float(normalized_discount_str) if normalized_discount_str else 0.0
        except ValueError:
            logger.warning(f"Could not parse discount_amount: '{normalized_discount_str}'")


        return {
            "invoice_id": invoice_data.get("invoice_number", ""),
            "date": self.normalizer.normalize_date(invoice_data.get("invoice_date", "")),
            "vendor_name": invoice_data.get("seller_name", ""),
            "customer_name": invoice_data.get("client_name", ""),
            "total_amount": total_amount,
            # Adding other metadata fields for completeness if desired
            "due_date": self.normalizer.normalize_date(invoice_data.get("due_date", "")),
            "tax_amount": tax_amount,
            "discount_amount": discount_amount,
            "bank_name": payment_data.get("bank_name", ""),
            "account_number": payment_data.get("account_number", ""),
            "payment_method": payment_data.get("payment_method", ""),
            "iban": payment_data.get("iban", ""),
            "tax_id": invoice_data.get("tax_id", "")
        }
    
    def extract_line_items(self, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts line items from ground truth."""
        items = ground_truth.get("items", [])
        line_items = []
        
        for item in items:
            if isinstance(item, dict):
                line_items.append({
                    "description": item.get("description", ""),
                    "quantity": item.get("quantity", ""),
                    "total_price": item.get("total_price", ""),
                    "psc": "",
                    "shortName": "",
                    "spendCategoryTitle": "",
                    "portfolioGroup": ""
                })
        
        return line_items
    
    def generate_annotation(self, image_path: Path, json_data_str: str, 
                          output_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Generates a complete LayoutLMv2 annotation for an image by performing OCR,
        aligning with ground truth, and structuring the output.
        """
        try:
            try:
                ground_truth = json.loads(json_data_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON for {image_path.name}: {e}", exc_info=True)
                self.stats['failed_json'] += 1
                return None
            
            ocr_results = self.get_word_level_ocr(image_path)
            if not ocr_results:
                logger.warning(f"No valid OCR results for {image_path.name}. Annotation skipped.")
                return None
            
            words = [item[0] for item in ocr_results]
            boxes = [item[1] for item in ocr_results]
            confidences = [item[2] for item in ocr_results]
            
            ner_tags = self.align_ocr_with_ground_truth(ocr_results, ground_truth)
            
            if not (len(words) == len(boxes) == len(ner_tags) == len(confidences)):
                logger.error(f"Length mismatch after alignment for {image_path.name}. Words={len(words)}, Boxes={len(boxes)}, Tags={len(ner_tags)}, Confidences={len(confidences)}", exc_info=True)
                self.stats['failed_alignment'] += 1
                return None
            
            invoice_metadata = self.extract_invoice_metadata(ground_truth)
            line_items = self.extract_line_items(ground_truth)
            
            annotation = {
                "words": words,
                "boxes": boxes,
                "ner_tags": ner_tags,
                "ocr_confidences": confidences,
                "document_psc": "",
                "document_category": "",
                "document_portfolio": "",
                "line_items_psc": line_items,
                "psc_breakdown": {},
                "portfolio_breakdown": {},
                "invoice_metadata": invoice_metadata,
                "alignment_stats": {
                    "total_words": len(words),
                    "tagged_words": sum(1 for tag in ner_tags if tag != "O"),
                    "tag_coverage": (sum(1 for tag in ner_tags if tag != "O") / len(words)) if words else 0
                }
            }
            
            output_file = output_dir / f"{image_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)
            
            coverage = annotation['alignment_stats']['tag_coverage']
            total = annotation['invoice_metadata']['total_amount']
            logger.info(f"Generated {output_file.name} (Coverage: {coverage:.1%}, Total: ${total:.2f})")
            
            self.stats['successful'] += 1
            return annotation
            
        except Exception as e:
            logger.error(f"Failed to generate annotation for {image_path.name} due to unexpected error: {e}", exc_info=True)
            self.stats['failed_alignment'] += 1
            return None
    
    def process_batch(self, csv_path: str, images_dir: str, output_dir: str):
        """Processes a batch of images and generates annotations from an external CSV file."""
        csv_file = Path(csv_path)
        images_path = Path(images_dir)
        output_path = Path(output_dir)
        
        if not csv_file.exists(): raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not images_path.exists(): raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            df = pd.read_csv(csv_file) # Read directly from file path
            logger.info(f"Loaded {len(df)} entries from {csv_path}.")
        except Exception as e:
            raise ValueError(f"Error reading CSV file '{csv_path}': {e}")
        
        for idx, row in df.iterrows():
            self.stats['processed'] += 1
            
            file_name = row["File Name"]
            json_data = row["Json Data"]
            
            image_path = images_path / file_name
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}. Skipping.")
                self.stats['skipped_missing_image'] += 1
                continue
            
            logger.info(f"Processing {file_name} ({idx + 1}/{len(df)})")
            
            self.generate_annotation(image_path, json_data, output_path)
        
        self.print_summary()
    
    def print_summary(self):
        """Prints a summary of the batch processing results."""
        logger.info("=== Processing Summary ===")
        logger.info(f"Total documents processed: {self.stats['processed']}")
        logger.info(f"Successfully annotated: {self.stats['successful']}")
        logger.info(f"Failed (OCR issues): {self.stats['failed_ocr']}")
        logger.info(f"Failed (JSON parsing issues): {self.stats['failed_json']}")
        logger.info(f"Failed (Alignment/General errors): {self.stats['failed_alignment']}")
        logger.info(f"Skipped (Missing image files): {self.stats['skipped_missing_image']}")
        
        if self.stats['processed'] > 0:
            success_rate = self.stats['successful'] / self.stats['processed'] * 100
            logger.info(f"Overall Annotation Success Rate: {success_rate:.1f}%")
        else:
            logger.info("No documents were processed.")

def main():
    """Main function to run the annotation generation process."""
    # IMPORTANT: Place your 'batch1_1.csv' file in the same directory as this script,
    # or provide the full path to it.
    CSV_FILE_PATH = "./batch1_1.csv"
    
    # IMPORTANT: Adjust IMAGES_BASE_DIRECTORY to where your JPG images are located.
    # Example: If your "File Name" in CSV is "batch1-0494.jpg"
    # and the actual image is at "/content/my_images_folder/batch1-0494.jpg",
    # then IMAGES_BASE_DIRECTORY should be "./my_images_folder".
    # Make sure this path exists and contains your images.
    IMAGES_BASE_DIRECTORY = "." 
    
    OUTPUT_ANNOTATIONS_DIRECTORY = "./annotations1"
    
    generator = AnnotationGenerator()
    
    try:
        generator.process_batch(
            csv_path=CSV_FILE_PATH, # Now passing the CSV file path
            images_dir=IMAGES_BASE_DIRECTORY,
            output_dir=OUTPUT_ANNOTATIONS_DIRECTORY
        )
        logger.info("Annotation generation process completed.")
        
    except Exception as e:
        logger.critical(f"FATAL: Annotation generation process failed unexpectedly: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
