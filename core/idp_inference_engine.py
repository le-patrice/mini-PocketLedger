import torch
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)  # Changed from LayoutLMv2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastai.text.all import *  # For fastai model loading
import numpy as np
import cv2
import easyocr
from datetime import datetime
import re  # For fallback extraction

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IDPInferenceEngine:
    """Complete inference engine for invoice processing with LayoutLMv3 and PSC classification."""

    def __init__(
        self,
        layoutlm_model_path: str = "models/layoutlmv3_invoice_extractor",  # Updated default path for v3
        psc_model_path: str = "models/psc_classifier",
        use_fallback: bool = False,  # Added flag for explicit fallback
    ):
        self.layoutlm_model_path = Path(layoutlm_model_path)
        self.psc_model_path = Path(psc_model_path)
        self.use_fallback = (
            use_fallback  # If true, skip model loading and use regex/keyword
        )

        # Initialize OCR reader
        # It is recommended to create the EasyOCR reader once for performance.
        self.ocr_reader = easyocr.Reader(["en"])
        logger.info("EasyOCR reader initialized.")

        # Initialize models (only if not using fallback)
        self.layoutlm_processor: Optional[LayoutLMv3Processor] = (
            None  # Type hint update
        )
        self.layoutlm_model: Optional[LayoutLMv3ForTokenClassification] = (
            None  # Type hint update
        )
        self.psc_learner: Optional[Learner] = None
        self.psc_id_to_label: Optional[Dict[int, str]] = None

        if not self.use_fallback:
            self._load_models()
        else:
            logger.warning(
                "IDPInferenceEngine initialized in FALLBACK mode. ML models will not be loaded."
            )

        # Define field mapping for extraction (should match your LayoutLMv3 trainer labels)
        self.id_to_label_map = {
            1: "INVOICE_NUM",
            2: "DATE",
            3: "DUE_DATE",
            4: "VENDOR_NAME",
            5: "VENDOR_ADDRESS",
            6: "CUSTOMER_NAME",
            7: "CUSTOMER_ADDRESS",
            8: "ITEM_DESCRIPTION",
            9: "QUANTITY",
            10: "UNIT_PRICE",
            11: "LINE_TOTAL",
            12: "SUBTOTAL",
            13: "TAX_AMOUNT",
            14: "DISCOUNT_AMOUNT",
            15: "TOTAL_AMOUNT",
            16: "CURRENCY",
            17: "HEADER",
        }
        # In a real system, this mapping should be loaded from the saved model config/processor
        # For now, it's a hardcoded placeholder to match trainer's base_labels.

    def _load_models(self):
        """Loads LayoutLMv3 and PSC classification models."""
        logger.info("Attempting to load ML models...")
        try:
            # Load LayoutLMv3 Processor and Model
            if self.layoutlm_model_path.exists():
                self.layoutlm_processor = LayoutLMv3Processor.from_pretrained(
                    str(self.layoutlm_model_path)
                )  # Changed
                self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained(
                    str(self.layoutlm_model_path)
                )  # Changed
                self.layoutlm_model.eval()  # Set to evaluation mode
                logger.info(f"LayoutLMv3 model loaded from {self.layoutlm_model_path}")

                # Update ID to label map from the loaded model's config
                if (
                    hasattr(self.layoutlm_model.config, "id2label")
                    and self.layoutlm_model.config.id2label
                ):
                    self.id_to_label_map = self.layoutlm_model.config.id2label
                    logger.info("Updated LayoutLMv3 label map from model config.")
                else:
                    logger.warning(
                        "LayoutLMv3 model config has no id2label. Using default internal map."
                    )

            else:
                raise FileNotFoundError(
                    f"LayoutLMv3 model not found at {self.layoutlm_model_path}"
                )
        except Exception as e:
            logger.error(
                f"Failed to load LayoutLMv3 model: {e}. Falling back to regex extraction."
            )
            self.use_fallback = True
            self.layoutlm_processor = None
            self.layoutlm_model = None

        try:
            # Load FastAI PSC Classifier Learner
            # PSCClassifierTrainer's load_trained_model handles the FastAI specific loading
            if self.psc_model_path.exists():
                from psc_classifier_trainer import (
                    PSCClassifierTrainer,
                )  # Import locally to avoid circular dep

                _, trainer_instance = PSCClassifierTrainer.load_trained_model(
                    str(self.psc_model_path)
                )
                self.psc_learner = trainer_instance.create_learner(
                    trainer_instance.create_dataloaders(
                        pd.DataFrame([{"text": "dummy", "psc": "Z999"}])
                    )  # Dummy dls
                )  # Recreate learner structure to load weights
                self.psc_learner.load_state_dict(
                    torch.load(self.psc_model_path / "learner.pkl").state_dict()
                )  # Load state dict
                self.psc_learner.eval()  # Set to evaluation mode
                self.psc_id_to_label = trainer_instance.idx_to_psc
                logger.info(f"PSC classifier model loaded from {self.psc_model_path}")
            else:
                raise FileNotFoundError(
                    f"PSC classifier model not found at {self.psc_model_path}"
                )
        except Exception as e:
            logger.error(
                f"Failed to load PSC classifier model: {e}. Falling back to keyword matching."
            )
            # self.psc_learner will remain None, triggering keyword fallback in classify_psc

    def extract_text_and_boxes(
        self, image_path: str
    ) -> Tuple[List[str], List[List[int]]]:
        """Performs OCR using EasyOCR and returns extracted words and their bounding boxes."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Perform OCR
            results = self.ocr_reader.readtext(image)

            words = []
            boxes = []
            for bbox, text, prob in results:
                if prob > 0.5:  # Filter by confidence
                    words.append(text)
                    # EasyOCR bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # Convert to [x_min, y_min, x_max, y_max]
                    x_coords = [int(p[0]) for p in bbox]
                    y_coords = [int(p[1]) for p in bbox]
                    xmin, ymin = int(min(x_coords)), int(min(y_coords))
                    xmax, ymax = int(max(x_coords)), int(max(y_coords))
                    boxes.append([xmin, ymin, xmax, ymax])
            return words, boxes
        except Exception as e:
            logger.error(f"Error during OCR for {image_path}: {e}")
            return [], []

    def _parse_layoutlm_predictions(
        self, words: List[str], boxes: List[List[int]], predictions: List[int]
    ) -> Dict:
        """
        Parses LayoutLMv3 token classification predictions into structured invoice information.
        This method groups B- and I- tags to reconstruct entities.
        """
        document_info = {
            "invoice_number": "N/A",
            "date": "N/A",
            "due_date": "N/A",
            "vendor_name": "N/A",
            "vendor_address": "N/A",
            "customer_name": "N/A",
            "customer_address": "N/A",
            "subtotal": "N/A",
            "tax_amount": "N/A",
            "discount_amount": "N/A",
            "total_amount": "N/A",
            "currency": "N/A",
            "header": "N/A",
        }
        line_items = []

        current_entity_words = []
        current_entity_type = None

        # Temporary storage for line item components before grouping
        temp_line_item_components = {
            "ITEM_DESCRIPTION": [],
            "QUANTITY": [],
            "UNIT_PRICE": [],
            "LINE_TOTAL": [],
        }

        for i, pred_id in enumerate(predictions):
            word = (
                words[i] if i < len(words) else ""
            )  # Safeguard against index out of bounds
            label = self.id_to_label_map.get(pred_id, "O")

            if label.startswith("B-"):
                # If a new entity starts, save the previous one if it exists
                if current_entity_type:
                    extracted_text = " ".join(current_entity_words)
                    field_name = current_entity_type.lower()
                    if field_name in document_info:
                        document_info[field_name] = extracted_text
                    elif field_name in temp_line_item_components:
                        temp_line_item_components[field_name].append(extracted_text)

                current_entity_type = label[2:]  # Remove "B-" prefix
                current_entity_words = [word]
            elif label.startswith("I-") and current_entity_type == label[2:]:
                # Continue current entity
                current_entity_words.append(word)
            else:
                # Outside or different entity type, save current entity if exists
                if current_entity_type:
                    extracted_text = " ".join(current_entity_words)
                    field_name = current_entity_type.lower()
                    if field_name in document_info:
                        document_info[field_name] = extracted_text
                    elif field_name in temp_line_item_components:
                        temp_line_item_components[field_name].append(extracted_text)
                current_entity_type = None
                current_entity_words = []

        # After loop, save any remaining entity
        if current_entity_type:
            extracted_text = " ".join(current_entity_words)
            field_name = current_entity_type.lower()
            if field_name in document_info:
                document_info[field_name] = extracted_text
            elif field_name in temp_line_item_components:
                temp_line_item_components[field_name].append(extracted_text)

        # Heuristic grouping for line items (simplified example, can be improved with relation extraction)
        max_items = (
            max(len(v) for v in temp_line_item_components.values())
            if temp_line_item_components
            else 0
        )
        for i in range(max_items):
            item_desc = (
                temp_line_item_components["ITEM_DESCRIPTION"][i]
                if i < len(temp_line_item_components["ITEM_DESCRIPTION"])
                else "N/A"
            )
            qty = (
                temp_line_item_components["QUANTITY"][i]
                if i < len(temp_line_item_components["QUANTITY"])
                else "N/A"
            )
            unit_price = (
                temp_line_item_components["UNIT_PRICE"][i]
                if i < len(temp_line_item_components["UNIT_PRICE"])
                else "N/A"
            )
            line_total = (
                temp_line_item_components["LINE_TOTAL"][i]
                if i < len(temp_line_item_components["LINE_TOTAL"])
                else "N/A"
            )
            line_items.append(
                {
                    "item_description": item_desc,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "line_total": line_total,
                }
            )

        return {"document_info": document_info, "line_items": line_items}

    def extract_information_layoutlm(self, image_path: str) -> Dict:
        """
        Extracts structured information from an invoice image using LayoutLMv3.
        Includes OCR and model inference.
        """
        if not self.layoutlm_model or not self.layoutlm_processor:
            logger.warning(
                "LayoutLMv3 model not loaded. Falling back to regex extraction."
            )
            return self._fallback_extraction(image_path)

        words, boxes = self.extract_text_and_boxes(image_path)
        if not words:
            logger.warning(
                f"No text extracted from {image_path} by OCR. Cannot perform LayoutLMv3 extraction."
            )
            return {"document_info": {}, "line_items": []}

        try:
            image = Image.open(image_path).convert("RGB")

            # Prepare inputs for LayoutLMv3
            # The processor handles image resizing, normalization, tokenization, and bbox normalization
            encoding = self.layoutlm_processor(
                image,
                words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,  # Ensure consistent max_length as used in training
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                encoding = {
                    k: v.to(self.layoutlm_model.device) for k, v in encoding.items()
                }

            with torch.no_grad():
                outputs = self.layoutlm_model(**encoding)

            # Get predictions (logits) and convert to label IDs
            predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

            # If the output is a single token prediction (e.g., for very short inputs),
            # it might not be a list. Convert to list if it's an int.
            if isinstance(predictions, int):
                predictions = [predictions]

            # Truncate words and boxes to match the actual number of tokens processed (after padding/truncation)
            # This is important because the processor might truncate inputs.
            # Get the actual sequence length from attention_mask
            seq_len = encoding.attention_mask.sum().item()
            processed_words = (
                words[:seq_len] if len(words) >= seq_len else words
            )  # Take up to seq_len
            processed_boxes = boxes[:seq_len] if len(boxes) >= seq_len else boxes
            processed_predictions = (
                predictions[:seq_len]
                if len(predictions) >= seq_len
                else processed_predictions
            )  # Fixed bug: use predictions[:seq_len]

            # Ensure words, boxes, predictions are aligned and not empty
            if not processed_words or not processed_boxes or not processed_predictions:
                logger.warning(
                    f"Empty or malformed processed data for {image_path} after LayoutLMv3."
                )
                return {"document_info": {}, "line_items": []}

            return self._parse_layoutlm_predictions(
                processed_words, processed_boxes, processed_predictions
            )

        except Exception as e:
            logger.error(f"Error during LayoutLMv3 extraction for {image_path}: {e}")
            return self._fallback_extraction(image_path)

    def classify_psc(self, item_description: str, psc_data: Dict) -> Dict:
        """
        Classifies an item description into a PSC using the trained PSC model or fallback.
        """
        # Ensure the PSC learner and its ID to label map are loaded
        if self.psc_learner and self.psc_id_to_label:
            try:
                # FastAI learner.predict returns (predicted_class_string, predicted_class_idx_tensor, probabilities_tensor)
                # Ensure the input text is a string
                pred_class_str, pred_idx_tensor, probs_tensor = (
                    self.psc_learner.predict(str(item_description))
                )

                predicted_psc_code = str(pred_class_str)  # This is the PSC string code
                confidence = float(
                    torch.max(probs_tensor)
                )  # Max probability for the predicted class

                # Get full PSC details from the PSC mapping provided by PSCClassifierTrainer
                psc_mapping = psc_data.get("psc_mapping", {})
                psc_details = psc_mapping.get(predicted_psc_code, {})

                probabilities = {}
                # Ensure the probabilities tensor aligns with the PSC class IDs
                if self.psc_id_to_label and probs_tensor.shape[-1] == len(
                    self.psc_id_to_label
                ):
                    for i, prob in enumerate(probs_tensor.tolist()):
                        probabilities[self.psc_id_to_label.get(i, f"class_{i}")] = (
                            float(prob)
                        )

                return {
                    "predicted_psc": predicted_psc_code,
                    "confidence": confidence,
                    "shortName": psc_details.get("shortName", "N/A"),
                    "spendCategoryTitle": psc_details.get("spendCategoryTitle", "N/A"),
                    "portfolioGroup": psc_details.get("portfolioGroup", "N/A"),
                    "probabilities": probabilities,
                }
            except Exception as e:
                logger.error(
                    f"PSC model prediction failed for '{item_description}': {e}. Falling back to keyword matching."
                )
                # Fallback to keyword matching if ML model prediction fails
                return self._keyword_psc_lookup(item_description, psc_data)
        else:
            logger.warning(
                "PSC classifier model not loaded. Using keyword matching for PSC classification."
            )
            return self._keyword_psc_lookup(item_description, psc_data)

    def _keyword_psc_lookup(self, description: str, psc_data: Dict) -> Dict:
        """Fallback for PSC classification using keyword matching from DataPreparationUtils."""
        # Use DataPreparationUtils's method for keyword-based lookup
        # Need an instance of DataPreparationUtils here.
        # This will load psc_data internally if not already loaded.
        from data_preparation_utils import DataPreparationUtils

        data_utils_instance = DataPreparationUtils()

        # data_utils_instance.psc_data might be None if not previously loaded/set.
        # Ensure it's populated if get_psc_by_description needs it.
        if not data_utils_instance.psc_data:
            data_utils_instance.psc_data = psc_data

        result = data_utils_instance.get_psc_by_description(description)
        if result:
            return {
                "predicted_psc": result.get("psc", "UNKNOWN"),
                "confidence": 0.5,  # Assign a default confidence for keyword match
                "shortName": result.get("shortName", "Unknown"),
                "spendCategoryTitle": result.get("spendCategoryTitle", "Unknown"),
                "portfolioGroup": result.get("portfolioGroup", "Unknown"),
                "probabilities": {},
            }
        else:
            return {
                "predicted_psc": "UNKNOWN",
                "confidence": 0.0,
                "shortName": "Unknown",
                "spendCategoryTitle": "Unknown",
                "portfolioGroup": "Unknown",
                "probabilities": {},
            }

    def _fallback_extraction(self, image_path: str) -> Dict:
        """
        Provides a basic regex-based extraction if LayoutLMv3 fails or is not loaded.
        This is a highly simplified fallback.
        """
        logger.warning(
            f"Performing fallback (regex-based) extraction for {image_path}."
        )
        words, _ = self.extract_text_and_boxes(image_path)
        full_text = " ".join(words)

        doc_info = {
            "invoice_number": "N/A",
            "date": "N/A",
            "total_amount": "N/A",
            "vendor_name": "N/A",
            "currency": "N/A",
            "header": "N/A",
        }
        line_items = []

        # Simple regex patterns for common fields
        invoice_num_match = re.search(
            r"(invoice|bill|receipt)\s*#?\s*([\w-]+)", full_text, re.IGNORECASE
        )
        if invoice_num_match:
            doc_info["invoice_number"] = invoice_num_match.group(2).strip()

        date_match = re.search(
            r"date[:\s]*(\d{2}[/-]\d{2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2})",
            full_text,
            re.IGNORECASE,
        )
        if date_match:
            doc_info["date"] = date_match.group(1).strip()

        total_match = re.search(
            r"(total|grand total|amount due)[:\s]*([$€£¥]?\s*\d[\d,\.]*)",
            full_text,
            re.IGNORECASE,
        )
        if total_match:
            doc_info["total_amount"] = total_match.group(2).strip()
            if "$" in total_match.group(2):
                doc_info["currency"] = "$"
            elif "€" in total_match.group(2):
                doc_info["currency"] = "€"
            elif "£" in total_match.group(2):
                doc_info["currency"] = "£"
            elif "¥" in total_match.group(2):
                doc_info["currency"] = "¥"

        # Very simplistic line item extraction (might not work well)
        # Just takes some generic text as a line item description
        if len(words) > 5:  # If there's enough text
            line_items.append(
                {
                    "item_description": " ".join(words[5 : min(len(words), 15)]),
                    "quantity": "1",
                    "unit_price": "N/A",
                    "line_total": "N/A",
                }
            )

        return {"document_info": doc_info, "line_items": line_items}

    def process_document(self, image_path: str, psc_data: Dict) -> Dict:
        """
        Processes a single invoice document (OCR, LayoutLMv3/regex extraction, PSC classification).
        """
        processed_at = datetime.now().isoformat()

        if self.use_fallback:
            logger.info(f"Processing {image_path} in FALLBACK mode.")
            extraction_result = self._fallback_extraction(image_path)
            extraction_method = "Fallback (Regex/OCR Only)"
        else:
            logger.info(f"Processing {image_path} with LayoutLMv3.")
            extraction_result = self.extract_information_layoutlm(image_path)
            extraction_method = "LayoutLMv3"

        final_line_items = []
        for item in extraction_result.get("line_items", []):
            item_desc = item.get("item_description", "N/A")
            # Classify PSC for each item description
            psc_classification = self.classify_psc(item_desc, psc_data)
            item["psc_classification"] = psc_classification
            final_line_items.append(item)

        return {
            "document_info": extraction_result.get("document_info", {}),
            "line_items": final_line_items,
            "processing_metadata": {
                "processed_at": processed_at,
                "extraction_method": extraction_method,
                "total_items_extracted": len(final_line_items),
            },
        }

    def batch_process(
        self, image_paths: List[str], output_dir: str, psc_data: Dict
    ) -> List[Dict]:
        """Processes a list of invoice documents in batch."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        logger.info(f"Starting batch processing of {len(image_paths)} documents...")

        for i, image_path_str in enumerate(image_paths):
            image_path = Path(image_path_str)
            try:
                result = self.process_document(str(image_path), psc_data)
                results.append(result)

                # Save individual result
                output_file = output_path / f"processed_document_{image_path.stem}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                logger.info(f"Processed {i+1}/{len(image_paths)}: {image_path.name}")

            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                results.append({"error": str(e), "image_path": image_path.name})

        # Save batch results
        batch_output_file = output_path / "batch_results.json"
        with open(batch_output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Batch processing completed. Results saved to {batch_output_file}."
        )
        return results


def create_inference_engine() -> IDPInferenceEngine:
    """Factory function to create and initialize the inference engine."""
    return IDPInferenceEngine()


if __name__ == "__main__":
    # Ensure 'models' directory exists for saving/loading
    Path("models").mkdir(exist_ok=True, parents=True)

    # Create an instance of DataPreparationUtils to load PSC data
    from data_preparation_utils import DataPreparationUtils

    data_utils = DataPreparationUtils()
    psc_data = data_utils.load_psc_data()

    # Example usage: Process a single document
    engine = create_inference_engine()

    sample_image = "data/kaggle_invoices/images/batch1-0001.jpg"  # Update with a valid path to an invoice image
    if Path(sample_image).exists():
        logger.info(f"Attempting to process single sample image: {sample_image}")
        result = engine.process_document(sample_image, psc_data)
        logger.info("Processing Result:")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        logger.warning(
            f"Sample image {sample_image} not found. Please provide a valid image path "
            "or ensure your 'data' directory is correctly set up."
        )
        logger.info("Proceeding with a dummy demo for PSC classification only.")
        # Dummy demo for PSC classification if no image is found
        sample_text_for_psc = (
            "Purchase of ergonomic office chairs and desks for new employees"
        )
        psc_result = engine.classify_psc(sample_text_for_psc, psc_data)
        logger.info(f"\nPSC Classification Demo for: '{sample_text_for_psc}'")
        logger.info(json.dumps(psc_result, indent=2, ensure_ascii=False))

    # Example usage: Batch process documents (optional, uncomment to run)
    batch_input_folder = "data/kaggle_invoices/images"  # Folder containing images
    batch_output_folder = "output/batch_processed"
    if Path(batch_input_folder).exists():
        image_files = [
            str(f) for f in Path(batch_input_folder).glob("*.jpg")
        ]  # Or *.png, etc.
        if image_files:
            logger.info(
                f"\nAttempting to batch process {len(image_files)} images from {batch_input_folder}"
            )
            engine.batch_process(image_files, batch_output_folder, psc_data)
        else:
            logger.warning(
                f"No image files found in {batch_input_folder} for batch processing."
            )
    else:
        logger.warning(
            f"Batch input folder {batch_input_folder} not found. Skipping batch processing demo."
        )
