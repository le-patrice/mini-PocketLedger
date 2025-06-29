import torch #type: ignore
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import cv2 #type: ignore
import easyocr #type: ignore
from datetime import datetime
import re
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """Data class for extracted entities with bounding boxes."""
    text: str
    entity_type: str
    bbox: List[int]
    confidence: float = 0.0

@dataclass
class LineItem:
    """Data class for invoice line items."""
    description: str
    quantity: str = "N/A"
    unit_price: str = "N/A"
    line_total: str = "N/A"
    category: Optional[Dict[str, Any]] = None

@dataclass
class DocumentInfo:
    """Data class for document information."""
    invoice_number: str = "N/A"
    date: str = "N/A"
    vendor_name: str = "N/A"
    vendor_address: str = "N/A"
    customer_name: str = "N/A"
    customer_address: str = "N/A"
    total_amount: str = "N/A"
    currency: str = "N/A"
    subtotal: str = "N/A"
    tax_amount: str = "N/A"
    discount_amount: str = "N/A"

class ModelLoader:
    """Handles dynamic loading of ML models with graceful fallbacks."""
    
    def __init__(self):
        self.layoutlm_processor = None
        self.layoutlm_model = None
        self.item_categorizer = None
        self._load_dependencies()
    
    def _load_dependencies(self):
        """Dynamically import ML dependencies."""
        try:
            from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification #type: ignore
            self.LayoutLMv3Processor = LayoutLMv3Processor
            self.LayoutLMv3ForTokenClassification = LayoutLMv3ForTokenClassification
            logger.info("LayoutLMv3 components imported successfully")
        except ImportError:
            logger.warning("LayoutLMv3 components not available")
            self.LayoutLMv3Processor = None
            self.LayoutLMv3ForTokenClassification = None
        
        try:
            from item_categorizer_trainer import ItemCategorizerTrainer
            self.ItemCategorizerTrainer = ItemCategorizerTrainer
            logger.info("ItemCategorizerTrainer imported successfully")
        except ImportError:
            logger.warning("ItemCategorizerTrainer not available")
            self.ItemCategorizerTrainer = None
    
    def load_layoutlm(self, model_path: Path, device: str) -> bool:
        """Load LayoutLMv3 model."""
        if not self.LayoutLMv3Processor or not self.LayoutLMv3ForTokenClassification:
            return False
        
        if not model_path.exists():
            logger.warning(f"LayoutLMv3 model not found at {model_path}")
            return False
        
        try:
            self.layoutlm_processor = self.LayoutLMv3Processor.from_pretrained(
                str(model_path), apply_ocr=False
            )
            self.layoutlm_model = self.LayoutLMv3ForTokenClassification.from_pretrained(
                str(model_path)
            ).to(device)
            self.layoutlm_model.eval()
            logger.info(f"LayoutLMv3 loaded successfully on {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3: {e}")
            return False
    
    def load_item_categorizer(self, model_path: Path, device: str) -> bool:
        """Load item categorizer model."""
        if not self.ItemCategorizerTrainer:
            return False
        
        if not model_path.exists():
            logger.warning(f"Item categorizer model not found at {model_path}")
            return False
        
        try:
            self.item_categorizer = self.ItemCategorizerTrainer.load_model(str(model_path))
            if hasattr(self.item_categorizer, 'model') and self.item_categorizer.model:
                self.item_categorizer.model.to(device).eval()
            logger.info(f"Item categorizer loaded successfully on {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load item categorizer: {e}")
            return False

class TextProcessor:
    """Handles text processing and cleaning operations."""
    
    @staticmethod
    def clean_and_convert_float(text: str) -> float:
        """Clean text and convert to float."""
        try:
            cleaned = re.sub(r'[$,€£¥\s]', '', str(text).strip())
            return float(cleaned) if cleaned else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def extract_currency(text: str) -> str:
        """Extract currency symbol from text."""
        match = re.search(r'([$€£¥])', str(text))
        return match.group(1) if match else "$"
    
    @staticmethod
    def format_price(value: float, currency: str = "$") -> str:
        """Format price with currency."""
        return f"{currency}{value:.2f}" if value > 0 else "N/A"

class OCRProcessor:
    """Handles OCR operations."""
    
    def __init__(self, use_gpu: bool = None):
        use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        logger.info(f"OCR initialized (GPU: {use_gpu})")
    
    def extract_text_and_boxes(self, image_path: str, confidence_threshold: float = 0.3) -> Tuple[List[str], List[List[int]], int, int]:
        """Extract text and bounding boxes from image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            height, width = image.shape[:2]
            results = self.reader.readtext(image)
            
            words, boxes = [], []
            for bbox_raw, text, confidence in results:
                if confidence > confidence_threshold:
                    words.append(text)
                    # Convert 4-point bbox to 2-point
                    x_coords = [int(p[0]) for p in bbox_raw]
                    y_coords = [int(p[1]) for p in bbox_raw]
                    boxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
            
            return words, boxes, width, height
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return [], [], 0, 0

class FallbackExtractor:
    """Handles regex-based fallback extraction."""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
    
    def extract(self, words: List[str]) -> Dict[str, Any]:
        """Extract information using regex patterns."""
        full_text = " ".join(words)
        
        doc_info = DocumentInfo()
        line_items = []
        
        # Extract invoice number
        inv_match = re.search(r"(invoice|bill|receipt|#|no)[:\s]*(\w[\w-]{3,})", full_text, re.IGNORECASE)
        if inv_match:
            doc_info.invoice_number = inv_match.group(2).strip().upper()
        
        # Extract date
        date_match = re.search(r"date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})", full_text, re.IGNORECASE)
        if date_match:
            doc_info.date = date_match.group(1).strip()
        
        # Extract total
        total_match = re.search(r"(total|grand total|amount due)[:\s]*([$€£¥]?\s*\d[\d,\.]*\.?\d*)", full_text, re.IGNORECASE)
        if total_match:
            total_text = total_match.group(2).strip()
            doc_info.total_amount = total_text
            doc_info.currency = self.text_processor.extract_currency(total_text)
        
        # Extract line items (simplified pattern)
        item_pattern = re.compile(r"(.+?)\s+(\d+)\s+([$€£¥]?\d[\d,\.]*)\s+([$€£¥]?\d[\d,\.]*)")
        for line in full_text.split('\n'):
            match = item_pattern.search(line)
            if match:
                try:
                    description = match.group(1).strip()
                    quantity = match.group(2).strip()
                    unit_price_raw = match.group(3).strip()
                    line_total_raw = match.group(4).strip()
                    
                    qty_val = self.text_processor.clean_and_convert_float(quantity)
                    total_val = self.text_processor.clean_and_convert_float(line_total_raw)
                    unit_val = self.text_processor.clean_and_convert_float(unit_price_raw)
                    
                    # Calculate unit price if missing
                    if unit_val == 0.0 and qty_val > 0 and total_val > 0:
                        unit_val = total_val / qty_val
                    
                    line_items.append(LineItem(
                        description=description,
                        quantity=quantity,
                        unit_price=self.text_processor.format_price(unit_val, doc_info.currency),
                        line_total=line_total_raw
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing line item '{line}': {e}")
        
        # Fallback: create generic item if no structured items found
        if not line_items and words:
            line_items.append(LineItem(description=" ".join(words[:20])))
        
        return {
            "document_info": doc_info.__dict__,
            "line_items": [item.__dict__ for item in line_items]
        }

class LayoutLMExtractor:
    """Handles LayoutLMv3-based extraction."""
    
    def __init__(self, model_loader: ModelLoader, text_processor: TextProcessor, device: str):
        self.model_loader = model_loader
        self.text_processor = text_processor
        self.device = device
    
    def extract(self, image_path: str, words: List[str], boxes: List[List[int]]) -> Dict[str, Any]:
        """Extract information using LayoutLMv3."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            encoding = self.model_loader.layoutlm_processor(
                image, words, boxes=boxes,
                return_tensors="pt", truncation=True,
                padding="max_length", max_length=512
            )
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = self.model_loader.layoutlm_model(**encoding)
            
            predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            # Process predictions
            actual_len = encoding.attention_mask.sum().item()
            processed_words = words[:actual_len]
            processed_boxes = boxes[:actual_len]
            processed_predictions = predictions[:actual_len]
            
            return self._parse_predictions(processed_words, processed_boxes, processed_predictions)
        
        except Exception as e:
            logger.error(f"LayoutLMv3 extraction failed: {e}")
            raise
    
    def _parse_predictions(self, words: List[str], boxes: List[List[int]], predictions: List[int]) -> Dict[str, Any]:
        """Parse LayoutLMv3 predictions into structured data."""
        # Get label mapping
        id2label = getattr(self.model_loader.layoutlm_model.config, 'id2label', {})
        
        # Group entities by type
        entities = []
        for word, bbox, pred_id in zip(words, boxes, predictions):
            label = id2label.get(pred_id, f"LABEL_{pred_id}")
            if label != "O":  # Skip "Outside" labels
                entities.append(ExtractedEntity(
                    text=word, entity_type=label.replace("B-", "").replace("I-", ""),
                    bbox=bbox, confidence=1.0
                ))
        
        # Group into document info and line items
        doc_info = DocumentInfo()
        line_item_candidates = []
        
        for entity in entities:
            entity_type = entity.entity_type.lower()
            
            # Map to document fields
            if "invoice" in entity_type or "number" in entity_type:
                doc_info.invoice_number = entity.text
            elif "date" in entity_type:
                doc_info.date = entity.text
            elif "vendor" in entity_type or "supplier" in entity_type:
                doc_info.vendor_name = entity.text
            elif "customer" in entity_type or "buyer" in entity_type:
                doc_info.customer_name = entity.text
            elif "total" in entity_type or "amount" in entity_type:
                doc_info.total_amount = entity.text
                doc_info.currency = self.text_processor.extract_currency(entity.text)
            elif any(x in entity_type for x in ["item", "description", "quantity", "price", "line"]):
                line_item_candidates.append(entity)
        
        # Group line items spatially
        line_items = self._group_line_items(line_item_candidates)
        
        return {
            "document_info": doc_info.__dict__,
            "line_items": [item.__dict__ for item in line_items]
        }
    
    def _group_line_items(self, candidates: List[ExtractedEntity], y_tolerance: int = 15) -> List[LineItem]:
        """Group line item components spatially."""
        if not candidates:
            return []
        
        # Sort by Y coordinate, then X
        candidates.sort(key=lambda x: (x.bbox[1], x.bbox[0]))
        
        grouped_items = []
        current_line = {}
        current_max_y = -1
        
        for candidate in candidates:
            y_min, y_max = candidate.bbox[1], candidate.bbox[3]
            field_type = candidate.entity_type.lower()
            
            # Check if starting new line
            if ("description" in field_type or "item" in field_type) or \
               (current_line and y_min > current_max_y + y_tolerance):
                
                # Finalize previous line
                if current_line:
                    grouped_items.append(self._finalize_line_item(current_line))
                
                # Start new line
                current_line = {"texts": {}, "min_y": y_min, "max_y": y_max}
                current_max_y = y_max
            
            # Add to current line
            if not current_line:
                current_line = {"texts": {}, "min_y": y_min, "max_y": y_max}
            
            # Map field types
            if "description" in field_type or "item" in field_type:
                key = "description"
            elif "quantity" in field_type or "qty" in field_type:
                key = "quantity"
            elif "price" in field_type:
                key = "unit_price"
            elif "total" in field_type or "amount" in field_type:
                key = "line_total"
            else:
                key = "description"  # Default
            
            current_line["texts"][key] = current_line["texts"].get(key, "") + " " + candidate.text
            current_line["max_y"] = max(current_line["max_y"], y_max)
            current_max_y = current_line["max_y"]
        
        # Finalize last line
        if current_line:
            grouped_items.append(self._finalize_line_item(current_line))
        
        return grouped_items
    
    def _finalize_line_item(self, line_data: Dict) -> LineItem:
        """Convert grouped line data to LineItem."""
        texts = line_data["texts"]
        
        description = texts.get("description", "").strip()
        quantity = texts.get("quantity", "").strip()
        unit_price = texts.get("unit_price", "").strip()
        line_total = texts.get("line_total", "").strip()
        
        # Calculate unit price if missing
        qty_val = self.text_processor.clean_and_convert_float(quantity)
        total_val = self.text_processor.clean_and_convert_float(line_total)
        unit_val = self.text_processor.clean_and_convert_float(unit_price)
        
        if unit_val == 0.0 and qty_val > 0 and total_val > 0:
            unit_val = total_val / qty_val
            unit_price = self.text_processor.format_price(unit_val)
        elif unit_val > 0:
            unit_price = self.text_processor.format_price(unit_val)
        else:
            unit_price = "N/A"
        
        return LineItem(
            description=description or "N/A",
            quantity=quantity or "N/A",
            unit_price=unit_price,
            line_total=line_total or "N/A"
        )

class IDPInferenceEngine:
    """Optimized Intelligent Document Processing inference engine."""
    
    def __init__(
        self,
        layoutlm_model_path: str = "models/layoutlmv3_invoice_extractor/fine_tuned_layoutlmv3",
        item_categorizer_model_path: str = "models/unspsc_item_classifier",
        use_fallback: bool = False
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fallback = use_fallback
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.ocr_processor = OCRProcessor()
        self.model_loader = ModelLoader()
        
        # Initialize extractors
        self.fallback_extractor = FallbackExtractor(self.text_processor)
        self.layoutlm_extractor = None
        
        # Load models
        if not use_fallback:
            self._load_models(layoutlm_model_path, item_categorizer_model_path)
        
        logger.info(f"IDPInferenceEngine initialized on {self.device}")
    
    def _load_models(self, layoutlm_path: str, categorizer_path: str):
        """Load ML models."""
        # Load LayoutLMv3
        if self.model_loader.load_layoutlm(Path(layoutlm_path), self.device):
            self.layoutlm_extractor = LayoutLMExtractor(
                self.model_loader, self.text_processor, self.device
            )
        else:
            logger.warning("LayoutLMv3 not loaded, using fallback")
            self.use_fallback = True
        
        # Load item categorizer
        self.model_loader.load_item_categorizer(Path(categorizer_path), self.device)
    
    def classify_item_category(self, description: str) -> Dict[str, Any]:
        """Classify item into UNSPSC category."""
        if not self.model_loader.item_categorizer:
            return {
                "predicted_category": "UNCLASSIFIED",
                "confidence": 0.0,
                "Segment Name": "N/A",
                "Family Name": "N/A",
                "Class Name": "N/A",
                "Commodity Name": "N/A"
            }
        
        try:
            result = self.model_loader.item_categorizer.predict_category(description)
            return result if "error" not in result else {
                "predicted_category": "ERROR_CLASSIFYING",
                "confidence": 0.0,
                "Segment Name": "N/A",
                "Family Name": "N/A",
                "Class Name": "N/A",
                "Commodity Name": "N/A"
            }
        except Exception as e:
            logger.error(f"Item classification failed for '{description}': {e}")
            return {
                "predicted_category": "ERROR_CLASSIFYING",
                "confidence": 0.0,
                "Segment Name": "N/A",
                "Family Name": "N/A",
                "Class Name": "N/A",
                "Commodity Name": "N/A"
            }
    
    def extract_information(self, image_path: str = None, demo_data: Dict = None) -> Dict[str, Any]:
        """Extract information from document."""
        # Handle demo data
        if demo_data:
            logger.info("Using demo data")
            return {
                "document_info": {
                    "invoice_number": "INV-2024-00123",
                    "date": "2024-06-27",
                    "vendor_name": "Tech & Fresh Supplies Inc.",
                    "vendor_address": "123 Main St, Anytown, CA 90210",
                    "customer_name": "Your Company Ltd.",
                    "total_amount": "$755.98",
                    "currency": "$"
                },
                "line_items": [
                    {"item_description": "Wireless Mouse Model X200", "quantity": "5", "unit_price": "$25.00", "line_total": "$125.00"},
                    {"item_description": "Mechanical Keyboard RGB", "quantity": "3", "unit_price": "$99.99", "line_total": "$299.97"},
                    {"item_description": "Organic Mixed Greens 5lb", "quantity": "10", "unit_price": "$8.50", "line_total": "$85.00"},
                    {"item_description": "A4 Copy Paper Box", "quantity": "15", "unit_price": "$16.40", "line_total": "$246.00"}
                ]
            }
        
        if not image_path:
            raise ValueError("Either image_path or demo_data must be provided")
        
        # Extract text and boxes
        words, boxes, width, height = self.ocr_processor.extract_text_and_boxes(image_path)
        if not words:
            logger.warning(f"No text extracted from {image_path}")
            return {"document_info": {}, "line_items": []}
        
        # Choose extraction method
        if self.use_fallback or not self.layoutlm_extractor:
            logger.info("Using fallback extraction")
            return self.fallback_extractor.extract(words)
        else:
            try:
                logger.info("Using LayoutLMv3 extraction")
                return self.layoutlm_extractor.extract(image_path, words, boxes)
            except Exception as e:
                logger.error(f"LayoutLMv3 failed, falling back: {e}")
                return self.fallback_extractor.extract(words)
    
    def process_document(self, image_path: str = None, demo_data: Dict = None) -> Dict[str, Any]:
        """Main document processing function."""
        start_time = datetime.now()
        
        # Extract information
        extraction_result = self.extract_information(image_path, demo_data)
        
        # Add categories to line items
        final_line_items = []
        for item in extraction_result.get("line_items", []):
            description = item.get("item_description", "")
            category = self.classify_item_category(description)
            
            final_item = {
                "item_description": item.get("item_description", "N/A"),
                "quantity": item.get("quantity", "N/A"),
                "unit_price": item.get("unit_price", "N/A"),
                "line_total": item.get("line_total", "N/A"),
                "category_classification": category
            }
            final_line_items.append(final_item)
        
        # Determine extraction method
        if demo_data:
            method = "Demo Data"
        elif self.use_fallback or not self.layoutlm_extractor:
            method = "Fallback (OCR + Regex)"
        else:
            method = "LayoutLMv3"
        
        return {
            "document_info": extraction_result.get("document_info", {}),
            "line_items": final_line_items,
            "processing_metadata": {
                "processed_at": start_time.isoformat(),
                "extraction_method": method,
                "total_items_extracted": len(final_line_items),
                "processing_time_ms": int((datetime.now() - start_time).total_seconds() * 1000)
            }
        }
    
    def batch_process(self, image_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        logger.info(f"Starting batch processing of {len(image_paths)} documents")
        
        for i, image_path_str in enumerate(image_paths):
            image_path = Path(image_path_str)
            
            if not image_path.exists():
                error_result = {"error": f"File not found: {image_path.name}", "image_path": str(image_path)}
                results.append(error_result)
                logger.warning(f"File not found: {image_path}")
                continue
            
            try:
                result = self.process_document(image_path=str(image_path))
                results.append(result)
                
                # Save individual result
                output_file = output_path / f"processed_{image_path.stem}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Processed {i+1}/{len(image_paths)}: {image_path.name}")
                
            except Exception as e:
                error_result = {"error": str(e), "image_path": str(image_path)}
                results.append(error_result)
                logger.error(f"Error processing {image_path.name}: {e}")
        
        # Save batch results
        batch_file = output_path / "batch_results.json"
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed. Results saved to {batch_file}")
        return results


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = IDPInferenceEngine()
    
    # Process single document
    result = engine.process_document(demo_data={"sample": "data"})
    print(json.dumps(result, indent=2))
    
    # Process actual image (uncomment when you have an image)
    # result = engine.process_document(image_path="path/to/invoice.jpg")
    # print(json.dumps(result, indent=2))
    
    # Batch processing (uncomment when you have images)
    # image_paths = ["invoice1.jpg", "invoice2.jpg"]
    # results = engine.batch_process(image_paths, "output_directory")