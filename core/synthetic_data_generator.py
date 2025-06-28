import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreparationUtils:
    """Optimized UNSPSC data handler."""
    
    def __init__(self):
        self.unspsc_url = "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/unspsc-codes.csv"
        self.local_unspsc_path = "data/unspsc-codes.csv"
        
    def load_unspsc_data(self) -> Dict:
        """Load and process UNSPSC data efficiently."""
        try:
            # Download if not exists locally
            if not Path(self.local_unspsc_path).exists():
                Path(self.local_unspsc_path).parent.mkdir(exist_ok=True, parents=True)
                response = requests.get(self.unspsc_url, timeout=30)
                response.raise_for_status()
                with open(self.local_unspsc_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Downloaded UNSPSC data to {self.local_unspsc_path}")
            
            # Load CSV with correct column names
            df = pd.read_csv(self.local_unspsc_path)
            
            # Create commodity mapping for efficient random selection
            commodity_data = []
            for _, row in df.iterrows():
                commodity_data.append({
                    "segment_name": str(row.get("Segment Name", "")),
                    "family_name": str(row.get("Family Name", "")),
                    "class_name": str(row.get("Class Name", "")),
                    "commodity_name": str(row.get("Commodity Name", "")),
                    "segment_code": str(row.get("Segment", "")),
                    "family_code": str(row.get("Family", "")),
                    "class_code": str(row.get("Class", "")),
                    "commodity_code": str(row.get("Commodity", ""))
                })
            
            logger.info(f"Loaded {len(commodity_data)} UNSPSC commodities")
            return {"commodities": commodity_data, "processed_df": df}
            
        except Exception as e:
            logger.error(f"Failed to load UNSPSC data: {e}")
            return {"commodities": [], "processed_df": pd.DataFrame()}
    
    def _normalize_bbox(self, bbox: List[int], img_width: int, img_height: int) -> List[int]:
        """Normalize bounding box to 0-1000 scale."""
        return [
            int((bbox[0] / img_width) * 1000),
            int((bbox[1] / img_height) * 1000),
            int((bbox[2] / img_width) * 1000),
            int((bbox[3] / img_height) * 1000)
        ]


class SyntheticInvoiceGenerator:
    """Improved synthetic invoice generator with proper layout management."""

    def __init__(self, output_dir: str = "data/synthetic_invoices"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)

        self.fake = Faker()
        self.data_utils = DataPreparationUtils()
        self.unspsc_data = self.data_utils.load_unspsc_data()
        self.commodities = self.unspsc_data.get("commodities", [])
        
        # Setup fonts
        self._setup_fonts()
        
        # Improved layouts with better spacing
        self.layouts = {
            "standard": {
                "width": 800, 
                "height": 1100, 
                "margin": 60, 
                "line_height": 25,
                "section_spacing": 30,
                "header_height": 80,
                "footer_height": 120
            },
            "compact": {
                "width": 750, 
                "height": 1000, 
                "margin": 50, 
                "line_height": 22,
                "section_spacing": 25,
                "header_height": 70,
                "footer_height": 100
            },
            "receipt": {
                "width": 600, 
                "height": 900, 
                "margin": 40, 
                "line_height": 20,
                "section_spacing": 20,
                "header_height": 60,
                "footer_height": 80
            }
        }
        
        # NER tags for LayoutLMv3
        self.ner_tags = [
            "O", "B-INVOICE_NUM", "I-INVOICE_NUM", "B-DATE", "I-DATE", "B-DUE_DATE", "I-DUE_DATE",
            "B-VENDOR_NAME", "I-VENDOR_NAME", "B-VENDOR_ADDRESS", "I-VENDOR_ADDRESS",
            "B-CUSTOMER_NAME", "I-CUSTOMER_NAME", "B-CUSTOMER_ADDRESS", "I-CUSTOMER_ADDRESS",
            "B-ITEM_DESCRIPTION", "I-ITEM_DESCRIPTION", "B-QUANTITY", "I-QUANTITY",
            "B-UNIT_PRICE", "I-UNIT_PRICE", "B-LINE_TOTAL", "I-LINE_TOTAL",
            "B-SUBTOTAL", "I-SUBTOTAL", "B-TAX_AMOUNT", "I-TAX_AMOUNT",
            "B-TOTAL_AMOUNT", "I-TOTAL_AMOUNT", "B-CURRENCY", "I-CURRENCY",
            "B-HEADER", "I-HEADER"
        ]
        self.tag_to_id = {tag: i for i, tag in enumerate(self.ner_tags)}

    def _setup_fonts(self):
        """Setup fonts with fallback."""
        try:
            font_path = Path("data/fonts/arial.ttf")
            if not font_path.exists():
                # Try to download font
                font_url = "https://github.com/googlefonts/RobotoMono/raw/main/fonts/ttf/RobotoMono-Regular.ttf"
                font_path.parent.mkdir(exist_ok=True, parents=True)
                response = requests.get(font_url, timeout=10)
                response.raise_for_status()
                with open(font_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded font to {font_path}")
            
            self.font_small = ImageFont.truetype(str(font_path), 12)
            self.font_medium = ImageFont.truetype(str(font_path), 16)
            self.font_large = ImageFont.truetype(str(font_path), 22)
            self.font_xlarge = ImageFont.truetype(str(font_path), 28)
        except Exception as e:
            logger.warning(f"Using default font: {e}")
            self.font_small = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_large = ImageFont.load_default()
            self.font_xlarge = ImageFont.load_default()

    def _get_random_commodity(self) -> Dict:
        """Get random UNSPSC commodity efficiently."""
        if not self.commodities:
            return {
                "segment_name": "Generic Products",
                "family_name": "Generic Family",
                "class_name": "Generic Class",
                "commodity_name": self.fake.word().title(),
                "segment_code": "10000000",
                "family_code": "10100000",
                "class_code": "10101500",
                "commodity_code": "10101501"
            }
        return random.choice(self.commodities)

    def _generate_line_item(self) -> Dict:
        """Generate realistic line item with UNSPSC data."""
        commodity = self._get_random_commodity()
        
        # Create realistic description - limit length to prevent overflow
        description = commodity["commodity_name"][:40]  # Limit description length
        if random.random() < 0.4:  # Add brand/model
            brand = self.fake.company()[:15]  # Limit brand length
            description = f"{brand} {description}"
        if random.random() < 0.3:  # Add specifications
            description = f"{description} {self.fake.bothify(text='##??')}"

        return {
            "description": description[:60],  # Final length limit
            "quantity": random.randint(1, 10),
            "unit_price": round(random.uniform(10.0, 500.0), 2),
            "unspsc_details": commodity
        }

    def _get_text_dimensions(self, text: str, font) -> Tuple[int, int]:
        """Get text dimensions for proper spacing."""
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _add_text_with_annotation(self, draw, text: str, tag: str, x: int, y: int, 
                                 font, width: int, height: int, annotations: Dict,
                                 max_width: Optional[int] = None) -> int:
        """Add text with proper word wrapping and annotations."""
        if not text.strip():
            return 0
            
        # Word wrap if max_width is specified
        if max_width:
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                if font.getlength(test_line) <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)  # Single word too long
            
            if current_line:
                lines.append(' '.join(current_line))
        else:
            lines = [text]
        
        total_height = 0
        for line_idx, line in enumerate(lines):
            words = line.split()
            current_x = x
            
            for word_idx, word in enumerate(words):
                # Calculate word bbox
                word_bbox = draw.textbbox((current_x, y + total_height), word, font=font)
                normalized_bbox = self.data_utils._normalize_bbox(
                    list(word_bbox), width, height
                )
                
                # Store annotations
                annotations["words"].append(word)
                annotations["boxes"].append(normalized_bbox)
                
                # Determine NER tag
                if line_idx == 0 and word_idx == 0:
                    ner_tag = self.tag_to_id.get(f"B-{tag}", self.tag_to_id["O"])
                elif tag in self.tag_to_id:
                    ner_tag = self.tag_to_id.get(f"I-{tag}", self.tag_to_id["O"])
                else:
                    ner_tag = self.tag_to_id["O"]
                
                annotations["ner_tags"].append(ner_tag)
                
                current_x += int(font.getlength(word + " "))
            
            # Draw the line
            draw.text((x, y + total_height), line, fill="black", font=font)
            line_height = self._get_text_dimensions(line, font)[1]
            total_height += line_height + 5  # Add small spacing between lines
        
        return total_height

    def _check_space_available(self, current_y: int, required_height: int, layout: Dict) -> bool:
        """Check if there's enough space for content."""
        available_height = layout["height"] - layout["margin"] - layout["footer_height"]
        return current_y + required_height <= available_height

    def generate_invoice(self, invoice_id: str, layout_type: str = "standard") -> Tuple[Image.Image, Dict]:
        """Generate optimized synthetic invoice with proper spacing."""
        layout = self.layouts[layout_type]
        img = Image.new("RGB", (layout["width"], layout["height"]), "white")
        draw = ImageDraw.Draw(img)
        
        annotations = {"words": [], "boxes": [], "ner_tags": []}
        current_y = layout["margin"]
        content_width = layout["width"] - (2 * layout["margin"])
        
        # Generate invoice data
        invoice_data = {
            "invoice_number": f"INV-{uuid.uuid4().hex[:8].upper()}",
            "invoice_date": self.fake.date_between(start_date="-1y", end_date="today").strftime("%m/%d/%Y"),
            "vendor_name": self.fake.company()[:30],  # Limit vendor name length
            "vendor_address": self.fake.address().replace("\n", ", ")[:80],  # Limit address length
            "customer_name": self.fake.name()[:25],  # Limit customer name length
            "customer_address": self.fake.address().replace("\n", ", ")[:80],  # Limit address length
            "line_items": [self._generate_line_item() for _ in range(random.randint(2, 4))]  # Reduced max items
        }
        
        # Calculate amounts
        subtotal = sum(item["quantity"] * item["unit_price"] for item in invoice_data["line_items"])
        tax_amount = round(subtotal * random.choice([0.05, 0.07, 0.10]), 2)
        total_amount = round(subtotal + tax_amount, 2)
        
        # Header with proper centering
        header_text = random.choice(["INVOICE", "BILL", "RECEIPT"])
        header_width = self._get_text_dimensions(header_text, self.font_xlarge)[0]
        header_x = (layout["width"] - header_width) // 2
        
        current_y += self._add_text_with_annotation(
            draw, header_text, "HEADER", header_x, current_y, 
            self.font_xlarge, layout["width"], layout["height"], annotations
        ) + layout["section_spacing"]
        
        # Invoice details section
        details_section = [
            (f'Invoice No.: {invoice_data["invoice_number"]}', "INVOICE_NUM"),
            (f'Date: {invoice_data["invoice_date"]}', "DATE"),
        ]
        
        for text, tag in details_section:
            current_y += self._add_text_with_annotation(
                draw, text, tag, layout["margin"], current_y, 
                self.font_medium, layout["width"], layout["height"], annotations
            ) + layout["line_height"]
        
        current_y += layout["section_spacing"]
        
        # Vendor information
        current_y += self._add_text_with_annotation(
            draw, f'Vendor: {invoice_data["vendor_name"]}', "VENDOR_NAME", 
            layout["margin"], current_y, self.font_medium, 
            layout["width"], layout["height"], annotations, content_width // 2
        ) + layout["line_height"]
        
        current_y += self._add_text_with_annotation(
            draw, invoice_data["vendor_address"], "VENDOR_ADDRESS", 
            layout["margin"], current_y, self.font_small, 
            layout["width"], layout["height"], annotations, content_width // 2
        ) + layout["section_spacing"]
        
        # Customer information
        current_y += self._add_text_with_annotation(
            draw, f'Bill To: {invoice_data["customer_name"]}', "CUSTOMER_NAME", 
            layout["margin"], current_y, self.font_medium, 
            layout["width"], layout["height"], annotations, content_width // 2
        ) + layout["line_height"]
        
        current_y += self._add_text_with_annotation(
            draw, invoice_data["customer_address"], "CUSTOMER_ADDRESS", 
            layout["margin"], current_y, self.font_small, 
            layout["width"], layout["height"], annotations, content_width // 2
        ) + layout["section_spacing"]
        
        # Check space before line items
        estimated_items_height = len(invoice_data["line_items"]) * (layout["line_height"] + 10) + 100
        if not self._check_space_available(current_y, estimated_items_height, layout):
            # Reduce number of line items if not enough space
            max_items = max(1, (layout["height"] - current_y - layout["footer_height"] - 100) // (layout["line_height"] + 10))
            invoice_data["line_items"] = invoice_data["line_items"][:max_items]
            # Recalculate totals
            subtotal = sum(item["quantity"] * item["unit_price"] for item in invoice_data["line_items"])
            tax_amount = round(subtotal * random.choice([0.05, 0.07, 0.10]), 2)
            total_amount = round(subtotal + tax_amount, 2)
        
        # Line items table header
        table_y = current_y
        col_positions = [0, 0.5, 0.7, 0.85]  # Adjusted column positions
        col_widths = [content_width * 0.5, content_width * 0.2, content_width * 0.15, content_width * 0.15]
        
        headers = ["Description", "Qty", "Price", "Total"]
        header_tags = ["ITEM_DESCRIPTION", "QUANTITY", "UNIT_PRICE", "LINE_TOTAL"]
        
        for i, (header, tag) in enumerate(zip(headers, header_tags)):
            x_pos = layout["margin"] + int(content_width * col_positions[i])
            self._add_text_with_annotation(
                draw, header, tag, x_pos, current_y, 
                self.font_medium, layout["width"], layout["height"], annotations,
                int(col_widths[i])
            )
        
        current_y += 35
        
        # Draw table separator line
        draw.line([(layout["margin"], current_y - 5), 
                  (layout["width"] - layout["margin"], current_y - 5)], 
                 fill="black", width=1)
        
        # Line items
        for item in invoice_data["line_items"]:
            line_total = round(item["quantity"] * item["unit_price"], 2)
            item_data = [
                (item["description"], "ITEM_DESCRIPTION"),
                (str(item["quantity"]), "QUANTITY"),
                (f"${item['unit_price']:.2f}", "UNIT_PRICE"),
                (f"${line_total:.2f}", "LINE_TOTAL")
            ]
            
            for i, (text, tag) in enumerate(item_data):
                x_pos = layout["margin"] + int(content_width * col_positions[i])
                self._add_text_with_annotation(
                    draw, text, tag, x_pos, current_y, 
                    self.font_small, layout["width"], layout["height"], annotations,
                    int(col_widths[i])
                )
            
            current_y += layout["line_height"] + 8
        
        current_y += layout["section_spacing"]
        
        # Totals section - right aligned
        totals_x = layout["margin"] + int(content_width * 0.6)
        totals = [
            (f"Subtotal: ${subtotal:.2f}", "SUBTOTAL"),
            (f"Tax: ${tax_amount:.2f}", "TAX_AMOUNT"),
            (f"TOTAL: ${total_amount:.2f}", "TOTAL_AMOUNT")
        ]
        
        for text, tag in totals:
            current_y += self._add_text_with_annotation(
                draw, text, tag, totals_x, current_y, 
                self.font_medium if tag != "TOTAL_AMOUNT" else self.font_large, 
                layout["width"], layout["height"], annotations,
                int(content_width * 0.35)
            ) + layout["line_height"]
        
        # Create final annotation
        annotation = {
            "id": invoice_id,
            "image_filename": f"{invoice_id}.png",
            "words": annotations["words"],
            "boxes": annotations["boxes"],
            "ner_tags": annotations["ner_tags"],
            "image_width": layout["width"],
            "image_height": layout["height"],
            "layout_type": layout_type,
            "total_words": len(annotations["words"]),
            "invoice_data": {
                **invoice_data, 
                "subtotal": subtotal, 
                "tax_amount": tax_amount, 
                "total_amount": total_amount
            }
        }
        
        return img, annotation

    def generate_batch(self, num_samples: int = 100):
        """Generate batch of synthetic invoices efficiently."""
        logger.info(f"Generating {num_samples} synthetic invoices...")
        
        successful_generations = 0
        failed_generations = 0
        
        for i in range(num_samples):
            invoice_id = f"synthetic_invoice_{i+1:04d}"
            layout_type = random.choice(list(self.layouts.keys()))
            
            try:
                img, annotation = self.generate_invoice(invoice_id, layout_type)
                
                # Validate annotation
                if len(annotation["words"]) != len(annotation["boxes"]) or \
                   len(annotation["words"]) != len(annotation["ner_tags"]):
                    logger.error(f"Annotation mismatch for {invoice_id}")
                    failed_generations += 1
                    continue
                
                # Save files
                img.save(self.output_dir / "images" / f"{invoice_id}.png", quality=95)
                with open(self.output_dir / "annotations" / f"{invoice_id}.json", "w") as f:
                    json.dump(annotation, f, indent=2)
                
                successful_generations += 1
                
                if (i + 1) % 25 == 0:
                    logger.info(f"Generated {successful_generations}/{i+1} invoices successfully")
                    
            except Exception as e:
                logger.error(f"Error generating invoice {invoice_id}: {e}")
                failed_generations += 1
        
        logger.info(f"Generation complete: {successful_generations} successful, {failed_generations} failed")
        return successful_generations, failed_generations

    def validate_dataset(self) -> Dict:
        """Validate generated dataset for overlaps and annotation consistency."""
        logger.info("Validating generated dataset...")
        
        validation_results = {
            "total_files": 0,
            "valid_annotations": 0,
            "annotation_errors": [],
            "image_errors": [],
            "bbox_issues": []
        }
        
        annotation_files = list((self.output_dir / "annotations").glob("*.json"))
        validation_results["total_files"] = len(annotation_files)
        
        for ann_file in annotation_files:
            try:
                with open(ann_file, "r") as f:
                    annotation = json.load(f)
                
                # Check annotation consistency
                words = annotation.get("words", [])
                boxes = annotation.get("boxes", [])
                ner_tags = annotation.get("ner_tags", [])
                
                if len(words) != len(boxes) or len(words) != len(ner_tags):
                    validation_results["annotation_errors"].append({
                        "file": ann_file.name,
                        "words": len(words),
                        "boxes": len(boxes),
                        "ner_tags": len(ner_tags)
                    })
                    continue
                
                # Check bounding boxes for overlaps
                overlap_count = 0
                for i, box1 in enumerate(boxes):
                    for j, box2 in enumerate(boxes[i+1:], i+1):
                        if self._boxes_overlap(box1, box2):
                            overlap_count += 1
                
                if overlap_count > len(boxes) * 0.1:  # Allow some natural overlap
                    validation_results["bbox_issues"].append({
                        "file": ann_file.name,
                        "overlap_count": overlap_count,
                        "total_boxes": len(boxes)
                    })
                
                # Check if image exists
                img_path = self.output_dir / "images" / annotation["image_filename"]
                if not img_path.exists():
                    validation_results["image_errors"].append(ann_file.name)
                    continue
                
                validation_results["valid_annotations"] += 1
                
            except Exception as e:
                validation_results["annotation_errors"].append({
                    "file": ann_file.name,
                    "error": str(e)
                })
        
        # Save validation report
        with open(self.output_dir / "validation_report.json", "w") as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation complete: {validation_results['valid_annotations']}/{validation_results['total_files']} valid")
        return validation_results
    
    def _boxes_overlap(self, box1: List[int], box2: List[int]) -> bool:
        """Check if two bounding boxes overlap significantly."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate overlap area
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = overlap_x * overlap_y
        
        # Calculate individual areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Check if overlap is significant (>30% of smaller box)
        if min(area1, area2) > 0:
            overlap_ratio = overlap_area / min(area1, area2)
            return overlap_ratio > 0.3
        
        return False

    def generate_dataset_manifest(self) -> Dict:
        """Create comprehensive dataset manifest with validation results."""
        annotations = list((self.output_dir / "annotations").glob("*.json"))
        
        # Collect statistics
        segment_counts = defaultdict(int)
        family_counts = defaultdict(int)
        class_counts = defaultdict(int)
        commodity_counts = defaultdict(int)
        layout_counts = defaultdict(int)
        word_counts = []
        
        for ann_file in annotations:
            try:
                with open(ann_file, "r") as f:
                    data = json.load(f)
                
                # Layout statistics
                layout_counts[data.get("layout_type", "unknown")] += 1
                word_counts.append(data.get("total_words", 0))
                
                # UNSPSC statistics
                for item in data.get("invoice_data", {}).get("line_items", []):
                    unspsc = item.get("unspsc_details", {})
                    segment_counts[unspsc.get("segment_name", "")] += 1
                    family_counts[unspsc.get("family_name", "")] += 1
                    class_counts[unspsc.get("class_name", "")] += 1
                    commodity_counts[unspsc.get("commodity_name", "")] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {ann_file.name}: {e}")
        
        manifest = {
            "dataset_info": {
                "name": "Synthetic Invoice Dataset with UNSPSC Classification",
                "generation_date": datetime.now().isoformat(),
                "total_samples": len(annotations),
                "unspsc_source": self.data_utils.unspsc_url,
                "average_words_per_invoice": sum(word_counts) / len(word_counts) if word_counts else 0
            },
            "layout_statistics": dict(layout_counts),
            "word_count_stats": {
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0,
                "average": sum(word_counts) / len(word_counts) if word_counts else 0
            },
            "unspsc_statistics": {
                "unique_segments": len(segment_counts),
                "unique_families": len(family_counts),
                "unique_classes": len(class_counts),
                "unique_commodities": len(commodity_counts),
                "top_segments": dict(sorted(segment_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_commodities": dict(sorted(commodity_counts.items(), key=lambda x: x[1], reverse=True)[:20])
            },
            "ner_tags": self.ner_tags,
            "tag_mapping": self.tag_to_id
        }
        
        with open(self.output_dir / "dataset_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Dataset manifest saved to {self.output_dir}/dataset_manifest.json")
        return manifest


if __name__ == "__main__":
    # Ensure directories exist
    Path("data/synthetic_invoices").mkdir(exist_ok=True, parents=True)
    
    # Generate synthetic dataset
    generator = SyntheticInvoiceGenerator(output_dir="data/synthetic_invoices")
    
    # Generate invoices
    successful, failed = generator.generate_batch(num_samples=100)
    
    # Validate dataset
    validation_results = generator.validate_dataset()
    
    # Generate manifest
    manifest = generator.generate_dataset_manifest()
    
    print(f"\nDataset Generation Summary:")
    print(f"Successful generations: {successful}")
    print(f"Failed generations: {failed}")
    print(f"Valid annotations: {validation_results['valid_annotations']}")
    print(f"Annotation errors: {len(validation_results['annotation_errors'])}")
    print(f"Bounding box issues: {len(validation_results['bbox_issues'])}")