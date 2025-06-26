import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import requests
from collections import defaultdict


class SyntheticInvoiceGenerator:
    """Generate synthetic invoice/receipt data with PSC classification integration."""

    def __init__(
        self, output_dir: str = "data/synthetic_data", psc_data: Optional[Dict] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)

        self.fake = Faker()
        self.psc_data = psc_data or self._load_psc_data()

        # Enhanced layout configurations with better spacing
        self.layouts = {
            "standard": {
                "width": 800,
                "height": 1100,
                "margin": 60,
                "header_height": 80,
                "section_padding": 25,
                "line_height": 22,
            },
            "compact": {
                "width": 750,
                "height": 1000,
                "margin": 50,
                "header_height": 70,
                "section_padding": 20,
                "line_height": 20,
            },
            "receipt": {
                "width": 500,
                "height": 800,
                "margin": 30,
                "header_height": 60,
                "section_padding": 15,
                "line_height": 18,
            },
        }

        # Enhanced font configuration
        self.font_config = {
            "title": {"size": 24, "weight": "bold"},
            "header": {"size": 14, "weight": "bold"},
            "subheader": {"size": 12, "weight": "bold"},
            "normal": {"size": 11, "weight": "normal"},
            "small": {"size": 9, "weight": "normal"},
            "tiny": {"size": 8, "weight": "normal"},
        }

        # Color scheme
        self.colors = {
            "black": "#000000",
            "dark_gray": "#333333",
            "medium_gray": "#666666",
            "light_gray": "#CCCCCC",
            "blue": "#2E86AB",
            "green": "#28A745",
            "red": "#DC3545",
        }

    def _load_psc_data(self) -> Dict:
        """Load PSC data from GitHub if not provided."""
        try:
            url = "https://raw.githubusercontent.com/le-patrice/Datasets/refs/heads/main/pscs.json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            raw_data = response.json()

            # Structure PSC data for generation
            psc_mapping = {}
            category_items = defaultdict(list)

            for item in raw_data:
                psc_code = item.get("psc", "")
                short_name = item.get("shortName", "")
                spend_category = item.get("spendCategoryTitle", "")
                portfolio_group = item.get("portfolioGroup", "")

                psc_info = {
                    "psc": psc_code,
                    "shortName": short_name,
                    "spendCategoryTitle": spend_category,
                    "portfolioGroup": portfolio_group,
                }

                psc_mapping[psc_code] = psc_info
                category_items[spend_category].append(psc_info)

            return {
                "psc_mapping": psc_mapping,
                "category_items": dict(category_items),
                "all_categories": list(category_items.keys()),
            }

        except Exception as e:
            print(f"Error loading PSC data: {e}")
            return {"psc_mapping": {}, "category_items": {}, "all_categories": []}

    def _get_font(self, font_type: str, try_system_fonts: bool = True):
        """Get font with fallback options."""
        size = self.font_config[font_type]["size"]

        if try_system_fonts:
            font_files = [
                "arial.ttf",
                "Arial.ttf",
                "helvetica.ttf",
                "Helvetica.ttf",
                "calibri.ttf",
                "Calibri.ttf",
                "verdana.ttf",
                "Verdana.ttf",
            ]

            for font_file in font_files:
                try:
                    return ImageFont.truetype(font_file, size)
                except:
                    continue

        # Fallback to default font
        try:
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()

    def _wrap_text(self, text: str, max_width: int, font, draw) -> List[str]:
        """Wrap text to fit within specified width."""
        if not text:
            return [""]

        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]

            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Handle very long single words
                    lines.append(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines if lines else [""]

    def _get_random_psc_items(self, num_items: int = None) -> List[Dict]:
        """Get random PSC items for invoice line items."""
        if not self.psc_data["psc_mapping"]:
            return []

        if num_items is None:
            num_items = random.randint(1, 5)  # Reduced to prevent overflow

        categories = self.psc_data["all_categories"]
        selected_items = []

        for _ in range(num_items):
            if categories:
                category = random.choice(categories)
                category_pscs = self.psc_data["category_items"].get(category, [])
                if category_pscs:
                    psc_item = random.choice(category_pscs)

                    item_data = {
                        "psc": psc_item["psc"],
                        "shortName": psc_item["shortName"],
                        "spendCategoryTitle": psc_item["spendCategoryTitle"],
                        "portfolioGroup": psc_item["portfolioGroup"],
                        "description": self._generate_item_description(psc_item),
                        "quantity": random.randint(1, 15),
                        "unit_price": round(random.uniform(25.0, 850.0), 2),
                    }
                    item_data["line_total"] = round(
                        item_data["quantity"] * item_data["unit_price"], 2
                    )
                    selected_items.append(item_data)

        return selected_items

    def _generate_item_description(self, psc_item: Dict) -> str:
        """Generate realistic item description based on PSC data."""
        base_name = psc_item["shortName"]

        # Truncate very long descriptions
        if len(base_name) > 60:
            # Split on common separators and take first meaningful part
            parts = base_name.split(",")[0].split(";")[0].split(" AND ")[0]
            base_name = parts[:60].strip()

        # Add some variation
        prefixes = ["", "Premium", "Standard", "Professional", "Industrial"]
        suffixes = ["", "System", "Kit", "Package", "Assembly"]

        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        # Construct description
        parts = [p for p in [prefix, base_name, suffix] if p]
        description = " ".join(parts)

        # Final length check
        if len(description) > 80:
            description = description[:77] + "..."

        return description

    def _determine_document_psc(self, line_items: List[Dict]) -> Dict:
        """Determine overall document PSC classification based on line items."""
        if not line_items:
            return {}

        category_totals = defaultdict(float)
        portfolio_totals = defaultdict(float)

        for item in line_items:
            category = item["spendCategoryTitle"]
            portfolio = item["portfolioGroup"]
            line_total = item["line_total"]

            category_totals[category] += line_total
            portfolio_totals[portfolio] += line_total

        dominant_category = max(category_totals.items(), key=lambda x: x[1])[0]
        dominant_portfolio = max(portfolio_totals.items(), key=lambda x: x[1])[0]

        dominant_psc = None
        for item in line_items:
            if item["spendCategoryTitle"] == dominant_category:
                dominant_psc = item["psc"]
                break

        return {
            "document_psc": dominant_psc,
            "document_category": dominant_category,
            "document_portfolio": dominant_portfolio,
            "category_breakdown": dict(category_totals),
            "portfolio_breakdown": dict(portfolio_totals),
        }

    def generate_invoice_data(self) -> Dict:
        """Generate synthetic invoice data with PSC integration."""
        invoice_data = {
            "invoice_id": f"INV-{random.randint(100000, 999999)}",
            "date": self.fake.date_between(start_date="-1y", end_date="today").strftime(
                "%Y-%m-%d"
            ),
            "due_date": (
                datetime.now() + timedelta(days=random.randint(7, 30))
            ).strftime("%Y-%m-%d"),
            "vendor_name": self.fake.company(),
            "vendor_address": self.fake.address().replace("\n", ", "),
            "vendor_email": self.fake.email(),
            "vendor_phone": self.fake.phone_number(),
            "bill_to_name": self.fake.name(),
            "bill_to_address": self.fake.address().replace("\n", ", "),
        }

        line_items = self._get_random_psc_items()
        invoice_data["line_items"] = line_items

        subtotal = sum(item["line_total"] for item in line_items)
        tax_rate = random.choice([0.05, 0.08, 0.10, 0.125])
        tax_amount = round(subtotal * tax_rate, 2)
        total_amount = round(subtotal + tax_amount, 2)

        invoice_data.update(
            {
                "subtotal": subtotal,
                "tax_rate": tax_rate,
                "tax_amount": tax_amount,
                "total_amount": total_amount,
            }
        )

        document_psc = self._determine_document_psc(line_items)
        invoice_data["psc_classification"] = document_psc

        return invoice_data

    def render_invoice_image(
        self, invoice_data: Dict, layout: str = "standard"
    ) -> Tuple[Image.Image, List[Dict]]:
        """Render invoice as image with improved layout and formatting."""
        config = self.layouts[layout]
        img = Image.new("RGB", (config["width"], config["height"]), "white")
        draw = ImageDraw.Draw(img)

        # Load fonts
        font_title = self._get_font("title")
        font_header = self._get_font("header")
        font_subheader = self._get_font("subheader")
        font_normal = self._get_font("normal")
        font_small = self._get_font("small")

        annotations = []
        margin = config["margin"]
        y_pos = margin
        content_width = config["width"] - (2 * margin)

        # === HEADER SECTION ===
        # Invoice title with background
        title_text = "INVOICE"
        title_bbox = draw.textbbox((0, 0), title_text, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        title_x = (config["width"] - title_width) // 2

        # Draw title background
        title_bg_padding = 15
        draw.rectangle(
            [
                title_x - title_bg_padding,
                y_pos - 5,
                title_x + title_width + title_bg_padding,
                y_pos + title_height + 10,
            ],
            fill=self.colors["light_gray"],
        )

        draw.text(
            (title_x, y_pos), title_text, fill=self.colors["black"], font=font_title
        )
        annotations.append(
            {
                "text": title_text,
                "bbox": [title_x, y_pos, title_x + title_width, y_pos + title_height],
                "label": "HEADER",
                "field_type": "title",
            }
        )
        y_pos += title_height + config["section_padding"]

        # === INVOICE INFO SECTION ===
        # Invoice number and date in two columns
        inv_section_y = y_pos

        # Left column - Invoice number
        inv_num_label = "Invoice #:"
        inv_num_value = invoice_data["invoice_id"]
        draw.text(
            (margin, y_pos),
            inv_num_label,
            fill=self.colors["dark_gray"],
            font=font_subheader,
        )
        label_bbox = draw.textbbox((margin, y_pos), inv_num_label, font=font_subheader)
        y_pos += config["line_height"]

        draw.text(
            (margin, y_pos), inv_num_value, fill=self.colors["black"], font=font_normal
        )
        value_bbox = draw.textbbox((margin, y_pos), inv_num_value, font=font_normal)
        annotations.append(
            {
                "text": inv_num_value,
                "bbox": [margin, y_pos, value_bbox[2], value_bbox[3]],
                "label": "INVOICE_NUM",
                "field_type": "invoice_number",
            }
        )

        # Right column - Date
        date_x = margin + content_width // 2
        date_label = "Date:"
        date_value = invoice_data["date"]
        draw.text(
            (date_x, inv_section_y),
            date_label,
            fill=self.colors["dark_gray"],
            font=font_subheader,
        )
        draw.text(
            (date_x, inv_section_y + config["line_height"]),
            date_value,
            fill=self.colors["black"],
            font=font_normal,
        )

        date_value_bbox = draw.textbbox(
            (date_x, inv_section_y + config["line_height"]),
            date_value,
            font=font_normal,
        )
        annotations.append(
            {
                "text": date_value,
                "bbox": [
                    date_x,
                    inv_section_y + config["line_height"],
                    date_value_bbox[2],
                    date_value_bbox[3],
                ],
                "label": "DATE",
                "field_type": "date",
            }
        )

        y_pos += config["section_padding"] + 10

        # === VENDOR SECTION ===
        vendor_label = "From:"
        vendor_name = invoice_data["vendor_name"]
        draw.text(
            (margin, y_pos),
            vendor_label,
            fill=self.colors["dark_gray"],
            font=font_subheader,
        )
        y_pos += config["line_height"]

        draw.text(
            (margin, y_pos), vendor_name, fill=self.colors["black"], font=font_normal
        )
        vendor_bbox = draw.textbbox((margin, y_pos), vendor_name, font=font_normal)
        annotations.append(
            {
                "text": vendor_name,
                "bbox": [margin, y_pos, vendor_bbox[2], vendor_bbox[3]],
                "label": "VENDOR",
                "field_type": "vendor_name",
            }
        )
        y_pos += config["section_padding"] + 15

        # === LINE ITEMS SECTION ===
        # Table header with background
        table_header_y = y_pos
        table_height = 25

        # Calculate column widths based on layout
        if layout == "receipt":
            desc_width = int(content_width * 0.5)
            qty_width = int(content_width * 0.15)
            price_width = int(content_width * 0.175)
            total_width = int(content_width * 0.175)
        else:
            desc_width = int(content_width * 0.55)
            qty_width = int(content_width * 0.15)
            price_width = int(content_width * 0.15)
            total_width = int(content_width * 0.15)

        # Column positions
        desc_x = margin
        qty_x = desc_x + desc_width
        price_x = qty_x + qty_width
        total_x = price_x + price_width

        # Draw table header background
        draw.rectangle(
            [
                margin,
                table_header_y,
                margin + content_width,
                table_header_y + table_height,
            ],
            fill=self.colors["light_gray"],
        )

        # Table headers
        headers = ["Description", "Qty", "Unit Price", "Total"]
        header_positions = [desc_x + 5, qty_x + 5, price_x + 5, total_x + 5]

        for i, header in enumerate(headers):
            draw.text(
                (header_positions[i], table_header_y + 6),
                header,
                fill=self.colors["black"],
                font=font_subheader,
            )

        y_pos = table_header_y + table_height + 5

        # Line items
        for item in invoice_data["line_items"]:
            item_start_y = y_pos

            # Description (with text wrapping)
            desc_text = item["description"]
            wrapped_desc = self._wrap_text(
                desc_text, desc_width - 10, font_normal, draw
            )

            desc_y = y_pos
            for line in wrapped_desc:
                draw.text(
                    (desc_x + 5, desc_y),
                    line,
                    fill=self.colors["black"],
                    font=font_normal,
                )
                desc_y += config["line_height"]

            # Store annotation for first line of description
            if wrapped_desc:
                desc_bbox = draw.textbbox(
                    (desc_x + 5, y_pos), wrapped_desc[0], font=font_normal
                )
                annotations.append(
                    {
                        "text": desc_text,
                        "bbox": [desc_x + 5, y_pos, desc_bbox[2], desc_bbox[3]],
                        "label": "ITEM_DESC",
                        "field_type": "line_item_description",
                        "psc": item["psc"],
                        "psc_short_name": item["shortName"],
                        "spend_category": item["spendCategoryTitle"],
                        "portfolio_group": item["portfolioGroup"],
                    }
                )

            # Quantity (centered)
            qty_text = str(item["quantity"])
            qty_bbox = draw.textbbox((0, 0), qty_text, font=font_normal)
            qty_center_x = qty_x + (qty_width - (qty_bbox[2] - qty_bbox[0])) // 2
            draw.text(
                (qty_center_x, y_pos),
                qty_text,
                fill=self.colors["black"],
                font=font_normal,
            )

            qty_actual_bbox = draw.textbbox(
                (qty_center_x, y_pos), qty_text, font=font_normal
            )
            annotations.append(
                {
                    "text": qty_text,
                    "bbox": [
                        qty_center_x,
                        y_pos,
                        qty_actual_bbox[2],
                        qty_actual_bbox[3],
                    ],
                    "label": "QTY",
                    "field_type": "quantity",
                }
            )

            # Unit price (right-aligned)
            price_text = f"${item['unit_price']:.2f}"
            price_bbox = draw.textbbox((0, 0), price_text, font=font_normal)
            price_right_x = price_x + price_width - (price_bbox[2] - price_bbox[0]) - 5
            draw.text(
                (price_right_x, y_pos),
                price_text,
                fill=self.colors["black"],
                font=font_normal,
            )

            price_actual_bbox = draw.textbbox(
                (price_right_x, y_pos), price_text, font=font_normal
            )
            annotations.append(
                {
                    "text": price_text,
                    "bbox": [
                        price_right_x,
                        y_pos,
                        price_actual_bbox[2],
                        price_actual_bbox[3],
                    ],
                    "label": "PRICE",
                    "field_type": "unit_price",
                }
            )

            # Line total (right-aligned)
            total_text = f"${item['line_total']:.2f}"
            total_bbox = draw.textbbox((0, 0), total_text, font=font_normal)
            total_right_x = total_x + total_width - (total_bbox[2] - total_bbox[0]) - 5
            draw.text(
                (total_right_x, y_pos),
                total_text,
                fill=self.colors["black"],
                font=font_normal,
            )

            total_actual_bbox = draw.textbbox(
                (total_right_x, y_pos), total_text, font=font_normal
            )
            annotations.append(
                {
                    "text": total_text,
                    "bbox": [
                        total_right_x,
                        y_pos,
                        total_actual_bbox[2],
                        total_actual_bbox[3],
                    ],
                    "label": "TOTAL",
                    "field_type": "line_total",
                }
            )

            # Move to next line (account for wrapped description)
            y_pos = max(desc_y, y_pos + config["line_height"]) + 5

        # === TOTALS SECTION ===
        y_pos += 15

        # Draw separator line
        totals_x = margin + content_width * 0.6
        draw.line(
            [(totals_x, y_pos), (margin + content_width, y_pos)],
            fill=self.colors["medium_gray"],
            width=2,
        )
        y_pos += 15

        # Subtotal
        subtotal_label = "Subtotal:"
        subtotal_value = f"${invoice_data['subtotal']:.2f}"
        draw.text(
            (totals_x, y_pos),
            subtotal_label,
            fill=self.colors["dark_gray"],
            font=font_normal,
        )

        subtotal_bbox = draw.textbbox((0, 0), subtotal_value, font=font_normal)
        subtotal_right_x = (
            margin + content_width - (subtotal_bbox[2] - subtotal_bbox[0])
        )
        draw.text(
            (subtotal_right_x, y_pos),
            subtotal_value,
            fill=self.colors["black"],
            font=font_normal,
        )

        subtotal_actual_bbox = draw.textbbox(
            (subtotal_right_x, y_pos), subtotal_value, font=font_normal
        )
        annotations.append(
            {
                "text": subtotal_value,
                "bbox": [
                    subtotal_right_x,
                    y_pos,
                    subtotal_actual_bbox[2],
                    subtotal_actual_bbox[3],
                ],
                "label": "SUBTOTAL",
                "field_type": "subtotal",
            }
        )
        y_pos += config["line_height"] + 5

        # Tax
        tax_label = f"Tax ({invoice_data['tax_rate']*100:.1f}%):"
        tax_value = f"${invoice_data['tax_amount']:.2f}"
        draw.text(
            (totals_x, y_pos),
            tax_label,
            fill=self.colors["dark_gray"],
            font=font_normal,
        )

        tax_bbox = draw.textbbox((0, 0), tax_value, font=font_normal)
        tax_right_x = margin + content_width - (tax_bbox[2] - tax_bbox[0])
        draw.text(
            (tax_right_x, y_pos), tax_value, fill=self.colors["black"], font=font_normal
        )

        tax_actual_bbox = draw.textbbox(
            (tax_right_x, y_pos), tax_value, font=font_normal
        )
        annotations.append(
            {
                "text": tax_value,
                "bbox": [tax_right_x, y_pos, tax_actual_bbox[2], tax_actual_bbox[3]],
                "label": "TAX",
                "field_type": "tax_amount",
            }
        )
        y_pos += config["line_height"] + 10

        # Total (highlighted)
        draw.line(
            [(totals_x, y_pos), (margin + content_width, y_pos)],
            fill=self.colors["black"],
            width=2,
        )
        y_pos += 8

        total_label = "TOTAL:"
        total_value = f"${invoice_data['total_amount']:.2f}"
        draw.text(
            (totals_x, y_pos), total_label, fill=self.colors["black"], font=font_header
        )

        total_bbox = draw.textbbox((0, 0), total_value, font=font_header)
        total_right_x = margin + content_width - (total_bbox[2] - total_bbox[0])
        draw.text(
            (total_right_x, y_pos),
            total_value,
            fill=self.colors["green"],
            font=font_header,
        )

        total_actual_bbox = draw.textbbox(
            (total_right_x, y_pos), total_value, font=font_header
        )
        annotations.append(
            {
                "text": total_value,
                "bbox": [
                    total_right_x,
                    y_pos,
                    total_actual_bbox[2],
                    total_actual_bbox[3],
                ],
                "label": "TOTAL_AMT",
                "field_type": "total_amount",
            }
        )

        return img, annotations

    def create_layoutlm_annotation(
        self, annotations: List[Dict], invoice_data: Dict
    ) -> Dict:
        """Create LayoutLM-compatible annotation format."""
        words = []
        boxes = []
        ner_tags = []

        label_map = {
            "HEADER": "B-HEADER",
            "INVOICE_NUM": "B-INVOICE_NUM",
            "DATE": "B-DATE",
            "VENDOR": "B-VENDOR",
            "ITEM_DESC": "B-ITEM_DESC",
            "QTY": "B-QTY",
            "PRICE": "B-PRICE",
            "TOTAL": "B-TOTAL",
            "SUBTOTAL": "B-SUBTOTAL",
            "TAX": "B-TAX",
            "TOTAL_AMT": "B-TOTAL_AMT",
        }

        for ann in annotations:
            words.append(ann["text"])
            boxes.append(ann["bbox"])
            ner_tags.append(label_map.get(ann["label"], "O"))

        psc_info = invoice_data.get("psc_classification", {})

        layoutlm_annotation = {
            "words": words,
            "boxes": boxes,
            "ner_tags": ner_tags,
            "document_psc": psc_info.get("document_psc", ""),
            "document_category": psc_info.get("document_category", ""),
            "document_portfolio": psc_info.get("document_portfolio", ""),
            "line_items_psc": [
                {
                    "psc": item["psc"],
                    "shortName": item["shortName"],
                    "spendCategoryTitle": item["spendCategoryTitle"],
                    "portfolioGroup": item["portfolioGroup"],
                    "description": item["description"],
                    "line_total": item["line_total"],
                }
                for item in invoice_data["line_items"]
            ],
            "psc_breakdown": psc_info.get("category_breakdown", {}),
            "portfolio_breakdown": psc_info.get("portfolio_breakdown", {}),
            "invoice_metadata": {
                "invoice_id": invoice_data["invoice_id"],
                "date": invoice_data["date"],
                "vendor_name": invoice_data["vendor_name"],
                "total_amount": invoice_data["total_amount"],
            },
        }

        return layoutlm_annotation

    def generate_batch(
        self, num_samples: int = 100, layout_types: List[str] = None
    ) -> None:
        """Generate a batch of synthetic invoices with annotations."""
        if layout_types is None:
            layout_types = ["standard", "compact", "receipt"]

        print(
            f"Generating {num_samples} enhanced synthetic invoices with PSC integration..."
        )

        for i in range(num_samples):
            try:
                invoice_data = self.generate_invoice_data()
                layout = random.choice(layout_types)
                image, annotations = self.render_invoice_image(invoice_data, layout)

                file_id = f"synthetic_{i:05d}_{uuid.uuid4().hex[:8]}"

                image_path = self.output_dir / "images" / f"{file_id}.png"
                image.save(image_path, dpi=(300, 300), quality=95)

                layoutlm_annotation = self.create_layoutlm_annotation(
                    annotations, invoice_data
                )

                annotation_path = self.output_dir / "annotations" / f"{file_id}.json"
                with open(annotation_path, "w", encoding="utf-8") as f:
                    json.dump(layoutlm_annotation, f, indent=2, ensure_ascii=False)

                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples")

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        print(f"Enhanced batch generation complete. Files saved in {self.output_dir}")

    def generate_dataset_manifest(self) -> Dict:
        """Generate a manifest file describing the synthetic dataset."""
        image_dir = self.output_dir / "images"
        annotation_dir = self.output_dir / "annotations"

        samples = []
        psc_stats = defaultdict(int)
        category_stats = defaultdict(int)
        portfolio_stats = defaultdict(int)

        for ann_file in annotation_dir.glob("*.json"):
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)

                image_file = image_dir / f"{ann_file.stem}.png"
                if image_file.exists():
                    samples.append(
                        {
                            "image_path": str(image_file.relative_to(self.output_dir)),
                            "annotation_path": str(
                                ann_file.relative_to(self.output_dir)
                            ),
                            "document_psc": ann_data.get("document_psc", ""),
                            "document_category": ann_data.get("document_category", ""),
                            "document_portfolio": ann_data.get(
                                "document_portfolio", ""
                            ),
                        }
                    )

                    # Update statistics
                    psc_stats[ann_data.get("document_psc", "")] += 1
                    category_stats[ann_data.get("document_category", "")] += 1
                    portfolio_stats[ann_data.get("document_portfolio", "")] += 1

            except Exception as e:
                print(f"Error processing {ann_file}: {e}")
                continue

        manifest = {
            "dataset_info": {
                "name": "Synthetic Invoice Dataset with PSC Classification",
                "version": "1.0",
                "generated_date": datetime.now().isoformat(),
                "total_samples": len(samples),
                "format": "LayoutLMv2 compatible",
            },
            "samples": samples,
            "statistics": {
                "psc_distribution": dict(psc_stats),
                "category_distribution": dict(category_stats),
                "portfolio_distribution": dict(portfolio_stats),
            },
            "psc_metadata": {
                "total_unique_pscs": len(psc_stats),
                "total_categories": len(category_stats),
                "total_portfolios": len(portfolio_stats),
            },
        }

        # Save manifest
        manifest_path = self.output_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Dataset manifest saved to {manifest_path}")
        return manifest


# Usage example and main execution
if __name__ == "__main__":
    # Initialize generator
    generator = SyntheticInvoiceGenerator(output_dir="data/synthetic_invoices")

    # Generate synthetic dataset
    generator.generate_batch(num_samples=500, layout_types=["standard", "compact"])

    # Create dataset manifest
    manifest = generator.generate_dataset_manifest()

    print("\nDataset Generation Summary:")
    print(f"Total samples: {manifest['dataset_info']['total_samples']}")
    print(f"Unique PSCs: {manifest['psc_metadata']['total_unique_pscs']}")
    print(f"Categories: {manifest['psc_metadata']['total_categories']}")
    print(f"Portfolio groups: {manifest['psc_metadata']['total_portfolios']}")

    print("\nTop 20 PSC categories:")
    for category, count in sorted(
        manifest["statistics"]["category_distribution"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]:
        print(f"  {category}: {count} samples")

    print(f"\nSynthetic dataset ready for LayoutLMv2 training!")
