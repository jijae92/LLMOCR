#!/usr/bin/env python3
"""
Bounding box and confidence visualization utilities.

Features:
- Draw bounding boxes on images
- Highlight low-confidence regions
- Color-code by confidence score
- Add text labels with predictions
"""

from typing import List, Tuple, Optional
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install pillow numpy")
    exit(1)


class BBoxVisualizer:
    """Visualize bounding boxes and confidence scores on images."""

    def __init__(
        self,
        low_confidence_threshold: float = 0.7,
        font_size: int = 16,
    ):
        """
        Initialize visualizer.

        Args:
            low_confidence_threshold: Threshold for low confidence highlighting
            font_size: Font size for labels
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.font_size = font_size

        # Try to load font
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                font_size
            )
        except:
            self.font = ImageFont.load_default()

    def confidence_to_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Map confidence score to color.

        High confidence -> Green
        Medium confidence -> Yellow
        Low confidence -> Red
        """
        if confidence >= 0.9:
            return (0, 200, 0)  # Green
        elif confidence >= 0.7:
            return (200, 200, 0)  # Yellow
        else:
            return (200, 0, 0)  # Red

    def draw_bboxes(
        self,
        image: Image.Image,
        bboxes: List[Dict],
        highlight_low_confidence: bool = True,
    ) -> Image.Image:
        """
        Draw bounding boxes on image.

        Args:
            image: Input PIL image
            bboxes: List of bounding boxes with format:
                {
                    'box': [x1, y1, x2, y2],
                    'text': 'predicted text',
                    'confidence': 0.95
                }
            highlight_low_confidence: Whether to highlight low confidence regions

        Returns:
            Image with bounding boxes drawn
        """
        # Create copy
        img = image.copy()
        draw = ImageDraw.Draw(img)

        for bbox_info in bboxes:
            box = bbox_info['box']
            text = bbox_info.get('text', '')
            confidence = bbox_info.get('confidence', 1.0)

            # Determine color based on confidence
            color = self.confidence_to_color(confidence)

            # Draw box
            line_width = 3 if confidence < self.low_confidence_threshold else 2
            draw.rectangle(box, outline=color, width=line_width)

            # Highlight low confidence with semi-transparent overlay
            if highlight_low_confidence and confidence < self.low_confidence_threshold:
                # Create overlay
                overlay = Image.new('RGBA', img.size, (255, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(box, fill=(255, 0, 0, 50))
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)

            # Draw label background
            label = f"{text} ({confidence:.2f})"
            text_bbox = draw.textbbox((0, 0), label, font=self.font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            label_box = [
                box[0],
                box[1] - text_height - 4,
                box[0] + text_width + 8,
                box[1]
            ]

            # Ensure label is within image bounds
            if label_box[1] < 0:
                label_box[1] = box[3]
                label_box[3] = box[3] + text_height + 4

            draw.rectangle(label_box, fill=color)
            draw.text(
                (label_box[0] + 4, label_box[1] + 2),
                label,
                fill='white',
                font=self.font
            )

        return img

    def create_confidence_heatmap(
        self,
        image: Image.Image,
        confidence_map: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Create confidence heatmap overlay.

        Args:
            image: Input PIL image
            confidence_map: 2D array of confidence scores (0-1)
            alpha: Transparency of overlay

        Returns:
            Image with confidence heatmap overlay
        """
        # Resize confidence map to match image size
        conf_map_resized = np.array(Image.fromarray(confidence_map).resize(
            image.size, Image.BILINEAR
        ))

        # Create heatmap using colormap
        # Red -> Yellow -> Green
        heatmap = np.zeros((*conf_map_resized.shape, 3), dtype=np.uint8)

        # Low confidence (< 0.5) -> Red
        mask = conf_map_resized < 0.5
        heatmap[mask] = [200, 0, 0]

        # Medium confidence (0.5 - 0.8) -> Yellow
        mask = (conf_map_resized >= 0.5) & (conf_map_resized < 0.8)
        heatmap[mask] = [200, 200, 0]

        # High confidence (>= 0.8) -> Green
        mask = conf_map_resized >= 0.8
        heatmap[mask] = [0, 200, 0]

        # Create overlay
        heatmap_img = Image.fromarray(heatmap)
        overlay = Image.blend(image, heatmap_img, alpha=alpha)

        return overlay

    def draw_word_boxes(
        self,
        image: Image.Image,
        words: List[Dict],
        show_confidence: bool = True,
    ) -> Image.Image:
        """
        Draw word-level bounding boxes.

        Args:
            image: Input PIL image
            words: List of word detections:
                {
                    'text': 'word',
                    'box': [x1, y1, x2, y2],
                    'confidence': 0.95
                }
            show_confidence: Whether to show confidence scores

        Returns:
            Image with word boxes drawn
        """
        bboxes = []
        for word in words:
            bbox_info = {
                'box': word['box'],
                'text': word['text'] if show_confidence else word['text'],
                'confidence': word.get('confidence', 1.0),
            }
            bboxes.append(bbox_info)

        return self.draw_bboxes(image, bboxes, highlight_low_confidence=True)

    def create_comparison_view(
        self,
        original: Image.Image,
        processed: Image.Image,
        prediction: str,
        ground_truth: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Image.Image:
        """
        Create side-by-side comparison view.

        Args:
            original: Original input image
            processed: Processed/annotated image
            prediction: OCR prediction text
            ground_truth: Optional ground truth text
            confidence: Optional confidence score

        Returns:
            Comparison image
        """
        # Resize images to same height
        max_height = 400
        width1 = int(original.width * max_height / original.height)
        width2 = int(processed.width * max_height / processed.height)

        img1 = original.resize((width1, max_height))
        img2 = processed.resize((width2, max_height))

        # Create combined image
        total_width = width1 + width2 + 30  # 30px spacing
        total_height = max_height + 150  # Extra space for text

        combined = Image.new('RGB', (total_width, total_height), 'white')
        combined.paste(img1, (10, 10))
        combined.paste(img2, (width1 + 20, 10))

        # Add labels
        draw = ImageDraw.Draw(combined)

        y_text = max_height + 30

        # Original label
        draw.text((10, y_text), "Original", fill='black', font=self.font)

        # Processed label
        draw.text((width1 + 20, y_text), "Annotated", fill='black', font=self.font)

        # Prediction
        y_text += 30
        pred_label = f"Prediction: {prediction}"
        if confidence is not None:
            pred_label += f" (conf: {confidence:.2f})"
        draw.text((10, y_text), pred_label, fill='blue', font=self.font)

        # Ground truth
        if ground_truth:
            y_text += 25
            draw.text((10, y_text), f"Ground Truth: {ground_truth}", fill='green', font=self.font)

        return combined


def main():
    """Example usage."""
    # Create dummy image
    img = Image.new('RGB', (800, 400), 'white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 100), "한국어 OCR 테스트", fill='black')
    draw.text((50, 200), "Sample Text 123", fill='black')

    # Example bounding boxes
    bboxes = [
        {
            'box': [40, 90, 200, 130],
            'text': '한국어 OCR',
            'confidence': 0.95,
        },
        {
            'box': [210, 90, 280, 130],
            'text': '테스트',
            'confidence': 0.65,  # Low confidence
        },
        {
            'box': [40, 190, 180, 230],
            'text': 'Sample Text',
            'confidence': 0.88,
        },
        {
            'box': [190, 190, 240, 230],
            'text': '123',
            'confidence': 0.55,  # Low confidence
        },
    ]

    # Create visualizer
    visualizer = BBoxVisualizer(low_confidence_threshold=0.7)

    # Draw bounding boxes
    annotated = visualizer.draw_bboxes(img, bboxes, highlight_low_confidence=True)

    # Save result
    output_dir = Path("reports/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.save(output_dir / "bbox_example.png")

    # Create comparison view
    comparison = visualizer.create_comparison_view(
        original=img,
        processed=annotated,
        prediction="한국어 OCR 테스트 Sample Text 123",
        ground_truth="한국어 OCR 테스트 Sample Text 456",
        confidence=0.82,
    )
    comparison.save(output_dir / "comparison_example.png")

    print(f"✓ Visualization examples saved to {output_dir}")


if __name__ == "__main__":
    main()
