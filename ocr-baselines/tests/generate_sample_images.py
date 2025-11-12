"""Generate sample images from text files for testing."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_text_image(
    text: str,
    output_path: Path,
    width: int = 800,
    font_size: int = 24,
    bg_color: str = "white",
    text_color: str = "black",
):
    """
    Create an image from text.

    Args:
        text: Text content
        output_path: Output image path
        width: Image width
        font_size: Font size
        bg_color: Background color
        text_color: Text color
    """
    # Calculate image height based on text
    lines = text.split("\n")
    line_height = font_size + 10
    height = len(lines) * line_height + 40

    # Create image
    image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    # Try to load a font (fallback to default if not available)
    try:
        # Try to use a Korean-compatible font
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

    # Draw text
    y_position = 20
    for line in lines:
        draw.text((20, y_position), line, fill=text_color, font=font)
        y_position += line_height

    # Save image
    image.save(output_path)
    print(f"Created: {output_path}")


def main():
    """Generate sample images from text files."""
    data_dir = Path(__file__).parent / "data_samples"

    # Find all text files
    for text_file in data_dir.glob("*.txt"):
        # Skip if image already exists
        image_file = text_file.with_suffix(".jpg")
        if image_file.exists():
            print(f"Skipping {text_file.name} (image already exists)")
            continue

        # Read text
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Generate image
        create_text_image(text, image_file)

    print("\nSample images generated successfully!")
    print("Note: For better testing, replace these with real scanned documents.")


if __name__ == "__main__":
    main()
