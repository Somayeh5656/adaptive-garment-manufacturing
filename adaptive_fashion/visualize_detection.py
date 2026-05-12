import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import YolosImageProcessor, YolosForObjectDetection
import os

# Load model and processor
processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
model.eval()

def detect_and_draw(image_path, output_path, threshold=0.3):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    draw = ImageDraw.Draw(image)

    # Try to load a larger font (20pt) – common on Ubuntu
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 50)  # 18 point
            break
    if font is None:
        font = ImageFont.load_default()  # fallback (small)
        print("Warning: No TrueType font found, using tiny default font.")

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        category = model.config.id2label[label.item()]
        text = f"{category}: {score:.2f}"

        # Draw bounding box
        draw.rectangle(box, outline="red", width=3)

        # Get text size (Pillow >= 8.0.0)
        try:
            bbox = draw.textbbox((box[0], box[1]-20), text, font=font)
        except AttributeError:
            # Fallback for older Pillow
            bbox = [box[0], box[1]-20, box[0]+len(text)*10, box[1]]
        # Draw white background for text
        draw.rectangle(bbox, fill="white")
        # Draw text in black
        draw.text((box[0], box[1]-20), text, fill="black", font=font)

    image.save(output_path)
    print(f"Saved detection image to {output_path}")

if __name__ == "__main__":
    input_image = "images/test_pic.jpg"   # adjust path
    output_image = "detected_output.jpg"
    detect_and_draw(input_image, output_image, threshold=0.3)