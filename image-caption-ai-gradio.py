import gradio as gr
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the processor and model only once (efficient for app lifecycle)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    """
    Generate a caption for the given image using the BLIP model.
    
    Args:
        image: A PIL Image or file path.

    Returns:
        A descriptive caption string.
    """
    # Handle URL string input (if needed)
    if isinstance(image, str):
        image = Image.open(requests.get(image, stream=True).raw).convert('RGB')

    # Ensure image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Preprocess the image with an optional text prompt
    inputs = processor(images=image, text="a photo of", return_tensors="pt")
    
    # Generate caption conditioned on the image and optional text
    output = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50
    )
    
    # Decode the generated token IDs to human-readable text
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Set up the Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ Image Captioning with BLIP",
    description="Upload any image and see what the BLIP model thinks it is! Built with Hugging Face Transformers and Gradio."
)

# Launch the web app (set share=True to get a public link)
iface.launch(share=True)