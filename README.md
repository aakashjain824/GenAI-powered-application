# GenAI-powered-application

## Image Captioning App with Gradio and BLIP

This project demonstrates an image captioning application built using [Gradio](https://www.gradio.app/) and the BLIP (Bootstrapped Language Image Pretraining) model. Users can upload an image, and the app generates a descriptive caption for it.

### Features

- Upload any image and receive an AI-generated caption
- Simple and interactive web interface powered by Gradio
- Utilizes the BLIP model for state-of-the-art image understanding

### Getting Started

1. **Install dependencies**:
    ```bash
    pip install gradio transformers torch
    ```

2. **Run the app**:
    ```bash
    python app.py
    ```

3. **Open the Gradio interface** in your browser to try out image captioning.

### Example

![example](assets/example.jpg)

*Caption generated: "A dog sitting on a bench in a park."*

### References

- [BLIP: Bootstrapped Language Image Pretraining](https://github.com/salesforce/BLIP)
- [Gradio Documentation](https://gradio.app/docs/)
