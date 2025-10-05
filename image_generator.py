
import torch
from diffusers import FluxPipeline
from datetime import datetime
import os

def make_filename(prefix: str = "image", extension: str = ".png") -> str:

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}{extension}"

def generate_image(prompt: str, prefix: str = "image", extension: str = ".png") -> str:

    try:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Set environment variable to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Load the pipeline with bfloat16
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )

        # Memory optimizations for RTX 3050
        pipe.enable_sequential_cpu_offload()  # Stricter offloading
        pipe.enable_attention_slicing()  # Reduce VRAM for attention layers

        # Verify GPU usage
        print("Model device:", next(pipe.transformer.parameters()).device)

        f"Generate a highly detailed and realistic portrait with accurate human proportions, natural skin tones, subtle lighting, and expressive facial features. Capture a sense of atmosphere with soft shadows, fine details in hair and clothing, and dynamic posing that highlights the beauty of the subject. Focus on natural textures, depth, and artistic composition to create a visually striking image. {prompt}"
        image = pipe(
            prompt,
            guidance_scale=1.0,
            num_inference_steps=10,
            max_sequence_length=512,
            width=753,  # Reduced to fit VRAM
            height=753,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # Generate unique filename
        filename = make_filename(prefix=prefix, extension=extension)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

        # Save the image
        image.save(f"static/{filename}")
        return filename

    except OSError as e:
        raise OSError(f"Failed to save image: {str(e)}")
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")
fname = generate_image("a boy")

print(f"Image saved as: static/{fname}")