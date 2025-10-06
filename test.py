
import torch
from diffusers import FluxPipeline
from datetime import datetime
import os

def make_filename(prefix: str = "image", extension: str = ".png") -> str:
    """
    Generates a unique filename based on timestamp.
    
    Args:
        prefix: Prefix for the filename (default: 'image')
        extension: File extension (default: '.png')
        
    Returns:
        str: Unique filename (e.g., 'image_20251004_232415_123456.png')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}{extension}"

def generate_image(prompt: str, prefix: str = "image", extension: str = ".png") -> str:
    """
    Generates an image using FluxPipeline and saves it with a unique filename.
    
    Args:
        prompt: Text prompt for image generation
        prefix: Prefix for the filename (default: 'image')
        extension: File extension (default: '.png')
        
    Returns:
        str: Path to the saved image
        
    Raises:
        ValueError: If the prompt is empty or invalid
        OSError: If the image cannot be saved
    """
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

        # Generate image with reduced resolution
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            width=512,  # Reduced to fit VRAM
            height=512,
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
    
    return f"static/{filename}"