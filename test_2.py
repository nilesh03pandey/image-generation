import torch
from diffusers import FluxImg2ImgPipeline  # Correct import for image-to-image
from PIL import Image
import os

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the image-to-image pipeline with bfloat16
pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

# Memory optimizations for RTX 3050
pipe.enable_sequential_cpu_offload()  # Stricter offloading to fit 3.68 GiB VRAM
pipe.enable_attention_slicing()  # Reduce VRAM for attention layers

# Debug hook to confirm GPU usage during inference
def debug_device(module, input, output):
    print(f"Active module device: {input[0].device}")
pipe.transformer.register_forward_hook(debug_device)

# Verify initial model device
print("Initial model device:", next(pipe.transformer.parameters()).device)  # May print 'meta'

# Load input image (your existing generated image)
init_image = Image.open("flux-schnell.png").convert("RGB").resize((512, 512))

# Generate edited image
prompt = "an apple in the hand"
image = pipe(
    prompt,
    image=init_image,  # Input image for image-to-image
    strength=0.8,  # How much to transform (0.0 = no change, 1.0 = full generation from prompt)
    guidance_scale=3.5,  # Recommended for FLUX.1-schnell; improves prompt adherence
    num_inference_steps=4,
    max_sequence_length=256,
    width=512,
    height=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# Save the output (overwrites the input or use a new name)
image.save("flux-edited.png")
print("Saved edited image: flux-edited.png")