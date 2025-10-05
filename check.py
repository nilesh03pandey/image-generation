# Quick text-to-image to create flux-schnell.png
from diffusers import FluxPipeline
import torch
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
image = pipe("a simple apple", num_inference_steps=4, width=512, height=512).images[0]
image.save("flux-schnell.png")