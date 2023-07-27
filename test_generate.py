from diffusers import DiffusionPipeline
import torch

base_repo_id = "./stable-diffusion-xl-base-1.0"
refiner_repo_id = "./stable-diffusion-xl-refiner-1.0"

prompt = "Illustration of punk rock kangaroos in the forest, wearing punk outfits, mohawks"
num_inference_steps = 100
high_noise_frac = 0.8
width = 1024
height = 1024
guidance_scale = 7.5

# Base model

print("Loading base model...")

base_pipe = DiffusionPipeline.from_pretrained(
    base_repo_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")
base_pipe.to("cuda")
base_pipe.unet = torch.compile(
    base_pipe.unet,
    mode="reduce-overhead",
    fullgraph=True)

# Refiner model

print("Loading refiner model...")

refiner_pipe = DiffusionPipeline.from_pretrained(
    refiner_repo_id,
    text_encoder_2=base_pipe.text_encoder_2,
    vae=base_pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")
refiner_pipe.to("cuda")
refiner_pipe.unet = torch.compile(
    refiner_pipe.unet,
    mode="reduce-overhead",
    fullgraph=True)

# Generate

print("Generating image...")

base_image = base_pipe(
    prompt=prompt,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent").images[0]

print("Refining image...")

refined_image = refiner_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
    image=base_image[None, :]).images[0]

refined_image.save(f"refined.png")
