# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
import torch
import io
import argparse
import uvicorn
import logging
import base64

class ImageRequest(BaseModel):
    prompt: str
    steps: int
    guide: Optional[float] = 7.5

app = FastAPI()

base_repo_id = "./stable-diffusion-xl-base-1.0"
refiner_repo_id = "./stable-diffusion-xl-refiner-1.0"
width = 1024
height = 1024
high_noise_frac = 0.8
enable_wrong_lora = True # Recommended: This seems to help a lot with hands!

vae = None
base_pipe = None
refiner_pipe = None

def generate_image(prompt, steps, guide):
    global vae, base_pipe, refiner_pipe

    # Community VAE FP16 fix from madebyollin
    if vae is None:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                            torch_dtype=torch.float16)
    # Base model
    if base_pipe is None:
        logging.info("Loading base model...")
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            base_repo_id,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False)

        if enable_wrong_lora:
            # Community "wrong" improvement from minimaxir
            base_pipe.load_lora_weights("minimaxir/sdxl-wrong-lora")

        base_pipe.to("cuda")
        base_pipe.unet = torch.compile(
            base_pipe.unet,
            mode="reduce-overhead",
            fullgraph=True)

    # Refiner model
    if refiner_pipe is None:
        logging.info("Loading refiner model...")
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_repo_id,
            text_encoder_2=base_pipe.text_encoder_2,
            vae=base_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False)

        refiner_pipe.to("cuda")
        refiner_pipe.unet = torch.compile(
            refiner_pipe.unet,
            mode="reduce-overhead",
            fullgraph=True)

    if enable_wrong_lora:
        negative_prompt="wrong"
    else:
        negative_prompt=None

    # Generate
    logging.info("Generating image...")
    base_image = base_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guide,
        num_inference_steps=steps,
        denoising_end=high_noise_frac,
        output_type="latent").images[0]

    logging.info("Refining image...")
    refined_image = refiner_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        denoising_start=high_noise_frac,
        image=base_image[None, :]).images[0]

    image_bytes = io.BytesIO()
    refined_image.save(image_bytes, format='PNG')
    return image_bytes.getvalue()

@app.post("/generate")
async def generate(request: ImageRequest):
    image_data = generate_image(request.prompt, request.steps, request.guide)
    image_b64 = base64.b64encode(image_data).decode()  # encode in base64 and convert to string
    return {"image": image_b64}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a image generation server.')
    parser.add_argument('--port', type=int, default=9000,
                        help='The port to start the server on.')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
