# client.py
import httpx
import asyncio
import argparse
import logging
from PIL import Image
import io
import os
import base64

output_file_number = 0

async def request_images(server_address, port, args):
    global output_file_number
    async with httpx.AsyncClient(timeout=None) as client:  # disable timeout
        while True:
            response = await client.post(f"http://{server_address}:{port}/generate", 
                                         json={"prompt": args.prompt, "steps": args.steps, "guide": args.guide})
            image_data = base64.b64decode(response.json()["image"])  # decode base64 data
            image = Image.open(io.BytesIO(image_data))
            image.save(os.path.join(args.outdir, f"{output_file_number}.png"))
            output_file_number += 1

def read_server_list(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    return [line.strip().split(":") for line in lines]

async def main(args):
    server_list = read_server_list(args.servers)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    tasks = []
    for server_address, port in server_list:
        tasks.append(request_images(server_address, int(port), args))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Request a generated image from the server.')
    parser.add_argument('--servers', type=str, default="servers.txt",
                        help='The filename containing the server addresses and ports.')
    parser.add_argument('--prompt', type=str, default="An astronaut riding a horse on a planet made of cheese",
                        help='The prompt for the image generation.')
    parser.add_argument('--steps', type=int, default=100,
                        help='The number of steps to use for the image generation.')
    parser.add_argument('--guide', type=float, default=7.5,
                        help='The guide value for the image generation.')
    parser.add_argument('--outdir', type=str, default="images",
                        help='The subfolder to save the images.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main(args))
