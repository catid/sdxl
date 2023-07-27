# SDXL Quick Start

This is a quick start guide for how to use SDXL 1.0 to play with the new models.

The goal is to try out all the cool new features to get the best possible quality out of a self-hosted image generation model:

+ 1024x1024 resolution.
+ Ensemble of expert denoisers, starting with the SDXL 1.0 base, and using the SDXL 1.0 refiner model for the last 20%.

The scripts also support multiple GPUs using a client/server model.

## Hardware

I'm using an Ubuntu Linux servers with 3x Nvidia RTX 3090 GPUs.

You can probably also limp along with Windows but everything is always harder to do on Windows.  Probably it's too hard to figure out on Windows and people will have to wait for developers to wrap it up in an easy-to-use package.  Typically I use Windows for my desktop OS and then use Tabby.sh and Visual Studio Code (with the Remote Development plugin) to do development on Linux servers.

## Download models

I'd suggest just downloading the models so you don't need to mess with tokens and stuff.  The scripts in this repo expect them to be downloaded.

```bash
sudo apt install git-lfs
git lfs install

mkdir -p ~/sdxl
cd ~/sdxl

git clone https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

## Setup and test

This assumes you have installed Conda: https://www.anaconda.com/download/

Set up environment and test the image generation:

```bash
cd ~/sdxl

# Note: 3.11 is not supported yet
conda create -n sdxl python=3.10
conda activate sdxl

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

python test_generate.py
```

## Multi-GPU/Multi-server

If you have multiple GPUs, you can use the `client.py` and `server.py` scripts to generate artwork in parallel.

On each server computer, run the setup instructions above.  Then for each GPU, open a separate terminal and run:

```bash
cd ~/sdxl

conda activate sdxl

CUDA_VISIBLE_DEVICES=0 python server.py --port 9000
```

Specify a different `--port` for each server.  Set the CUDA_VISIBLE_DEVICES=1 and so on for each different GPU.  Using tmux with a pane for each GPU is what I do.

Create a new text file `servers.txt` with the contents to match your network.  I have it set up like this:

```
gpu1.lan:9000
gpu1.lan:9001
gpu1.lan:9002
gpu2.lan:9000
gpu2.lan:9001
gpu2.lan:9002
```

Then open a new terminal to run the client:

```bash
cd ~/sdxl

conda activate sdxl

python client.py --servers servers.txt --outdir images --prompt "An astronaut riding a horse on a planet made of cheese" --guide 7.5 --steps 100

# The previous parameters are the defaults.  You can just type:
python client.py --prompt "An astronaut riding a horse on a planet made of cheese"
```

This will generate 1024x1024 images continuously until you close the client in a `images` subfolder.

## Discussion

To achieve full quality from an image generation server, 24 GB of VRAM seems required right now.  It uses about 15.3 GB VRAM for the first image, and then after that the usage goes up to 21.3 GB.

It uses almost no CPU, only GPU.

On the RTX 4090 both models generate at about 4 iterations per second, so 100 iterations takes about 25 seconds per image per GPU.
