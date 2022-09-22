#!/usr/bin/env python3
import datetime
import hashlib
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from bottle import route, request, run
from PIL import Image
from io import BytesIO
import requests


def isodatetime():
    return datetime.datetime.now().isoformat()


# prepare
model_name = 'CompVis/stable-diffusion-v1-4'
model_name = 'hakurei/waifu-diffusion'
DEVICE = 'cuda'
# select your VRAM
dtype, rev = (torch.float32, "main")
dtype, rev = (torch.float16, "fp16")

with open("token.txt") as f:
    token = f.read().replace("\n", "")

# init text to image
pipe = StableDiffusionPipeline.from_pretrained(
    model_name, torch_dtype=dtype, revision=rev, use_auth_token=token
).to(DEVICE)
pipe.enable_attention_slicing()
safety_checker = pipe.safety_checker

# init image to image
img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_name, torch_dtype=dtype, revision=rev, use_auth_token=token
).to(DEVICE)
img_pipe.enable_attention_slicing()
img_safety_checker = img_pipe.safety_checker

print("loaded models after:", isodatetime(), flush=True)


def skip_safety_checker(images, *args, **kwargs):
    return images, False


def render(prompt, samples=1, height=512, width=512, steps=50, scale=7.5, seed=None, skip=False):
    print("start rendering    :", isodatetime(), flush=True)

    if seed is None:
        seed = torch.random.seed()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    if skip:
        pipe.safety_checker = skip_safety_checker
    else:
        pipe.safety_checker = safety_checker

    with autocast(DEVICE):
        images = pipe(
            [prompt] * samples,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        )

    print("ended rendering    :", isodatetime(), flush=True)

    s512 = hashlib.sha512(prompt.encode()).hexdigest()[:32]
    iname = bytes(prompt, 'utf-8')[:100].decode('utf-8', 'ignore').replace(' ', '_')
    for i, image in enumerate(images['images']):
        image.save(
            "output/%s__%s__steps_%d__scale_%0.2f__seed_%d__n_%d.png"
            % (s512, iname, steps, scale, seed, i + 1)
        )


def img2img(prompt, url=None, steps=50, seed=None, skip=False):
    print("start rendering    :", isodatetime(), flush=True)

    image = None
    if url:
        r = requests.get(url, timeout=30)
        if r and r.status_code == 200:
            image = Image.open(BytesIO(r.content)).convert('RGB')
            image = image.resize((512, 512))

    if seed is None:
        seed = torch.random.seed()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    if skip:
        img_pipe.safety_checker = skip_safety_checker
    else:
        img_pipe.safety_checker = img_safety_checker

    with autocast(DEVICE):
        images = img_pipe(
            prompt=prompt,
            init_image=image,
            strength=0.75,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator
        ).images

    print("ended rendering    :", isodatetime(), flush=True)

    s512 = hashlib.sha512(prompt.encode()).hexdigest()[:32]
    iname = bytes(prompt, 'utf-8')[:100].decode('utf-8', 'ignore').replace(' ', '_')
    images[0].save(
        'output/%s__%s__steps_%d__seed_%d.png' % (s512, iname, steps, seed)
    )


def check_nsfw():
    return 'nsfw' in dict(request.query)


@route('/')
def hello():
    return 'Hello World!'


@route('/<text>')
def main(text):
    print('text', text, flush=True)
    url = request.query.get('url')
    if url:
        img2img(text, url=url, steps=20, skip=check_nsfw())
    else:
        render(text, steps=20, skip=check_nsfw())


@route('/<text>/<steps:int>')
def main_steps(text, steps):
    print('text', text, flush=True)
    url = request.query.get('url')
    if url:
        img2img(text, url=url, steps=steps, skip=check_nsfw())
    else:
        render(text, steps=steps, skip=check_nsfw())


@route('/<text>/<steps:int>/<seed:int>')
def main_steps_seed(text, steps, seed):
    print('text', text, flush=True)
    url = request.query.get('url')
    if url:
        img2img(text, url=url, steps=steps, seed=seed, skip=check_nsfw())
    else:
        render(text, steps=steps, seed=seed, skip=check_nsfw())


run(host='0.0.0.0', port=8080, debug=True)
