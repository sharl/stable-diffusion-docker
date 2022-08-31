#!/usr/bin/env python
import datetime
import random
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from bottle import route, run


def isodatetime():
    return datetime.datetime.now().isoformat()


# prepare
model_name = "CompVis/stable-diffusion-v1-4"
DEVICE = 'cuda'
dtype, rev = (torch.float32, "main")

print("load pipeline start:", isodatetime())

with open("token.txt") as f:
    token = f.read().replace("\n", "")
pipe = StableDiffusionPipeline.from_pretrained(
    model_name, torch_dtype=dtype, revision=rev, use_auth_token=token
).to(DEVICE)

print("loaded models after:", isodatetime(), flush=True)


def render(prompt, samples=1, height=512, width=512, steps=50, scale=7.5):
    seed = random.randint(1, 2 ** 31)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with autocast(DEVICE):
        images = pipe(
            [prompt] * samples,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        )

    print("loaded images after:", isodatetime())

    for i, image in enumerate(images["sample"]):
        iname = prompt.replace(" ", "_")
        image.save(
            "output/%s__steps_%d__scale_%0.2f__seed_%d__n_%d.png"
            % (iname, steps, scale, seed, i + 1)
        )

    print("completed pipeline:", isodatetime(), flush=True)


@route('/')
def hello():
    return 'Hello World!'


@route('/<text>')
def main(text):
    print('text', text)
    render(text, steps=20)


run(host='0.0.0.0', port=8080, debug=True)
