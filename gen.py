#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import requests
from urllib.parse import quote

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--skip', action='store_true')
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--timeout', type=int, default=30)
parser.add_argument('args', nargs='+', default='')
args = parser.parse_args()
skip = '?nsfw' if args.skip else ''

prompt = quote(' '.join(args.args))
if args.seed:
    requests.get('http://localhost:8080/{}/{}/{}{}'.format(prompt, args.steps, args.seed, skip), timeout=args.timeout)
else:
    requests.get('http://localhost:8080/{}/{}{}'.format(prompt, args.steps, skip), timeout=args.timeout)
