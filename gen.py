#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import requests
from urllib.parse import quote


text = quote(' '.join(sys.argv[1:]))
requests.get('http://localhost:8080/' + text, timeout=30)
