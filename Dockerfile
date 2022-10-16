FROM tensorflow/tensorflow:latest-gpu

RUN pip install --upgrade pip diffusers==0.5.1 pillow scipy torch==1.12.1+cu116 transformers==4.23.1 ftfy \
  --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install bottle requests

RUN useradd -m huggingface

USER huggingface

WORKDIR /home/huggingface

RUN mkdir -p /home/huggingface/.cache/huggingface \
  && mkdir -p /home/huggingface/output

COPY docker-entrypoint.py /usr/local/bin
COPY token.txt /home/huggingface

ENV PYTHONUNBUFFERED 1
EXPOSE 8080
ENTRYPOINT [ "docker-entrypoint.py" ]
