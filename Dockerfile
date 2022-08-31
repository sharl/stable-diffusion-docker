FROM tensorflow/tensorflow:latest-gpu

RUN pip install diffusers pillow torch transformers \
  --extra-index-url https://download.pytorch.org/whl/cu116
## RUN sed -i -e '/has_nsfw_concept in/,+9d' /usr/local/lib/python3.8/dist-packages/diffusers/pipelines/stable_diffusion/safety_checker.py
RUN pip install bottle

RUN useradd -m huggingface

USER huggingface

WORKDIR /home/huggingface

RUN mkdir -p /home/huggingface/.cache/huggingface \
  && mkdir -p /home/huggingface/output

COPY docker-entrypoint.py /usr/local/bin
COPY token.txt /home/huggingface

EXPOSE 8080
ENTRYPOINT [ "docker-entrypoint.py" ]
