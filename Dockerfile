FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install required packages and dependencies
RUN apt-get update && \
    apt-get install -y git wget && \
    pip install --upgrade pip

WORKDIR /app
COPY . /app

WORKDIR /app/nnUNet
RUN pip install -e .

WORKDIR /app

ENTRYPOINT ["python", "infer_nnunet.py"]