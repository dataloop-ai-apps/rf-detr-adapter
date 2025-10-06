FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.11_cuda11.8_pytorch2

USER root

RUN apt-get update && apt-get install -y curl

# Create directory and set ownership in one step
RUN mkdir -p /tmp/app && chown 1000:1000 /tmp/app
RUN mkdir -p /tmp/app/weights && chown 1000:1000 /tmp/app/weights

USER 1000

# Download weights
RUN wget -O /tmp/app/weights/rf-detr-base-coco.pth https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth
RUN wget -O /tmp/app/weights/rf-detr-base-2.pth https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth
RUN wget -O /tmp/app/weights/rf-detr-large.pth https://storage.googleapis.com/rfdetr/rf-detr-large.pth

# Install the rest using default PyPI
RUN pip install \
    dtlpy \
    git+https://github.com/roboflow/rf-detr.git \
    git+https://github.com/dataloop-ai-apps/dtlpy-converters \
    numpy==1.26.4


