FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2

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

# Add this to solve this issue :
# File "/tmp/.local/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1955, in __getattr__
#   module = self._get_module(self._class_to_module[name])
# File "/tmp/.local/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1969, in _get_module
#   raise RuntimeError(
# RuntimeError: Failed to import transformers.models.bloom.modeling_bloom because of the following error (look up to see its traceback):
# module 'torch' has no attribute 'compiler'

# Install PyTorch from custom index
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Install the rest using default PyPI
RUN pip install \
    dtlpy \
    git+https://github.com/roboflow/rf-detr.git \
    git+https://github.com/dataloop-ai-apps/dtlpy-converters \
    numpy==1.26.4


