# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY infer.py /opt/ml/code/infer.py

COPY requirements.txt .

RUN pip install --no-cache-dir pip --upgrade && pip install --no-cache-dir -r requirements.txt

# Defines cifar10.py as script entrypoint 
ENV SAGEMAKER_PROGRAM infer.py
