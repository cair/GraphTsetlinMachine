FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y python3-pip git

COPY . /app
WORKDIR /app


# You should have a requirements.txt to define your dependencies
RUN pip3 install .

CMD [ "tail", "-f", "/dev/null" ]