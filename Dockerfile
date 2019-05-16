FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER danil328

# Install Python 3.6
RUN apt-get update && \
    apt-get install -y python3.6 python3.6-dev

# Link Python to Python 3.6
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

# Install PIP
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Python packages
RUN pip --no-cache-dir install \
        colorlover \
        h5py \
        keras \
        matplotlib \
        numpy \
        pandas \
        scikit-image \
        scipy \
        sklearn \
        opencv-python \
        tqdm \
        kekas \
        albumentations \
        torch \
        torchvision

RUN pip --no-cache-dir install --upgrade python-dateutil==2.6.1

# Install Python 3.6 extra packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.6-tk

# Clean up commands
RUN rm -rf /root/.cache/pip/* && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /usr/local/src/*

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Environment Variables
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

COPY ./ /root
VOLUME /output
VOLUME /test

CMD ["python3"]