FROM nvidia/cuda:8.0-runtime-ubuntu16.04

LABEL maintainer "alex"

# Install some basic utilities
# install python
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
            aptitude git vim make wget zip zsh pkg-config \
            build-essential checkinstall p7zip-full python-pip \
            python3-pip tmux ffmpeg i7z unrar htop cmake g++  \
            curl libopenblas-dev python-numpy python3-numpy \
            python python-tk idle python-pmw python-imaging \
            libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
            libtbb2 libtbb-dev  libdc1394-22-dev libavcodec-dev  \
            libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev \
            gfortran && \
    apt-get autoremove && \
    apt-get clean && \
    aptitude install -y python-dev && \
    # update pip and setuptools
    pip install --upgrade pip setuptools
# Create a working directory

WORKDIR /data

RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
RUN pip install torchvision

RUN pip install opencv-python

RUN pip install pretrainedmodels --no-deps

RUN pip install tqdm munch

RUN apt-get update && apt-get install nano python-qt4 -y

RUN apt-get update && apt-get install libavdevice-dev libavfilter-dev -y

COPY . .

RUN cd /data/utils/PyAV && python setup.py install

RUN pip install scikit-learn catboost


###############################################
# 以上部分用户可定制，以下部分不可删除
###############################################
# 项目构建
#WORKDIR /data
#COPY . .

# 依赖软件安装
RUN pip install -r requirements.txt -i http://pypi.douban.com/simple  --trusted-host pypi.douban.com

