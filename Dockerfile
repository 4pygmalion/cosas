FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y python3 python3-pip libgl1 libglib2.0-0 git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && apt-get install -y gcc python3 python3-pip libgl1-mesa-glx libglib2.0-0 git

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR

COPY ./cosas/ $PROJECTDIR/cosas
COPY ./inference.py $PROJECTDIR
COPY ./requirements.txt $PROJECTDIR
COPY model.pth $PROJECTDIR
COPY setup.py $PROJECTDIR

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN bash ./install_spams.sh
RUN python3 -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels
RUN python3 -m pip install SimpleITK
RUN python3 -m pip install .


RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

LABEL team="SMF"
LABEL maintainer1="SMF_hoheon <hoheon0509@mf.seegene.com>"
LABEL maintainer2="SMF_wonchan <wonchanjeong53@gmail.com>"
ENTRYPOINT ["python3", "inference.py"]