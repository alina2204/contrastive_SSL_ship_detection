FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN echo Acquire::http::proxy "http://apt.theresis.org:3142"; > /etc/apt/apt.conf.d/01proxy
RUN apt update && apt install -y python3-pip git libsm6 libxext6 cython3 libxrender-dev
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install torch==1.8.1 torchvision==0.9.1 
RUN pip3 install lightly==1.1.4 pytorch-lightning==1.1.8 matplotlib scikit-learn mlflow boto3 scikit-learn scipy torchsat pytorch-ignite seaborn scikit-image
# RUN pip3 install pycocotools==2.0.2

RUN echo export DIS_VIS=1 >> /root/.bashrc
RUN echo export LIGHTLY_SERVER_LOCATION=localhsot >> /root/.bashrc
