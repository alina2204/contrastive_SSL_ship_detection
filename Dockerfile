RUN apt update && apt install -y python3-pip git libsm6 libxext6 cython3 libxrender-dev
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install torch==1.8.1 torchvision==0.9.1 
RUN pip3 install lightly==1.1.4 pytorch-lightning==1.1.8 matplotlib scikit-learn mlflow boto3 scikit-learn scipy torchsat pytorch-ignite seaborn scikit-image

