mkdir torchServing_example
cd torchServing_example

git clone https://github.com/pytorch/serve.git
cd serve

conda create -n torchServe python=3.8
source activate torchServe

#Intall TorchServe and torch-model-archiver

#install dependencies
pip install -U -r requirements/torch_cu110_linux.txt
pip install -U -r requirements/productions.txt

#install torchserve and torch-model-archiver
pip install torchserve torch-model-archiver

#Install dependencies for yolov5 https://github.com/ultralytics/yolov5/blob/master/requirements.txt
pip install opencv-python==4.5.1.48
pip install seaborn==0.11.1
pip install pandas
cd ..

#Download openjdk 11 for linux
wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz
tar xfv openjdk-11+28_linux-x64_bin.tar.gz
