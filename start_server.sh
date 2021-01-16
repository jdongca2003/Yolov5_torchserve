CUR_DIR=`pwd`
export PATH=${CUR_DIR}/jdk-11/bin:$PATH
export JAVA_HOME=${CUR_DIR}/jdk-11
export PATH=${JAVA_HOME}/bin:$PATH

mkdir -p workspace
#change tensorServer temp dir
export TEMP=${CUR_DIR}/workspace

torchserve --start --model-store model_store --models yolo5=yolo5.mar --ts-config ./config.properties
