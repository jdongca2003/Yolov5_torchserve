#download the model from Ultralytics/yolov5/releases

model_name=yolov5x.pt
#if you use the following models, you need to update the value for key 'serializedFile' in MAR-INF/MANIFEST.json. 
#yolov5s.pt, yolov5m.pt, yolov5l.pt

wget https://github.com/ultralytics/yolov5/releases/download/v4.0/${model_name}

