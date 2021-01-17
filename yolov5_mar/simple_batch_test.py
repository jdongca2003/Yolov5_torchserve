import torch
from PIL import Image
import sys
import time
from hubconf import create_model

def load_image_list(filename):
    file_lists=[]
    with open(filename, 'rt') as f:
        for file_path in f:
            file_path = file_path.strip()
            file_lists.append(file_path)
    return file_lists

if __name__ == '__main__':
    #yolov5s, yolov5m, yolov5l, yolov5x
    model_name = sys.argv[1]
    #one image path per line
    image_list_file = sys.argv[2]

    pretrained_model_file = "{}.pt".format(model_name)
    model = create_model(model_name, pretrained_model_file)
    #model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    batch_size = 10
    image_files = load_image_list(image_list_file)
    num_images = len(image_files)
    prog_start = time.time()
    for i in range(0, num_images, batch_size):
        batch = image_files[i:(i+batch_size)]
        images=[]
        for image_path in batch:
            img = Image.open(image_path)
            images.append(img)
        results = model(images)
        print(i)

    prog_end = time.time()
    total_elapsed_time = prog_end - prog_start
    print('Throughput(#images/sec): {}'.format(num_images/float(total_elapsed_time)) )
