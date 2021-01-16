import torch
from PIL import Image
from hubconf import create_model
import logging
import os
import time
import io
import base64

logger = logging.getLogger(__name__)

class YoloHandler(object):
    
    def __init__(self):
       self.model = None
       self.device = None
       self.initialized = False
       self.context = None
       self.manifest = None
       self.map_location = None
       self.explain = False

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        yolov5_model_name = os.path.splitext(serialized_file)[0]
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = create_model(yolov5_model_name, model_pt_path, channels=3, classes=80, autoshape=True)
        logger.info("Loading eager model")
        self.model.to(self.device)
        self.model.eval()

        logger.info('Model file %s loaded successfully', model_pt_path)
        self.initialized = True

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return images

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        data_preprocess = self.preprocess(data)

        if not self._is_explain():
            output = self.model(data_preprocess)
            output = self.postprocess(output)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output

    def _is_explain(self):
        return False

    def postprocess(self, data):
        result = []
        preds = data.pred
        class_names = data.names
        for batch_idx, pred in enumerate(preds):
            if pred is not None:
                retval=[]
                for *box, conf, cls in pred:  # xyxy, confidence, class
                    _retval = {}
                    xmin = float(box[0])
                    ymin = float(box[1])
                    xmax = float(box[2])
                    ymax = float(box[3])
                    _retval['boxes'] = [xmin, ymin, xmax, ymax]
                    _retval['score'] = float(conf)
                    _retval['class'] = class_names[int(cls)]
                    retval.append(_retval)
                result.append(retval)

        return result
