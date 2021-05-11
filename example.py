import cv2
from PIL import Image
from det_model.yolo import DET_YOLO, results_numpy

det_model = DET_YOLO(model='yolov5s') # load the yolo model

input_img = cv2.imread('./img/bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
# input_img = Image.open('./img/zidane.jpg')  # PIL image

det_results = det_model(input_img) # detect

# tensor -> numpy
# det_results = results_numpy(det_results)

# show info
if det_model.single_img:
    print('\tDetecting using {}'.format(det_model.yolo_model))
    print('\tResults shape: {}, type: {}'.format(
        det_results.shape, type(det_results)))
    print('\tResults Format: [xmin, ymin, xmax, ymax, confidence, class]')
    print('\tDetection Results:\n', det_results)

