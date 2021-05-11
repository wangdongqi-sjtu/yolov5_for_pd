# YOLO v5 for PD Project
This is a YOLO v5 Detector used for PD Project. 

Instead of cloing the [original repo](https://github.com/ultralytics/yolov5), the model in this repo is loaded by `torch.hub` module. Please ensure you can connect to Internet before the first use.

## Clone Repo
```shell
git clone https://github.com/wangdongqi-sjtu/yolov5_for_pd.git
cd yolov5_for_pd
```
## Before You Start
Start from a **Python>=3.8** environment with **PyTorch>=1.7** installed, as well as **pyyaml>=5.3** for reading YOLOv5 configuration files. 
To install PyTorch see <https://pytorch.org/get-started/locally>. To install YOLOv5 requirements:
```shell
$ pip install -r ./requirements.txt
```

## Intialize the Detector
The YOLO v5 detector is wrapped as a `nn.Module` class named `DET_YOLO` in `./det_model/yolo.py`.

When instance `DET_YOLO` class, the initial parameters includes:

- **model (str)**:        Choose the YOLO v5 model. Defaults to `'yolov5s'`. To be specific, `'yolov5s'` is the lightest and fastest YOLOv5 model. For details on all available models, please see
                          [Readme.md](https://github.com/ultralytics/yolov5#pretrained-checkpoints.)
- **single_img (bool)**:  If True, only reveive a single image as input. In this case, 
                                return the results as a tensor. Otherwise list. Defaults to True.

- **img_resize (int)**:   Resize scale of input. Defaults to 640.

- **only_people (bool)**: If True, only detect people. Defaults to True.

- **conf (float)**:       Conference threshold (0-1). Defaults to None (using the default iou threshold).

- **iou (float)**:        IoU threshold (0-1). Defaults to None (using the default iou threshold).

- **verbose (bool)**:     If True, show the verbose info of model. Defaults to False.

- **pt_path(str)**:       Path which stores the model pt. *Change it before using this class.*



## How to Use
⚠**NOTES:** Please change the default `pt_path` in `DET_YOLO` according to your workspace at first.

After proper instializing and creating the YOLO detector, you can use it just like any `torch.Module`.
```python
det_model = DET_YOLO(model='yolov5s') 
```
The input is image(s) loaded by PIL or OpenCV.

When `single_img==True`, the output is a tensor. Otherwise the output is a list. Every element in this list is a tensor, representing the result of an image.

The format of every prediction item is: `[xmin, ymin, xmax, ymax, confidence, class]`

`example.py` shows how to create the yolo detector and embed it in your own code.

## Reference
This repository is modified on the base of [Load YOLOv5 from PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36).

You can refer to it for more details about model settings.
