import onnx
import torch
from models.experimental import End2End, attempt_load
from utils.torch_utils import select_device

device = 'cpu'
weights = './weights/yolov5l.pt'
data = './data/coco128.yaml'
dnn = False
path = './weights/yolov5l.onnx'

im = torch.randn((1, 3, 640, 640))
device = select_device(device)
model = attempt_load(weights, map_location=device)
model.eval()

Tensorrt = False
max_wh = None if Tensorrt else 640
dynamic = True
dynamic_batch = False
if dynamic:
    input_names = ['images']
    output_names = ['output']
    dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                    'output': {0: 'batch', 2: 'y', 3: 'x'}}

if dynamic_batch:
    input_names = ['images']
    dynamic_axes = {
        'images': {
            0: 'batch',
        }, }
    if Tensorrt:
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        output_axes = {
            'num_dets': {0: 'batch'},
            'det_boxes': {0: 'batch'},
            'det_scores': {0: 'batch'},
            'det_classes': {0: 'batch'},
        }
    else:
        output_names = ['output']
        output_axes = {
            'output': {0: 'batch'},
        }
    dynamic_axes.update(output_axes)

print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if Tensorrt is None else 'onnxruntime')
model = End2End(model, max_obj=1000, iou_thres=0.45, score_thres=0.45, max_wh=max_wh, device=device, n_classes=2)

torch.onnx.export(model=model,
                  f=path,
                  args=im,
                  verbose=False,
                  opset_version=12,
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=dynamic_axes)

onnx_model = onnx.load(path)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model

import onnxruntime
device = select_device(device)
if device == 'cpu':
    providers = ['CPUExecutionProvider']
else:
    providers = ['CUDAExecutionProvider']
session = onnxruntime.InferenceSession(path, providers=providers)
print('------')