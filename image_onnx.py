import cv2
import onnxruntime
from temporal.tracker import *
from utils.plots import Annotator, colors


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class Yolov5FireSmoke():
    def __init__(self, weights='weights/yolov5s.onnx', names=('smoke', 'fire'), img_size=640, confidence=0.6,
                 temporal='tracker', area_thresh=0.05, window_size=20, persistence_thresh=0.5, stride=32, auto=True):
        self.device = 'cpu'
        self.names = names
        self.img_size = img_size
        self.confidence = confidence
        self.stride = stride
        self.auto = auto
        self.temporal = temporal  # temporal analysis technique used after detection' 'persistence or tracker or None"
        self.area_thresh = area_thresh  # suppression threshold when temporal analysis technique is tracker
        self.window_size = window_size   # sliding window size for temporal analysis technique
        self.persistence_thresh = persistence_thresh

        # Load onnx model
        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider']
        self.session = onnxruntime.InferenceSession(weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [self.session.get_outputs()[i].name for i in range(len(self.session.get_outputs()))]

        # Temporal analysis technique
        if temporal == 'tracker':
            print('\nFire detector with tracker\n')
            self.tracker = ObjectTracker(area_thresh, window_size)
            self.log = Log()
            self.threshold = area_thresh
        elif temporal == 'persistence':
            print('\nFire detector with temporal persistence\n')
            self.threshold = persistence_thresh
            self.temporal_buffer = np.zeros((window_size))
            self.pos = 0
        else:
            print('\nFire detector without temporal analysis technique\n')
            self.threshold = None
            self.window_size = None

    def preprocess(self, img0):
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]  # Padded resize
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.array(img, dtype=np.float32)
        img /= 255.         # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, axis=0)
        return img

    def postprecess(self, pred, image0, img):
        det = []
        for i in range(pred.shape[0]):
            if pred[i, -1] >= self.confidence:      # compare with confidence
                det.append(pred[i])
        annotator = Annotator(image0, line_width=3, example=str(self.names))
        if len(det):
            det = np.asarray(det)
            det = det[:, 1:]
            # Rescale boxes from img_size to im0 size
            det[:, 0:4] = scale_coords(img.shape[2:], det[:, 0:4], image0.shape).round()

            if self.temporal == 'tracker':
                mask = np.ones((det.shape[0],), dtype=bool)
                for *xyxy, cls, conf in reversed(det):
                    # Bounding box coordinates
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    # Confidence score of bounding box
                    conf_score = conf
                    class_id = int(cls)
                    # Recording bounding box coordinates
                    self.log.coord_objs.append([xmin, ymin, xmax, ymax])
                    # Recording class identifier
                    self.log.classes.append(class_id)
                    # Recording confidence score
                    self.log.conf_scores.append(conf_score)

                # Object tracking
                centroids, areas = self.tracker.tracking(self.log.coord_objs)
                # Add to the log
                self.log.centroids, self.log.areas = self.log.update(centroids, areas)
                # Detections that should be suppressed
                idxs = self.tracker.bbox_suppression(self.log)

                # Bounding box suppression
                if len(idxs):
                    try:
                        mask[idxs] = False
                        det = det[mask]
                    except:
                        pass

                # Clear list of object coordinates per frame
                self.log.coord_objs.clear()
                self.log.classes.clear()
                self.log.conf_scores.clear()

            if self.temporal == 'persistence':

                mask = np.ones((det.shape[0],), dtype=bool)

                self.temporal_buffer[self.pos] = True
                self.pos = (self.pos + 1) % self.window_size

                if np.sum(self.temporal_buffer) <= (self.threshold * self.window_size):
                    try:
                        mask[:] = False
                        det = det[mask]
                    except:
                        pass

            # Write results
            for *xyxy, cls, conf in reversed(det):
                c = int(cls)  # integer class
                label = (f'{self.names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        im0 = annotator.result()
        cv2.imshow("fere detection", im0)
        cv2.waitKey(10000)  # 1 millisecond

    def run(self, image_path):
        image0 = cv2.imread(image_path)
        img = self.preprocess(image0)
        pred = self.session.run(self.output_names, {self.input_name: img})[0]
        self.postprecess(pred, image0, img)


if __name__ == "__main__":
    Detection = Yolov5FireSmoke(temporal=None)
    Detection.run(image_path='/media/ubuntu/DATA/DATASET/Fire_dataset/Smoke Detection for Fire Alarm dataset/initial_test/initial_test/initial_test/000001.jpg')
