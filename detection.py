import cv2
import numpy as np

min_conf = 0.5
nms_thresh = 0.5


class Detector:
    def __init__(self, net, layer, clazz=0):
        self.net = net
        self.layer = layer
        self.clazz = clazz

    def detect(self, frame, clazz=None):
        if clazz is None:
            clazz = self.clazz

        (H, W) = frame.shape[:2]
        results = []
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layers = self.net.forward(self.layer)

        boxes = []
        centroids = []
        confs = []

        for lay in layers:
            for detection in lay:
                scores = detection[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if class_id == clazz and conf > float(min_conf):
                    box = detection[0:4] * np.array([W, H, W, H])
                    (cx, cy, width, height) = box.astype("int")

                    x = int(cx - (width / 2))
                    y = int(cy - (height / 2))

                    boxes.append([x, y, int(width), int(height)])

                    centroids.append((cx, cy))
                    confs.append(float(conf))

        idxs = cv2.dnn.NMSBoxes(
            boxes, confs, float(min_conf), float(nms_thresh))

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                r = (confs[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

        return results
