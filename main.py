import cv2
import numpy as np
from point import Point
from tools import generate_detections as gdet

from sorting import preprocessing
from sorting import nn_matching
from sorting.detection import Detection
from sorting.tracker import Tracker

def load_yolo():
    #1

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if class_id != 0:
                continue
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def idPerson(boxes):
    x, y, w, h = boxes
    mx = (x+x+w)/2
    my = (y+y+h)/2
    p = Point(mx,my)
    return Point.getId(p)





def draw_labels(window,boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(idPerson(boxes[i]))
            color = colors[i]

            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow(window, img)

def webcam_detects():
    model, classes, colors, output_layers = load_yolo()
    cap1 = cv2.VideoCapture('videos/camera2.mp4')
    cap2 = cv2.VideoCapture('videos/camera1.mp4')

    while True:
        _, frame1 = cap1.read()
        ret , frame2 = cap2.read()
        height1, width1, channels1 = frame1.shape
        height2, width2, channels2 = frame2.shape
        blob1, outputs1 = detect_objects(frame1, model, output_layers)
        blob2, outputs2 = detect_objects(frame2, model, output_layers)
        boxes1, confs1, class_ids1 = get_box_dimensions(outputs1, height1, width1)
        boxes2, confs2, class_ids2 = get_box_dimensions(outputs2, height2, width2)
        draw_labels('camera1',boxes1, confs1, colors, class_ids1, classes, frame1)
        draw_labels('camera2',boxes2, confs2, colors, class_ids2, classes, frame2)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap1.release()
    cap2.release()

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness,text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2),color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id),(x1, y1+30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0,0,255),thickness=text_thickness)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap1 = cv2.VideoCapture('videos/camera2.mp4')
    nms_max_overlap = 0.4
    max_cosine_distance = 0.2
    nn_budget = None
    model_filename = 'mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)
    track_cnt = dict()
    frame_cnt = 0
    images_by_id = dict()
    ids_per_frame = []

    while True:

        _, frame1 = cap1.read()
        height1, width1, channels1 = frame1.shape
        blob1, outputs1 = detect_objects(frame1, model, output_layers)
        boxes1, confs1, class_ids1 = get_box_dimensions(outputs1, height1, width1)
        features = encoder(frame1, boxes1)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes1, features)]

        text_scale, text_thickness, line_thickness = get_FrameLabels(frame1)
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap,
                                                   scores)  # preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # length = len(indices)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        tmp_ids = []

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < height1 and bbox[2] < width1:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    images_by_id[track.track_id] = [frame1[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append(
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                    images_by_id[track.track_id].append(frame1[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            cv2_addBox(track.track_id, frame1, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), line_thickness,
                       text_thickness, text_scale)
        ids_per_frame.append(set(tmp_ids))



        frame_cnt += 1




        draw_labels('camera1',boxes1, confs1, colors, class_ids1, classes, frame1)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap1.release()

    # who is using this machine

    # tensorflow version
 # 1.15 need to instal and again check
if __name__ == '__main__':
    webcam_detect()