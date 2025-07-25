import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g. "best.pt")')
parser.add_argument('--source', required=True, help='Source: image/video file, folder, usb0, picamera0, or screen1')
parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH (e.g. 1280x720)')
parser.add_argument('--record', action='store_true', help='Save output video (requires resolution)')
args = parser.parse_args()

# Parse args
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Validate model path
if not os.path.exists(model_path):
    print('ERROR: Invalid model path.')
    sys.exit(1)

# Load model
model = YOLO(model_path)
labels = model.names

# Detect source type
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
elif 'screen1' in img_source:
    source_type = 'screen1'
else:
    print(f'Invalid source: {img_source}')
    sys.exit(1)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.lower().split('x'))

# Setup recording
if record:
    if source_type not in ['video','usb','screen1']:
        print('Recording only supports video/usb/screen1')
        sys.exit(1)
    if not user_res:
        print('Recording requires --resolution')
        sys.exit(1)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH))

# Init source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()
elif source_type == 'screen1':
    sct = mss()
    monitor = sct.monitors[1]  # Main screen
    if user_res:
        monitor = {"top": 0, "left": 0, "width": resW, "height": resH}
    else:
        resW, resH = monitor['width'], monitor['height']

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Main loop
while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Finished all images.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print('Video ended or camera error.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
    elif source_type == 'screen1':
        frame = np.array(sct.grab(monitor))[..., :3]

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    if source_type in ['video', 'usb', 'picamera', 'screen1']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO Detection', frame)
    if record: recorder.write(frame)

    key = cv2.waitKey(5 if source_type != 'image' else 0)
    if key in [ord('q'), ord('Q')]: break
    elif key in [ord('s'), ord('S')]: cv2.waitKey()
    elif key in [ord('p'), ord('P')]: cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']: cap.release()
elif source_type == 'picamera': cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
