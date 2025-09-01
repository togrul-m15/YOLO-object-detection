import os
import time
import cv2
from ultralytics import YOLO

#COCO class names
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

#load of YOLOv5 model
model = YOLO('yolov5n.pt')
print("[INFO] YOLOv5 model loaded")

#inference for a single image
def infer_image_yolo(input_path, save_path=None, conf_thresh=0.5):
    image = cv2.imread(input_path)
    assert image is not None, f"Could not load image {input_path}"

    results = model(image, conf=conf_thresh)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{class_names[cls_id]}: {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, image)
        print(f"[IMG] Saved -> {save_path}")
    cv2.imshow("YOLOv5 Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#inference for a single video
def infer_video_yolo(input_path, save_path=None, conf_thresh=0.5):
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), f"Cannot open video {input_path}"

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    print(f"[INFO] Processing video: {input_path} -> {save_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model(frame, conf=conf_thresh)
        dt = (time.time() - t0) * 1000

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{class_names[cls_id]}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Inference: {dt:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer:
            writer.write(frame)
        cv2.imshow("YOLOv5 Video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[VID] Saved -> {save_path}")

#main processing logic
def main():
    video_inputs = [
        ("IMG_1656.mov", "lamine_yamal1.mov"),
        ("IMG_1659.mov", "lamine_yamal2.mov"),
        ("IMG_2654.mov", "lamine_yamal3.mov"),
        ("IMG_2656.mov", "lamine_yamal4.mov"),
        ("IMG_6309.mov", "lamine_yamal5.mov")
    ]

    image_inputs = [
        ("toga1.jpg", "toga_output1.jpg"),
        ("toga2.jpg", "toga_output2.jpg"),
        ("toga_mastered_photography.jpg", "toga_mastered_photography_output1.jpg"),
        ("toga_mastered_malikoskyy1.jpg", "toga_mastered_malikoskyy_output1.jpg"),
        ("toga_maga1.jpg", "toga_maga_output1.jpg"),
    ]

    video_output_dir = "./video_output"
    image_output_dir = "./photo_output"
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    conf_thresh = 0.5

    # Process videos
    for input_path, output_name in video_inputs:
        save_path = os.path.join(video_output_dir, output_name)
        infer_video_yolo(input_path, save_path, conf_thresh)

    # Process images
    for input_path, output_name in image_inputs:
        save_path = os.path.join(image_output_dir, output_name)
        infer_image_yolo(input_path, save_path, conf_thresh)

if __name__ == "__main__":
    main()
