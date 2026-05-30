from ultralytics import YOLO
import cv2

INPUT = "data/sample_rubble.jpg"
OUT = "data/detect_out.jpg"

def run():
    model = YOLO("yolov8n.pt")
    results = model.predict(source=INPUT, save=False, imgsz=640, conf=0.25)

    img = cv2.imread(INPUT)
    boxes = results[0].boxes
    for box in boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,f"{cls}:{conf:.2f}",(x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.imwrite(OUT, img)
    print("Saved:", OUT)

if __name__ == '__main__':
    run()

