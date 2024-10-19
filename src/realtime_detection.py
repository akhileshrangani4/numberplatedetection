import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr

def visualize_preprocessed(preprocessed):
    if isinstance(preprocessed, np.ndarray):
        if preprocessed.shape[0] == 3:
            vis = np.transpose(preprocessed, (1, 2, 0))
        else:
            vis = preprocessed
    else:
        vis = preprocessed.squeeze().permute(1, 2, 0).cpu().numpy()
    
    vis = (vis - vis.min()) / (vis.max() - vis.min())
    vis = (vis * 255).astype(np.uint8)
    return vis

def preprocess_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def detect_license_plate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while keeping edges sharp
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Find edges
    edged = cv2.Canny(blur, 30, 200)
    
    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    license_plate = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            license_plate = approx
            break
    
    if license_plate is not None:
        x, y, w, h = cv2.boundingRect(license_plate)
        plate = image[y:y+h, x:x+w]
        return plate, (x, y, w, h)
    return None, None

def realtime_detect_and_read_plates(confidence_threshold=0.5):
    # Load the pre-trained YOLOv5 model for license plate detection
    model = YOLO('yolov8s.pt')  # or yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    cap = cv2.VideoCapture('data/raw/videos/TrafficControlCCTV.mp4')

    # Get original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect license plates
        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, confidence in zip(boxes, confidences):
                if confidence > confidence_threshold:
                    x1, y1, x2, y2 = box
                    plate_img = frame[y1:y2, x1:x2]

                    # Perform OCR on the plate image
                    try:
                        ocr_result = reader.readtext(plate_img)
                        if ocr_result:
                            plate_text = ocr_result[0][1]
                            print(f"Detected plate: {plate_text}")
                        else:
                            plate_text = ""
                    except Exception as e:
                        print(f"OCR error: {e}")
                        plate_text = ""

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Create a filled rectangle for text background
                    cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
                    
                    # Put OCR result and confidence on the image
                    cv2.putText(frame, f"{plate_text}", (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (x1 + 5, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow("Real-time License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_detect_and_read_plates(confidence_threshold=0.5)
