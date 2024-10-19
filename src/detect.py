import cv2
import torch
import pytesseract
from models import YOLOModel
from utils import preprocess_image, draw_boxes

def detect_and_read_plates(model, video_path, output_path, confidence_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        input_tensor = torch.from_numpy(preprocess_image(frame)).unsqueeze(0).to(device)

        with torch.no_grad():
            detections = model(input_tensor)

        boxes = detections[0].cpu().numpy()
        labels = detections[1].cpu().numpy()

        for box in boxes:
            if box[4] >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box[:4])
                plate_img = frame[y1:y2, x1:x2]
                
                # Perform OCR on the plate image
                plate_text = pytesseract.image_to_string(plate_img, config='--psm 7 --oem 3')
                
                # Draw bounding box and OCR result
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    video.release()
    out.release()
    print(f"Detection completed. Output saved to {output_path}")

if __name__ == "__main__":
    model = YOLOModel("config/model_config.yaml")
    model.load_state_dict(torch.load("trained_model.pth"))
    
    video_path = "path/to/test_video.mp4"
    output_path = "path/to/output_video.mp4"
    
    detect_and_read_plates(model, video_path, output_path)
