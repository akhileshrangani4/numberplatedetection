import cv2
import numpy as np

def preprocess_image(image, input_size=(416, 416)):
    # Resize image
    resized = cv2.resize(image, input_size)
    
    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0
    
    # Transpose dimensions for PyTorch (C, H, W)
    transposed = np.transpose(normalized, (2, 0, 1))
    
    print(f"Preprocessed image shape: {transposed.shape}")
    print(f"Preprocessed image min: {transposed.min()}, max: {transposed.max()}")
    
    return transposed

def draw_boxes(image, boxes, labels, confidence_threshold=0.5):
    for box, label in zip(boxes, labels):
        if box[4] >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            class_id = int(label)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Plate: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image
