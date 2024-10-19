import torch
from models import YOLOModel
from dataset import CustomDataset
from torch.utils.data import DataLoader
from utils import preprocess_image
import yaml

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_detections = 0
    total_detections = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy (you may need to adjust this based on your specific output format)
            predicted = outputs.argmax(dim=1)
            correct_detections += (predicted == labels).sum().item()
            total_detections += labels.numel()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_detections / total_detections
    return avg_loss, accuracy

if __name__ == "__main__":
    with open("config/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    model = YOLOModel("config/model_config.yaml")
    model.load_state_dict(torch.load("trained_model.pth"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_dataset = CustomDataset(data_config['val'], data_config['val'].replace('images', 'labels'), transform=preprocess_image)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    loss, accuracy = evaluate_model(model, val_loader, device)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

