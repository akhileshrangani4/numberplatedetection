import torch
import yaml
from models import YOLOModel
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import preprocess_image
from dataset import CustomDataset, custom_collate_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim

def train_model(data_yaml, model_config, epochs, batch_size=16, learning_rate=0.001):
    # Load data configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Initialize model
    model = YOLOModel(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define separate loss functions for classification and bounding box regression
    cls_criterion = nn.BCEWithLogitsLoss()
    bbox_criterion = nn.MSELoss(reduction='none')

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define data augmentation
    train_transform = A.Compose([
        A.RandomResizedCrop(416, 416),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Load dataset
    train_dataset = CustomDataset(data_config['train'], data_config['train'].replace('images', 'labels'), transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_images, batch_labels, batch_bboxes in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_bboxes = batch_bboxes.to(device)

            optimizer.zero_grad()
            cls_outputs, bbox_outputs = model(batch_images)
            
            # Prepare target tensors
            batch_size, num_predictions, _ = cls_outputs.shape
            target_cls = torch.zeros_like(cls_outputs)
            target_bbox = torch.zeros_like(bbox_outputs)
            
            for i in range(batch_size):
                num_objects = (batch_labels[i] >= 0).sum()
                target_cls[i, :num_objects, batch_labels[i][:num_objects].long()] = 1
                target_bbox[i, :num_objects] = batch_bboxes[i, :num_objects]
            
            # Calculate classification loss
            cls_loss = cls_criterion(cls_outputs, target_cls)
            
            # Calculate bounding box regression loss
            bbox_loss = bbox_criterion(bbox_outputs, target_bbox)
            
            # Create a mask for valid bounding boxes
            valid_bbox_mask = (target_bbox.sum(dim=-1) > 0).float()
            
            # Apply the mask to the bounding box loss
            bbox_loss = (bbox_loss.mean(dim=-1) * valid_bbox_mask).sum() / (valid_bbox_mask.sum() + 1e-6)
            
            # Combine losses
            loss = cls_loss + bbox_loss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
    print("Training completed. Model saved as 'trained_model.pth'")

if __name__ == "__main__":
    data_yaml = "../config/data.yaml"
    model_config = "../config/model_config.yaml"
    epochs = 100
    train_model(data_yaml, model_config, epochs)
