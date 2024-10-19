import torch
import torch.nn as nn
import yaml

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOModel(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_classes = self.config['num_classes']
        self.backbone = self._create_backbone()
        self.neck = self._create_neck()

        # Determine the output size of the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 416, 416)
            backbone_output = self.backbone(dummy_input)
            self.backbone_out_channels = backbone_output.shape[1]
            self.output_size = backbone_output.shape[2:]

        self.cls_head = self._create_cls_head()
        self.bbox_head = self._create_bbox_head()

    def _create_backbone(self):
        layers = []
        in_channels = 3
        for layer_config in self.config['backbone']:
            if layer_config['type'] == 'conv':
                layers.append(ConvBlock(in_channels, layer_config['filters'], layer_config['kernel_size'], layer_config.get('stride', 1)))
                in_channels = layer_config['filters']
        return nn.Sequential(*layers)

    def _create_neck(self):
        return nn.Identity()

    def _create_cls_head(self):
        return nn.Conv2d(self.backbone_out_channels, self.num_classes, 1)

    def _create_bbox_head(self):
        return nn.Conv2d(self.backbone_out_channels, 4, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        cls_output = self.cls_head(features)
        bbox_output = self.bbox_head(features)

        batch_size, _, grid_h, grid_w = cls_output.shape
        cls_output = cls_output.permute(0, 2, 3, 1).contiguous().view(batch_size, grid_h * grid_w, self.num_classes)
        bbox_output = bbox_output.permute(0, 2, 3, 1).contiguous().view(batch_size, grid_h * grid_w, 4)

        return cls_output, bbox_output

def create_model(config_path):
    return YOLOModel(config_path)
