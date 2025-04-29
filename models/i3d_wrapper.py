import torch
import torch.nn as nn
from models.pytorch_i3d import InceptionI3d

class I3DWrapper(nn.Module):
    """Wrapper for I3D that handles temporal dimension automatically"""
    
    def __init__(self, num_classes, pretrained_path=None):
        super(I3DWrapper, self).__init__()
        
        # Initialize the base I3D model
        self.i3d = InceptionI3d(num_classes=400, in_channels=3)
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.i3d.load_state_dict(torch.load(pretrained_path))
        
        # Replace logits layer for our task
        self.i3d.replace_logits(num_classes)
        
    def forward(self, x):
        # Pass through base I3D model
        out = self.i3d(x)
        
        # Handle temporal dimension explicitly - if output has shape [batch, class, time]
        if out.dim() == 3:
            # Average over time dimension
            out = out.mean(dim=2)
        
        return out
