import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import normalize

# SimCLR Class
class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128, temperature=0.07):
        """
        Initialize SimCLR model.
        
        Args:
        - backbone (nn.Module): The base neural network (e.g., ResNet).
        - projection_dim (int): Dimension of the projection head.
        - temperature (float): Temperature scaling for contrastive loss.
        """
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.temperature = temperature
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights for the projection head.
        """
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        """
        Forward pass for SimCLR.
        
        Args:
        - x (torch.Tensor): Input batch of images.
        
        Returns:
        - out (torch.Tensor): Projected output for contrastive loss.
        """
        features = self.backbone(x)
        out = self.projection_head(features)
        return normalize(out, dim=1)

    def contrastive_loss(self, z_i, z_j):
        """
        Contrastive loss function based on cosine similarity.
        
        Args:
        - z_i (torch.Tensor): Embedding of the first image in the pair.
        - z_j (torch.Tensor): Embedding of the second image in the pair.
        
        Returns:
        - loss (torch.Tensor): Computed contrastive loss.
        """
        similarity = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(z_i.size(0)).cuda()
        loss = nn.CrossEntropyLoss()(similarity, labels)
        return loss


# MoCo Class
class MoCo(nn.Module):
    def __init__(self, backbone, projection_dim=128, m=0.999):
        """
        Initialize MoCo model.
        
        Args:
        - backbone (nn.Module): The base neural network (e.g., ResNet).
        - projection_dim (int): Dimension of the projection head.
        - m (float): Momentum for the key encoder.
        """
        super(MoCo, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.m = m
        self._initialize_weights()

        # Initialize the key encoder as a copy of the backbone
        self.key_encoder = self._initialize_key_encoder()

    def _initialize_weights(self):
        """
        Initialize weights for the projection head.
        """
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def _initialize_key_encoder(self):
        """
        Initialize the key encoder with the same architecture as the backbone.
        """
        key_encoder = nn.ModuleList(self.backbone.children())
        return nn.Sequential(*key_encoder)

    def update_key_encoder(self):
        """
        Update the key encoder using momentum.
        """
        for param_k, param_q in zip(self.key_encoder.parameters(), self.backbone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x):
        """
        Forward pass for MoCo.
        
        Args:
        - x (torch.Tensor): Input batch of images.
        
        Returns:
        - out_q (torch.Tensor): Query projection.
        - out_k (torch.Tensor): Key projection.
        """
        out_q = self.projection_head(self.backbone(x))
        out_k = self.projection_head(self.key_encoder(x).detach())  # No gradients for key encoder
        return out_q, out_k

    def contrastive_loss(self, z_q, z_k):
        """
        Contrastive loss for MoCo.
        
        Args:
        - z_q (torch.Tensor): Query embeddings.
        - z_k (torch.Tensor): Key embeddings.
        
        Returns:
        - loss (torch.Tensor): Contrastive loss based on cosine similarity.
        """
        similarity = torch.matmul(z_q, z_k.T) / 0.07  # temperature scaling
        labels = torch.arange(z_q.size(0)).cuda()
        loss = nn.CrossEntropyLoss()(similarity, labels)
        return loss


# BYOL Class
class BYOL(nn.Module):
    def __init__(self, backbone, projection_dim=256, hidden_dim=4096):
        """
        Initialize BYOL model.
        
        Args:
        - backbone (nn.Module): The base neural network (e.g., ResNet).
        - projection_dim (int): The output dimension of the projection head.
        - hidden_dim (int): The dimension of the hidden layer in the projection head.
        """
        super(BYOL, self).__init__()
        self.backbone = backbone
        self.online_encoder = self._build_encoder(projection_dim, hidden_dim)
        self.target_encoder = self._build_encoder(projection_dim, hidden_dim)
        self._initialize_weights()

        # Initialize the target encoder with the same weights as the online encoder
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

    def _build_encoder(self, projection_dim, hidden_dim):
        """
        Build the projection encoder (both online and target encoders).
        
        Args:
        - projection_dim (int): The output dimension of the projection head.
        - hidden_dim (int): The dimension of the hidden layer.
        
        Returns:
        - encoder (nn.Module): The constructed encoder.
        """
        return nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def _initialize_weights(self):
        """
        Initialize the weights for the encoder heads.
        """
        for m in self.online_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        for m in self.target_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def update_target_encoder(self, beta=0.99):
        """
        Update the target encoder with the moving average of the online encoder's parameters.
        
        Args:
        - beta (float): Momentum factor for the moving average.
        """
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = beta * target_params.data + (1 - beta) * online_params.data

    def forward(self, x1, x2):
        """
        Forward pass for BYOL.
        
        Args:
        - x1 (torch.Tensor): First image in the pair.
        - x2 (torch.Tensor): Second image in the pair.
        
        Returns:
        - z1 (torch.Tensor): Embedding of the first image in the pair (from online encoder).
        - z2 (torch.Tensor): Embedding of the second image in the pair (from target encoder).
        """
        z1 = self.online_encoder(self.backbone(x1))
        z2 = self.target_encoder(self.backbone(x2).detach())  # Target encoder is frozen
        return z1, z2

    def loss(self, z1, z2):
        """
        Compute the loss function for BYOL.
        
        Args:
        - z1 (torch.Tensor): Embedding of the first image.
        - z2 (torch.Tensor): Embedding of the second image.
        
        Returns:
        - loss (torch.Tensor): The loss based on cosine similarity.
        """
        z1 = normalize(z1, dim=1)
        z2 = normalize(z2, dim=1)
        loss = 2 - 2 * (z1 * z2).sum(dim=1).mean()
        return loss


# Utility function for creating the base model (ResNet, etc.)
def create_backbone(base_model='resnet50', pretrained=True):
    """
    Create a backbone model (e.g., ResNet).

    Args:
    - base_model (str): The base model to use (e.g., 'resnet50').
    - pretrained (bool): Whether to use pretrained weights.

    Returns:
    - model (nn.Module): The backbone model.
    """
    if base_model == 'resnet50':
        model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=pretrained)
        model.fc = nn.Identity()  # Remove the final classification layer
    else:
        raise ValueError(f"Unsupported model type: {base_model}")
    return model
