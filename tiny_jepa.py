import torch
import torch.nn as nn

class TinyJEPA(nn.Module):
    def __init__(self, input_dim=10, latent_dim=16, future_dim=5):
        super().__init__()
        # Context Encoder (Encoder A)
        self.context_enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Target Encoder (Encoder B) - No gradient updates here!
        self.target_enc = nn.Sequential(
            nn.Linear(future_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Initialize Target Encoder with Context Encoder's weights
        # Note: If input_dim != future_dim, this load_state_dict will raise a size mismatch error.
        # Ensure they are the same if you want to initialize them identically.
        if input_dim == future_dim:
            self.target_enc.load_state_dict(self.context_enc.state_dict())
            
        # Freeze target encoder
        for param in self.target_enc.parameters():
            param.requires_grad = False

    def forward(self, past, future):
        z_c = self.context_enc(past)
        z_t = self.target_enc(future) # Target embedding
        z_hat = self.predictor(z_c)   # Predicted embedding
        return z_hat, z_t.detach()    # We don't backprop through Target Encoder

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
        """
        EMA update for the target encoder.
        target_params = momentum * target_params + (1 - momentum) * context_params
        """
        for param_q, param_k in zip(self.context_enc.parameters(), self.target_enc.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
