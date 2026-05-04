import torch
from tiny_jepa import TinyJEPA

def main():
    # Note: We set future_dim = input_dim = 10 so the state_dict copy works
    # If they are different, the target encoder must be initialized independently 
    # or the first layer excluded from the weight copy.
    model = TinyJEPA(input_dim=10, latent_dim=16, future_dim=10)
    
    # Dummy inputs: batch_size=32, input_dim=10, future_dim=10
    past = torch.randn(32, 10)
    future = torch.randn(32, 10)
    
    # Forward pass
    z_hat, z_t = model(past, future)
    
    print("Shape of predicted embedding (z_hat):", z_hat.shape)
    print("Shape of target embedding (z_t):", z_t.shape)
    
    # Dummy loss and backward pass
    loss = torch.nn.functional.mse_loss(z_hat, z_t)
    loss.backward()
    
    print("Loss:", loss.item())
    
    # EMA update
    # In a real training loop, this happens after optimizer.step()
    model.update_target_encoder(momentum=0.99)
    print("Successfully performed EMA update on the target encoder.")

if __name__ == "__main__":
    main()
