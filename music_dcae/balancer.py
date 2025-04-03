import torch
from torch.autograd import grad


class Balancer:
    """
    Balancer for dynamically re-weighting multiple losses based on gradient norms.

    Args:
        weights (dict): Predefined weights for each loss. 
                        Example: {"mse_loss": 1.0, "adv_loss": 1.0}
        ema_decay (float): Decay factor for exponential moving average (default: 0.99).
        epsilon (float): Small value to avoid division by zero (default: 1e-8).
    """
    def __init__(self, weights, ema_decay=0.99, epsilon=1e-8):
        self.weights = weights
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.ema_values = {key: 0.0 for key in weights}  # Initialize EMA for each loss

    def forward(self, losses, grad_inputs):
        """
        Re-weight the input losses based on gradient norms and return a combined loss.

        Args:
            losses (dict): Dictionary of losses with names as keys and loss tensors as values.
                           Example: {"mse_loss": mse_loss, "adv_loss": adv_loss}
            grad_inputs (dict): Dictionary of inputs for autograd.grad corresponding to each loss.
                                Example: {"mse_loss": recon_mels, "adv_loss": recon_mels}

        Returns:
            torch.Tensor: Combined weighted loss.
        """
        # Validate inputs
        if set(losses.keys()) != set(grad_inputs.keys()):
            raise ValueError("Keys of losses and grad_inputs must match.")
        
        norm_values = {}

        # Compute gradient norms for each loss
        for name, loss in losses.items():
            loss_grad, = grad(loss.mean(), [grad_inputs[name]], create_graph=True)
            dims = tuple(range(1, loss_grad.ndim))  # Exclude batch dimension
            grad_norm = torch.linalg.vector_norm(loss_grad, ord=2, dim=dims).mean()

            # Update EMA for the gradient norm
            if self.ema_values[name] == 0.0:
                self.ema_values[name] = grad_norm.item()
            else:
                self.ema_values[name] = (
                    self.ema_values[name] * self.ema_decay + grad_norm.item() * (1 - self.ema_decay)
                )

            # Normalize gradient norm
            norm_values[name] = grad_norm / (self.ema_values[name] + self.epsilon)

        # Compute dynamic weights
        total_norm = sum(norm_values.values())
        dynamic_weights = {name: norm / total_norm for name, norm in norm_values.items()}

        # Combine losses with dynamic weights
        loss = 0.0
        log_weights = {}
        for name in losses:
            loss = loss + self.weights[name] * dynamic_weights[name] * losses[name]
            log_weights[f"{name}_weight"] = dynamic_weights[name]
        return loss, log_weights


if __name__ == "__main__":
    # Example usage
    mel_real = torch.randn(1, 80, 10)
    generator = torch.nn.Linear(10, 10)
    recon_mels = generator(mel_real)
    discriminator = torch.nn.Linear(10, 1)
    disc_out = discriminator(recon_mels)

    mse_loss = torch.nn.functional.mse_loss(recon_mels, mel_real).mean()
    adv_loss = torch.nn.functional.softplus(-disc_out).mean()
    losses = {"mse_loss": mse_loss, "adv_loss": adv_loss}
    grad_inputs = {"mse_loss": recon_mels, "adv_loss": recon_mels}
    print("losses", losses)
    # Define predefined weights for each loss
    weights = {"mse_loss": 1.0, "adv_loss": 1.0}

    # Initialize balancer
    balancer = Balancer(weights)

    # Forward pass
    loss, log_weights = balancer.forward(losses, grad_inputs)
    print("Combined Loss:", loss)
    print("Dynamic Weights:", log_weights)
