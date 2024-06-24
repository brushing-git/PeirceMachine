import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .utils import set_device

class ExpertLinearRegression(nn.Module):
    def __init__(self, in_features: int, std: float = 0.1) -> None:
        super().__init__()

        self.layer = nn.Linear(in_features=in_features, out_features=1, bias=False)
        self.std = std
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.layer(x)
        return out
    
    def predict_prob(self, x: torch.tensor) -> torch.tensor:
        mu = self.layer(x)
        out = torch.normal(mean=mu, std=self.std)
        return out

class ExpertLogisticRegression(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()

        self.layer = nn.Linear(in_features=in_features, out_features=1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.activation(self.layer(x))
        return out

class NoisyTopKRouter(nn.Module):
    def __init__(self, dim_model: int, n_experts: int, top_k: int) -> None:
        super(NoisyTopKRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(dim_model, n_experts)
        self.noise_linear = nn.Linear(dim_model, n_experts)
    
    def forward(self, x) -> tuple:
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)

        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noise_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))

        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        out = F.softmax(sparse_logits, dim=-1)
        return out, indices

class SparseMoE(nn.Module):
    def __init__(self, n_experts: int, in_features: int, top_k: int, expert) -> None:
        super().__init__()

        self.experts = nn.ModuleList([expert(in_features) for _ in range(n_experts)])
        self.router = NoisyTopKRouter(dim_model=in_features, n_experts=n_experts, top_k=top_k)
        self.top_k = top_k
    
    def forward(self, x: torch.tensor, predict_prob: bool = False) -> torch.tensor:
        # Route the weights
        routing_weights, indices = self.router(x)
        final_out = torch.zeros(x.shape[0], 1, device=x.device)

        # Reshape for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_out = routing_weights.view(-1, routing_weights.size(-1))

        # Process experts in parallel
        for i, expert in enumerate(self.experts):
            expert_in = flat_x
            if predict_prob and callable(getattr(expert, "predict_prob", None)):
                expert_out = expert.predict_prob(expert_in)
            else:
                expert_out = expert(expert_in)

            # Gating scores
            gating_scores = flat_gating_out[:, i].unsqueeze(1)
            weighted_out = expert_out * gating_scores

            # Update the final out
            final_out += weighted_out
        
        return final_out

class MoELinearRegression(nn.Module):
    def __init__(self, n_experts: int, in_features: int, top_k: int = 2) -> None:
        super().__init__()

        # The primary predictor
        self.sparse_moe = SparseMoE(n_experts=n_experts, in_features=in_features, top_k=top_k, expert=ExpertLinearRegression)

        # Criterion
        self.criterion = nn.MSELoss()

        # The device
        self.device = set_device()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.sparse_moe(x)
        return out
    
    def predict(self, x: torch.tensor) -> torch.tensor:
        out = self.sparse_moe(x, predict_prob=True)
        return out

    def evaluate(self, te_loader) -> float:
        self.eval()
        self.to(self.device)

        val_loss = []

        with torch.no_grad():
            for x, y in iter(te_loader):
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y)
                val_loss.append(loss.item())
        
        avg_loss = np.mean(val_loss)
        return avg_loss
    
    def fit(self, tr_loader, te_loader, epochs: int, lr: float) -> None:
        self.to(self.device)

        # Optimizer
        optim_fn = torch.optim.Adam(params=self.parameters(), lr=lr)

        # Learning rate decay
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_fn, gamma=0.75, last_epoch=-1)

        for step in range(epochs):
            self.train()

            train_loss = []

            for x, y in iter(tr_loader):
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y)
                optim_fn.zero_grad()
                loss.backward()
                optim_fn.step()
                
                train_loss.append(loss.item())
            
            te_loss = self.evaluate(te_loader=te_loader)
            tr_loss = np.mean(train_loss)

            # Step the scheulder
            lr_scheduler.step()

            # Print the result every 5 steps
            if step % 10 == 0:
                current_lr = optim_fn.param_groups[0]['lr']
                s = (f'Metrics for epoch {step} are:\n tr_loss: {tr_loss}, te_loss: {te_loss}, current_lr: {current_lr}.')
                print(s)

class MoELogisticRegression(nn.Module):
    def __init__(self, n_experts: int, in_features: int, top_k: int = 2) -> None:
        super().__init__()

        # The primary predictor
        self.sparse_moe = SparseMoE(n_experts=n_experts, in_features=in_features, top_k=top_k, expert=ExpertLogisticRegression)

        # Criterion
        self.criterion = nn.BCELoss()

        # The device
        self.device = set_device()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.sparse_moe(x)
        return out
    
    def evaluate(self, te_loader) -> float:
        self.eval()
        self.to(self.device)

        val_loss = []
        val_acc = []
        with torch.no_grad():
            for x, y in iter(te_loader):
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y)
                val_loss.append(loss.item())
                acc = ((y_hat > 0.5).float() == (y > 0.0)).float().mean().item()
                val_acc.append(acc)
        
        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        return avg_loss, avg_acc
    
    def fit(self, tr_loader, te_loader, epochs: int, lr: float) -> None:
        self.to(self.device)
        optim_fn = torch.optim.Adam(params=self.parameters(), lr=lr)

        for step in range(epochs):
            self.train()

            train_loss = []

            for x, y in iter(tr_loader):
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y)
                optim_fn.zero_grad()
                loss.backward()
                optim_fn.step()
                
                train_loss.append(loss.item())
            
            te_loss, te_acc = self.evaluate(te_loader=te_loader)
            tr_loss = np.mean(train_loss)

            # Print the results every 5 steps
            if step % 10 == 0:
                s = (f'Metrics for epoch {step} are:\n tr_loss: {tr_loss}, te_loss: {te_loss}, te_acc: {te_acc}.')
                print(s)        