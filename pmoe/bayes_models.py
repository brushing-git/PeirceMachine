import numpy as np
import torch
import hamiltorch as ht
import torchbnn as bnn
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from tqdm import tqdm
from copy import deepcopy
from .utils import identity_fn, set_device

class BayesLinearRegression():
    def __init__(self, rvs: int, likelihood_sig: float, Phi: list=[identity_fn], 
                 rnd_seed=0) -> None:

        # Probability values
        self.prior_mu = np.zeros(rvs)
        self.prior_cov = np.eye(rvs, rvs)
        self.posterior_mu = self.prior_mu
        self.posterior_cov = self.prior_cov
        self.likelihood_sig = likelihood_sig
        self.generator = np.random.default_rng(rnd_seed)

        # Design Matrix
        self.Phi = Phi # Phi is a list of transformations

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        design_mat = np.column_stack([phi(X) for phi in self.Phi])
        return design_mat

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Compute design matrix
        design_mat = self.design_matrix(x)
        # Ensure mu is the right shape
        mu = self.posterior_mu.reshape(-1,1)
        mu = design_mat @ mu
        # We want independent samples drawn for making predictions (since our data is i.i.d.) so we take the diagonal
        sig = np.diagonal(design_mat @ self.posterior_cov @ design_mat.T) + (self.likelihood_sig ** 2)
        # Numpy using the standard deviation so we take the square root of the variance
        y = self.generator.normal(loc=mu.flatten(), scale=np.sqrt(sig))
        return y
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Params
        design_mat = self.design_matrix(X)
        inverse_prior_cov = np.linalg.inv(self.prior_cov)
        neg_likelihood_sig = self.likelihood_sig ** -2

        # Compute posteriors
        precision_mat = inverse_prior_cov + (neg_likelihood_sig) * (design_mat.T @ design_mat)
        posterior_cov = np.linalg.inv(precision_mat)
        posterior_mu = posterior_cov @ (inverse_prior_cov @ self.prior_mu + (neg_likelihood_sig) * (design_mat.T @ Y))

        # Store posteriors
        self.posterior_mu = posterior_mu
        self.posterior_cov = posterior_cov

class BayesLogisticRegressionSGHMC(nn.Module):
    """
    Bayesian Logistic Regression that learns via SG-HMC.

    Attributes:
    c : float : friction term for the SG-HMC
    layer : nn.Linear : linear layer of (in_features, 1)
    activation : nn.Sigmoid : sigmoid activation
    prior : MultivariateNormal : MultivariateNormal distribution N(0,I)
    criterion : nn.BCELoss : Binary Cross Entropy Loss for probabilities
    device : torch.device : device

    Methods:
    forward : makes a prediction based on model parameters
    predict_prob : makes a prediction based on ensemble members
    _update_params : updates the model parameters per SG-HMC as found in Chen 2014
    eval : evaluates the model on a test set
    fit : trains the model via SG-HMC
    """
    def __init__(self, 
                 in_features: int, 
                 alpha: float = 0.9, 
                 weight_decay: float = 5e-4,
                 temperature: float = 1.0) -> None:
        super(BayesLogisticRegressionSGHMC, self).__init__()

        # Device
        self.device = set_device()

        # Ensembles
        self.ensembles = []

        # Hyperparameters
        self.alpha = alpha
        self.temperature = temperature
        self.weight_decay = weight_decay

        # Weights and activation
        self.model = nn.Sequential(nn.Linear(in_features=in_features, out_features=1, bias=False),
                                   nn.Sigmoid())

        # Criterion
        self.criterion = nn.BCELoss()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.model(x)
        return out
    
    def predict_prob(self, x: torch.tensor) -> torch.tensor:
        if len(self.ensembles) == 0:
            raise Exception('No ensembles saved.')
        return sum(model(x) for model in self.ensembles) / len(self.ensembles)
    
    def _log_prior(self) -> float:
        log_prob = 0
        for p in self.parameters():
            flat_p = p.view(-1)
            sum_squares = torch.sum(flat_p ** 2)
            log_prob += -0.5 * sum_squares - 0.5 * len(flat_p) * torch.log(torch.tensor(2 * torch.pi))
        
        return log_prob
    
    def _adjust_learning_rate(self, k: int, epoch: int) -> float:
        return (k+epoch)**-1.0

    def _update_params(self, lr: float, epoch: int, burn_in: int, dataset_size: int) -> None:
        for p in self.parameters():
            # Create the momentum if it doesn't exist
            if not hasattr(p, 'r'):
                p.r = torch.randn(p.size()).to(self.device)
            # Get gradients
            d_p = p.grad.data

            # Apply weight decay
            d_p.add_(p.data, alpha=self.weight_decay)

            # Update weights
            p.data.add_(p.r)

            # Update the momentum
            r_new = (1 - self.alpha)*p.r - lr*d_p

            # Add the noise if past the burn-in
            if epoch > burn_in:
                eps = torch.randn(p.size()).to(self.device)
                r_new += (2.0*lr*self.alpha*self.temperature/dataset_size)**0.5 * eps
            
            p.r = r_new
    
    def _save_ensemble(self) -> None:
        model_copy = deepcopy(self.model)
        self.ensembles.append(model_copy)

    def evaluate(self, te_loader, ensemble=False) -> tuple:
        self.eval()
        self.to(self.device)
        val_loss = []
        val_acc = []

        with torch.no_grad():
            for x, y in iter(te_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                if ensemble:
                    y_hat = self.predict_prob(x)
                else:
                    y_hat = self.forward(x)

                # Calculate loss and accuracy
                loss = self.criterion(y_hat, y).detach().item()
                val_loss.append(loss)
                acc = ((y_hat > 0.5).float() == (y > 0.0)).float().mean().item()
                val_acc.append(acc)
            
            avg_loss = np.mean(val_loss)
            avg_acc = np.mean(val_acc)
        
        return avg_loss, avg_acc
                

    def fit(self, tr_loader, te_loader, k: int, epochs: int, epoch_to_save: int) -> None:
        self.to(self.device)

        # Clear the ensembles
        self.ensembles = []
        
        for step in range(epochs):
            self.train()

            train_loss = []
            for x, y in iter(tr_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Adjust the learning rate
                lr = self._adjust_learning_rate(k=k, epoch=step)

                # Zero the grad
                self.zero_grad()

                # Forward pass
                y_hat = self.forward(x)

                # Compute the prior loss
                dataset_size = len(tr_loader.dataset)
                prior_loss = -dataset_size ** -1 * self._log_prior()

                # Compute the likelihood loss
                likelihood_loss = x.size(0)**-1 * self.criterion(y_hat, y)

                # Update loss and backward
                loss = likelihood_loss + prior_loss
                loss.backward()

                # Clip the gradients to prevent an explosion
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # Update the weights
                self._update_params(lr, epoch=step, burn_in=epoch_to_save, dataset_size=dataset_size)

                # Add the loss
                train_loss.append(loss.item())
            
            te_loss, te_acc = self.evaluate(te_loader)
            avg_train_loss = np.mean(train_loss)

            # Save the ensemble
            if step > epoch_to_save:
                self._save_ensemble()

            if step % 5 == 0:
                s = (f'Metrics for epoch {step} are:\n tr_loss: {avg_train_loss}, te_loss: {te_loss}, te_acc: {te_acc}')
                print(s)
        
        # Convert the ensemble list to a module list
        self.ensembles = nn.ModuleList(self.ensembles)        

class BayesLogisticRegressionHMC(nn.Module):
    """
    Samples from a Bayesian Logistic Regression using full HMC through the Hamiltorch package.

    Attributes:

    """
    def __init__(self, in_features: int) -> None:
        super(BayesLogisticRegressionHMC, self).__init__()

        # Device
        self.device = set_device()

        # Weights and activation
        self.layer = nn.Linear(in_features=in_features, out_features=1, bias=False)
        self.activation = nn.Sigmoid()

        # Form the prior
        tau_list = []
        for w in self.parameters():
            tau_list.append(1.)
        tau_list = torch.tensor(tau_list).to(self.device)
        
        self.tau_list = tau_list
        self.params_hmc = None
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.layer(x)
        return out
    
    def predict(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred_list, _ = ht.predict_model(self, 
                                        self.params_hmc, 
                                        x=x, 
                                        y=y, 
                                        model_loss='binary_class_linear_output', 
                                        tau_out=1., 
                                        tau_list=self.tau_list)
        avg_pred = torch.mean(pred_list, 0)
        return avg_pred
    
    def evaluate(self, x: torch.tensor, y: torch.tensor) -> float:
        x = x.to(self.device)
        y = y.to(self.device)
        params_hmc = self.params_hmc

        pred_list, _ = ht.predict_model(self, 
                                        params_hmc, 
                                        x=x, 
                                        y=y, 
                                        model_loss='binary_class_linear_output', 
                                        tau_out=1., 
                                        tau_list=self.tau_list)
        
        _, pred = torch.max(pred_list, 2)
        acc = []
        for i in range(pred.shape[0]):
            a = (pred[i].float() == y.flatten()).sum().float() / y.shape[0]
            a = a.item()
            acc.append(a)
        
        return sum(acc) / len(acc)

    def fit(self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            n_samples: int, 
            step_size: float, 
            n_steps_per_sample: int,
            burn: int=0) -> None:
        
        x = x.to(self.device)
        y = y.to(self.device)

        ht.set_random_seed(0)
        params_init = ht.util.flatten(self).to(self.device).clone()
        self.params_hmc = ht.sample_model(self, 
                                        x, 
                                        y, 
                                        params_init=params_init,
                                        model_loss='binary_class_linear_output',
                                        num_samples=n_samples,
                                        step_size=step_size,
                                        num_steps_per_sample=n_steps_per_sample,
                                        burn=burn,
                                        tau_out=1.,
                                        tau_list=self.tau_list)

class BayesianLogisticRegressionVI(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(BayesianLogisticRegressionVI, self).__init__()

        # Device
        self.device = torch.device('cpu')

        # Weights and activation
        self.layer = bnn.BayesLinear(prior_mu=0.0, 
                                     prior_sigma=1.0, 
                                     in_features=in_features, 
                                     out_features=1,
                                     bias=False)
        self.activation = nn.Sigmoid()

        # Loss functions
        self.criterion = nn.BCELoss()
        self.kl_loss = bnn.loss.BKLLoss()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.activation(self.layer(x))
        return out
    
    def predict_prob(self, x: torch.tensor, n_samples: int) -> torch.tensor:
        batch_size = x.size(0)

        preds_sum = torch.zeros(batch_size, 1).to(self.device)

        for _ in range(n_samples):
            logits = self.forward(x)
            preds_sum += logits
        
        avg_pred = preds_sum / n_samples
        return avg_pred
    
    def evaluate(self, te_loader, n_samples: int) -> tuple:
        self.eval()
        self.to(self.device)
        val_loss = []
        val_acc = []

        with torch.no_grad():
            for x, y in iter(te_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                y_hat = self.predict_prob(x, n_samples)
                
                # Calculate loss and accuracy
                loss = self.criterion(y_hat, y).detach().item()
                val_loss.append(loss)
                acc = ((y_hat > 0.5) == (y > 0.0)).int()
                val_acc.append(acc)
            
            avg_loss = np.mean(val_loss)
            avg_acc = torch.cat(val_acc).float().mean().item()
        
        return avg_loss, avg_acc

    def fit(self, 
            tr_loader, 
            te_loader, 
            epochs: int, 
            lr: float, 
            t: float=1.0, 
            n_samples: int = 1) -> None:
        self.to(self.device)

        optim_fn = torch.optim.Adam(self.parameters(), lr=lr)

        for step in range(epochs):
            self.train()
            train_loss = []

            for x, y in iter(tr_loader):
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.forward(x)

                # Calculate loss
                likelihood = self.criterion(y_hat, y)
                kl = self.kl_loss(self)
                loss = likelihood + t*kl

                # Zero the grad
                optim_fn.zero_grad()
                loss.backward()

                # Clip the gradients to prevent a gradient explosion
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # Step the optimizer
                optim_fn.step()

                # Add the loss
                train_loss.append(loss.item())
            
            te_loss, te_acc = self.evaluate(te_loader, n_samples)
            avg_train_loss = np.mean(train_loss)

            if step % 5 == 0:
                s = (f'Metrics for epoch {step} are:\n tr_loss: {avg_train_loss}, te_loss: {te_loss}, te_acc: {te_acc}')
                print(s)