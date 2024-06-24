import numpy as np
import torchbnn as bnn
import torch

def kl_div(mu_0, sig_0, mu_1, sig_1, t) -> float:
  """
  Calculates the kl divergence with a temperature factor for 2 normal distributions.
  """

  pi = np.pi
  alpha = 0.5 * np.log(2*pi*(sig_1**2))
  gamma = t*((torch.log(2*pi*(sig_0**2)) + 1) * 0.5)
  beta = 0.5 * (((sig_0**2) + (mu_0 - mu_1)**2) / (sig_1**2))
  kl = alpha + beta - gamma
  return kl.sum()

def bayes_kl(model, t, device, reduction='mean', last_layer_only=False):
  kl = torch.Tensor([0]).to(device)
  kl_sum = torch.Tensor([0]).to(device)
  n = torch.Tensor([0]).to(device)

  for m in model.modules():
    if isinstance(m, (bnn.BayesLinear, bnn.BayesConv2d)):
      weight_mu = m.weight_mu
      weight_sig = torch.exp(m.weight_log_sigma)
      prior_mu = m.prior_mu
      prior_sig = np.exp(m.prior_log_sigma)

      kl = kl_div(weight_mu, weight_sig, prior_mu, prior_sig, t)
      kl_sum += kl
      n += len(weight_mu.view(-1))

      if m.bias:
        bias_mu = m.bias_mu
        bias_sig = torch.exp(m.bias_log_sigma)

        kl = kl_div(bias_mu, bias_sig, prior_mu, prior_sig, t)
        kl_sum += kl
        n += len(bias_mu.view(-1))

  if last_layer_only or n == 0 :
    return kl

  if reduction == 'mean' :
    return kl_sum/n
  elif reduction == 'sum' :
    return kl_sum
  else :
    raise ValueError(reduction + " is not valid")

def identity_fn(x: float) -> float:
    return x

def set_device():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device