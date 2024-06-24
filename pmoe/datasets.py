import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

gen = np.random.default_rng(12345)

class NumpyDataset(Dataset):
    def __init__(self, X, Y, transform=torch.from_numpy):
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
            y = torch.tensor(y).unsqueeze(-1)
            x = x.type(torch.float32)
            y = y.type(torch.float32)
        return x, y

def build_linear_dataset(lower_bound: float, 
                         upper_bound: float, 
                         slopes: list, 
                         intercept: float, 
                         n_samples: int, 
                         std_dev: float=1.0) -> np.ndarray:
    """
    Builds a multi-variate dataset for linear regression.  Includes the default feature for the intercept.

    Params:
    lower_bound : float : the lower bound for the dataset
    upper_bound : float : the upper bound for the dataset
    slopes : list : list of slope values
    intercept : float : intercept value
    n_samples : int : number of datapoints to build
    std_dev : float=1.0 : the variance around the line

    Returns:
    data : np.ndarray : a (n_samples, len(slopes)+2) array of generated data points
    """
    # Build the slope-intercept formulation
    m = len(slopes)
    slopes = np.array(slopes)
    slopes = slopes.reshape(m, 1)
    slopes = np.vstack((slopes, np.array([intercept])))
    slope_intercepts = lambda x: x @ slopes + gen.normal(loc=0., scale=std_dev)
    
    # Build the features and the labels
    features = gen.uniform(lower_bound, upper_bound, size=(n_samples, m))
    int_feat = np.ones((n_samples, 1))
    features = np.hstack((features, int_feat))
    labels = slope_intercepts(features)

    # Put the features and the data together
    data = np.hstack((features, labels))

    return data

def build_unipolynomial_dataset(lower_bound: float,
                             upper_bound: float,
                             degree: int,
                             n_samples: int,
                             std_dev: float=1.0,
                             rescale: bool = False,
                             coefficients: list = None) -> tuple:
    """
    Builds a univariate dataset for polynomial regression.

    Params
    lower_bound : float : the lower bound for the dataset
    upper_bound : float : the upper bound for the dataset
    degree : int : the degree of polynomial
    n_samples : int : number of datapoints to build
    std_dev: float=1.0 : the variance around the polynomial
    rescale : bool=False : rescale the data; only use if doing large polynomials and large upper/lower bounds

    Returns
    data: np.ndarray : a (n_samples, 2) array of generated datapoints
    coefficients : np.ndarray : a (degree+1) array of polynomial cofficients
    """

    # Compute the initial set of real features
    X = gen.uniform(lower_bound, upper_bound, size=(n_samples, 1))
    if coefficients:
        coefficients = np.array(coefficients[:degree+1])
    else:
        coefficients = gen.normal(loc=0.0, scale=10.0, size=degree + 1)

    # Do the polynomial
    Y = np.polyval(coefficients, X)

    # Rescale so that we do not have extreme values
    if rescale:
        data_scale = scale(np.hstack((X,Y)))
        X, Y = data_scale[:,:-1], np.expand_dims(data_scale[:,-1], axis=1)
    
    # Add some noise
    noise = gen.normal(loc=0.0, scale=std_dev, size=(n_samples, 1))
    Y += noise

    # Add the dummy feature for bias
    int_feat = np.ones((n_samples, 1)) # For the bias
    X = np.hstack((X, int_feat))

    # Stack the data
    data = np.hstack((X, Y))
    return data, coefficients

def build_vc_dataset(lower_bound: float,
                     upper_bound: float,
                     n_samples: int, 
                     features: int,
                     degree: int) -> np.ndarray:
    """
    Builds a dataset to test VC-dimension.  Samples from a n degree polynomial and then classifies the 
    samples based on whether they fall above the polynomial or below.

    We specify the third dimension of the point in terms of a normal distribution to be close to the 
    decision boundary:  this increases the difficulty of the problem.

    The VC-dimension is exactly degree+1 of the polynomial.

    Params:
    lower_bound : float : the lower bound for the dataset
    upper_bound : float : the upper bound for the dataset
    n_samples : int : the number of samples to be generated
    features : int : the number of features to be generated, must be greater than 1

    Returns:
    data : np.ndarray : dataset of features and polynomial labels with shape (n_samples, features+2)
    """

    # Make sure the number of features is greater than 1
    if features < 2:
        raise ValueError('Features must greater than 1.')

    # Generate the features used to produce the polynomial
    X = gen.uniform(lower_bound, upper_bound, size=(n_samples, features-1))

    # Generate the coefficients
    coefficients = gen.normal(loc=0.0, scale=10.0, size=degree + 1)

    # Initialize values for the polynomial
    poly_values = np.zeros((n_samples, features-1))

    # Compute the polynomial values
    for i in range(features-1):
        poly_values[:,i] = np.polyval(coefficients, X[:,i])
    poly_values = np.sum(poly_values, axis=1)

    # Re-expand
    poly_values = np.expand_dims(poly_values, axis=1)

    # Add the last feature to be normally distributed around the polynomial
    std_dev = np.std(poly_values)
    noise = gen.normal(loc=0.0, scale=std_dev, size=(n_samples, 1))
    last_feature = poly_values + noise
    X = np.hstack((X, last_feature))

    # Scale the data to deal with the outlier volume
    data_scale = np.hstack((X, poly_values))
    data_scale = scale(data_scale)

    # Threshold the values to yield the classification
    Y = (data_scale[:,-2] > data_scale[:,-1]).astype(float).reshape(n_samples, 1)

    # Add the dummy feature for the bias
    int_feat = np.ones((n_samples, 1))
    X = data_scale[:,:-1] # bring in the scaled features
    X = np.hstack((X, int_feat))

    # Build the data
    data = np.hstack((X,Y))

    return data

def build_trte_dataloader(dataset: np.ndarray, te_size=0.2) -> tuple:
    X = dataset[:,:-1]; Y = dataset[:,-1]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=te_size)

    tr_data = NumpyDataset(X_train, Y_train)
    te_data = NumpyDataset(X_test, Y_test)

    tr_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    te_loader = DataLoader(te_data, batch_size=64, shuffle=True)

    return tr_loader, te_loader