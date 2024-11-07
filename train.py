import numpy as np
from multiprocessing import get_context
import os
import ssm
import util

def trainer(y_train, latent_dim, K, path, id, max_iters):
    """
    Train a single model using the given training data.
    
    Parameters:
    - y_train: Training data.
    - latent_dim: Dimension of the latent space.
    - K: Number of discrete states.
    - path: Directory path to save the model.
    - id: Unique identifier for the model.
    - max_iters: Maximum number of iterations for training.
    """
    np.random.seed()
    model_path = path + str(id) + ".dill"
    obs_dim = np.max([tr.shape[-1] for tr in y_train])
    
    if not os.path.exists(model_path):
        # Define and initialize the model based on the type of observations
        if isinstance(y_train[0][0, 0], np.int64):
            # Poisson observations
            model = ssm.SLDS(N=obs_dim, K=K, D=latent_dim, transitions="recurrent_only",
                             dynamics="diagonal_gaussian",
                             emissions="poisson",
                             emission_kwargs=dict(link="softplus"))
        else:
            # Gaussian observations
            model = ssm.SLDS(N=obs_dim, K=K, D=latent_dim, transitions="recurrent_only",
                             dynamics="diagonal_gaussian",
                             emissions="gaussian")
        
        # Initialize the model with the training data
        model.initialize(y_train, num_init_iters=20)
        
        # Fit the model using the Laplace-EM algorithm
        elbos, q = model.fit(y_train, method="laplace_em", num_iters=max_iters, initialize=False)
    else:
        # Load the existing model if it already exists
        model, elbos, q = util.load_model(model_path)

    # Save the trained model
    util.save_model(model_path, model, elbos, q)


def train_models(path, y_train, latent_dim, K, model_num, max_iters, pool_size):
    """
    Train multiple models in parallel.
    
    Parameters:
    - path: Directory path to save the models.
    - y_train: Training data.
    - latent_dim: Dimension of the latent space.
    - K: Number of discrete states.
    - model_num: Number of models to train.
    - max_iters: Maximum number of iterations for training.
    - pool_size: Number of parallel processes to use.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    
    print("Training begins...")
    return_packs = []
    
    # Use multiprocessing to train models in parallel
    with get_context("spawn").Pool(processes=pool_size) as pool:
        for i in range(model_num):
            return_packs.append(pool.apply_async(trainer, args=(y_train, latent_dim, K, path, str(i), max_iters)))
        pool.close()
        pool.join()

    print("Training completed!")