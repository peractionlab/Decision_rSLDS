import scipy.io as sio
import numpy as np
import dill
import train, evaluate
from scipy.stats import wilcoxon

## Training parameters
predict_len = 10  # Number of prediction steps ahead
latent_dim = 20   # Dimension of the latent space
model_num = 40    # Number of models to train
pool_size = 20    # Size of the pool for model selection

DIM = '5'         # Dimension of the observed data
S = '0_05'        # Standard deviation of the noise

# _, DIM, S = sys.argv  # Uncomment this line to take command line arguments

def split_data(data):
    """
    Splits the data into training, validation, and test sets.
    
    Parameters:
    data (numpy.ndarray): The input data to be split.
    
    Returns:
    tuple: A tuple containing the training, validation, and test sets.
    """
    n = data.shape[0]
    train = data[:int(n * 0.7)]
    val = data[int(n * 0.7):int(n * 0.8)]
    test = data[int(n * 0.8):]
    return train, val, test


if __name__ == "__main__":
    path_prefix = "/home/jamie/Code/rSLDS/synth/"  # Path to the project directory
    data = sio.loadmat(path_prefix + "data/gaussian/ObsDim" + DIM + "_Q0_01_S" + S + ".mat")
    Xs = data["Xs"]  # Observed data
    Zs = data["Zs"]  # Latent states
    Ys = data["Ys"]  # Output data
    C_true = data["C"]  # True transformation matrix
    d_true = data["d"].squeeze()  # True bias vector

    # Split the data into training, validation, and test sets
    x_train, x_val, x_test = split_data(Xs)
    z_train, z_val, z_test = split_data(Zs)
    y_train, y_val, y_test = split_data(Ys)

    # Convert y_train, y_val, y_test to lists
    y_train = list(y_train)
    y_val = list(y_val)
    y_test = list(y_test)
    
    # %%
    lds_path = path_prefix + "models/gaussian/ObsDim" + DIM + "_Q0_01_S" + S + "_K1/"
    rslds_path = path_prefix + "models/gaussian/ObsDim" + DIM + "_Q0_01_S" + S + "_K3/"
    
    # Train models and evaluate them
    train.train_models(lds_path, y_train, latent_dim, 1, 10, 50, pool_size)
    train.train_models(rslds_path, y_train, latent_dim, 3, model_num, 150, pool_size)

    lds_id = evaluate.select_best_model(lds_path, y_val, 10, pool_size)
    rslds_id = evaluate.select_best_model(rslds_path, y_val, model_num, pool_size)
    
    import time
    t = time.time()
    _, each_eR2s_K1, each_R2s_K1, each_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, False, x_test, C_true, d_true)
    print("LDS time cost = ", time.time() - t)
    _, each_eR2s_K3, each_R2s_K3, each_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, False, x_test, C_true, d_true)
    print("rSLDS time cost = ", time.time() - t)

    _, train_elbos_K1, test_elbos_K1, eR2s_K1, R2s_K1, maes_K1 = evaluate.get_across_trial_evaluation(lds_path, lds_id, y_test, True, False, x_test, C_true, d_true)
    print("Gaussian LDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f | MAE = %.3f" %
          (train_elbos_K1, test_elbos_K1, R2s_K1[0], eR2s_K1[0], np.mean(each_R2s_K1[:, 0]), maes_K1[0]))
    
    _, train_elbos_K3, test_elbos_K3, eR2s_K3, R2s_K3, maes_K3 = evaluate.get_across_trial_evaluation(rslds_path, rslds_id, y_test, True, False, x_test, C_true, d_true)
    print("Gaussian 3-rSLDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f | MAE = %.3f" %
          (train_elbos_K3, test_elbos_K3, R2s_K3[0], eR2s_K3[0], np.mean(each_R2s_K3[:, 0]), maes_K3[0]))

    with open(lds_path + str(lds_id) + ".dill", 'rb') as f:
        lds = dill.load(f)

    with open(rslds_path + str(rslds_id) + ".dill", 'rb') as f:
        rslds = dill.load(f)
    print("Best LDS score = %.3f" % evaluate.evaluate_inferred_dynamic(lds, C_true, d_true))
    print("Best rSLDS score = %.3f" % evaluate.evaluate_inferred_dynamic(rslds, C_true, d_true))
    
    # Compute the p value between outcomes from lds and rslds model
    test_elbos_K1, best_eR2s_K1, best_R2s_K1, best_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, False, x_test, C_true, d_true)
    test_elbos_K3, best_eR2s_K3, best_R2s_K3, best_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, False, x_test, C_true, d_true)
    
    p_R2 = []
    p_mae = []
    for i in range(predict_len):
        best_K1 = [x[i] for x in best_R2s_K1]
        best_K3 = [x[i] for x in best_R2s_K3]
        stat, p = wilcoxon(best_K1, best_K3)
        p_R2.append(p)
        
        best_K1 = [x[i] for x in best_maes_K1]
        best_K3 = [x[i] for x in best_maes_K3]
        stat, p = wilcoxon(best_K1, best_K3)
        p_mae.append(p)
        
    print("p value of R^2 results: ", p_R2)
    print("p value of MAE results: ", p_mae)
    
