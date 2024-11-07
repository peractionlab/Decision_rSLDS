# %%
import scipy.io as sio
import numpy as np
import dill
import sys
import matplotlib.pyplot as plt
import util, train, evaluate, plotting
from scipy.stats import wilcoxon

## training parameters
predict_len = 10
latent_dim = 2
model_num = 20
pool_size = 5

DIM = '200'
RATE = '2'

# _, DIM, RATE = sys.argv


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


 # %%
if __name__ == "__main__":
    path_prefix = "/home/jamie/Code/rSLDS/synth/"   # use your own project path
    data = sio.loadmat(path_prefix + "data/poisson/ObsDim" + DIM + "_Q0_01_Rate" + RATE + ".mat")
    Xs = data["Xs"]
    Zs = data["Zs"]
    Ys = data["Ys"].astype(int)
    C_true = data["C"]
    d_true = data["d"].squeeze()

    x_train, x_val, x_test = split_data(Xs)
    z_train, z_val, z_test = split_data(Zs)
    y_train, y_val, y_test = split_data(Ys)

    r_train = np.log(1 + np.exp(z_train))
    r_val = np.log(1 + np.exp(z_val))
    r_test = np.log(1 + np.exp(z_test))
    
    y_train = list(y_train)
    y_val = list(y_val)
    y_test = list(y_test)
    
    lds_path = path_prefix + "models/poisson/ObsDim" + DIM + "_Q0_01_Rate" + RATE + "_K1/"
    rslds_path = path_prefix + "models/poisson/ObsDim" + DIM + "_Q0_01_Rate" + RATE + "_K3/"
    
    train.train_models(lds_path, y_train, latent_dim, 1, model_num, 20, pool_size)
    train.train_models(rslds_path, y_train, latent_dim, 3, model_num, 200, pool_size)

    lds_id = evaluate.select_best_model(lds_path, y_val, model_num, pool_size)
    rslds_id = evaluate.select_best_model(rslds_path, y_val, model_num, pool_size)
    lds_id = 9
    rslds_id = 13
        

    import time
    t = time.time()
    _, each_eR2s_K1, each_R2s_K1, each_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, False, x_test, C_true, d_true)
    print("LDS time cost = ", time.time() - t)
    _, each_eR2s_K3, each_R2s_K3, each_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, False, x_test, C_true, d_true)
    print("rSLDS time cost = ", time.time() - t)

    _, train_elbos_K1, test_elbos_K1, eR2s_K1, R2s_K1, maes_K1 = evaluate.get_across_trial_evaluation(lds_path, lds_id, y_test, True, False, x_test, C_true, d_true)
    print("Poisson LDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f / %.4f | MAE = %.3f" %
          (train_elbos_K1, test_elbos_K1, R2s_K1[0], eR2s_K1[0], np.mean(each_R2s_K1[:, 0]), np.mean(each_eR2s_K1[:, 0]), maes_K1[0]))
    
    _, train_elbos_K3, test_elbos_K3, eR2s_K3, R2s_K3, maes_K3 = evaluate.get_across_trial_evaluation(rslds_path, rslds_id, y_test, True, False, x_test, C_true, d_true)
    print("Poisson 3-rSLDS")
    print("Train ELBO = %.3f | Test ELBO = %.3f | across R2 = %.4f / %.4f | individual R2 = %.4f / %.4f | MAE = %.3f" %
          (train_elbos_K3, test_elbos_K3, R2s_K3[0], eR2s_K3[0], np.mean(each_R2s_K3[:, 0]), np.mean(each_eR2s_K3[:, 0]), maes_K3[0]))
    
    
    with open(lds_path + str(lds_id) + ".dill", 'rb') as f:
        lds = dill.load(f)

    with open(rslds_path + str(rslds_id) + ".dill", 'rb') as f:
        rslds = dill.load(f)
    print("Best LDS score = %.3f" % evaluate.evaluate_inferred_dynamic(lds, C_true, d_true))
    print("Best rSLDS score = %.3f" % evaluate.evaluate_inferred_dynamic(rslds, C_true, d_true))
    
    # compute the p value between outcomes from lds and rslds model
    test_elbos_K1, best_eR2s_K1, best_R2s_K1, best_maes_K1 = evaluate.get_individual_trial_evaluation(lds_path, lds_id, y_test, False, x_test, C_true, d_true)
    print(np.mean(np.stack(best_R2s_K1, axis=0), axis=0)[0])
    test_elbos_K3, best_eR2s_K3, best_R2s_K3, best_maes_K3 = evaluate.get_individual_trial_evaluation(rslds_path, rslds_id, y_test, False, x_test, C_true, d_true)
    print(np.mean(np.stack(best_R2s_K3, axis=0), axis=0)[0])
    best_R2s_K1 = [x[0] for x in best_R2s_K1]
    best_R2s_K3 = [x[0] for x in best_R2s_K3]
    best_maes_K1 = [x[0] for x in best_maes_K1]
    best_maes_K3 = [x[0] for x in best_maes_K3]
    stat, p_elbo = wilcoxon(test_elbos_K1, test_elbos_K3)
    stat, p_R2 = wilcoxon(best_R2s_K1, best_R2s_K3)
    stat, p_mae = wilcoxon(best_maes_K1, best_maes_K3)
    
    print (p_elbo, p_R2, p_mae)