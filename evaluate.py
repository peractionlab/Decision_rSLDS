import numpy as np
from multiprocessing import get_context
import os
import dill
import util

def compute_approximate_sum(mu):
    """
    Compute an approximate sum for a given mu.
    
    Parameters:
    - mu: Input value.
    
    Returns:
    - Approximate sum.
    """
    appr_term = 0
    factorial = 1
    for i in range(1, 51):
        factorial *= i
        appr_term += i * np.log(i) * np.power(mu, i) / factorial
    return np.exp(-mu) * appr_term - mu * np.log(mu + 1e-8)

def make_last_value_prediction(y_test, predict_len=10, interval=1):
    """
    Make predictions based on the last observed value.
    
    Parameters:
    - y_test: Test data.
    - predict_len: Number of steps to predict ahead.
    - interval: Interval between predictions.
    
    Returns:
    - Predicted values and true values.
    """
    save_y_pred = []
    save_y_true = []
    for i in range(len(y_test)): 
        y_preds = []
        y_trues = []
        index = list(range(1, y_test[i].shape[0]-predict_len, interval))     
        for t in index:
            r_pred = np.repeat(y_test[i][t-1][None, :], predict_len, axis=0)
            y_preds.append(r_pred)
            y_trues.append(y_test[i][t:t+predict_len, :])
                
        save_y_pred.append(y_preds)
        save_y_true.append(y_trues)
    
    return save_y_pred, save_y_true

def make_latent_variable_prediction(path, model_id, x_test, save_y_pred, C_true, d_true, predict_len=10, interval=1):
    """
    Make predictions for latent variables.
    
    Parameters:
    - path: Directory path.
    - model_id: Model identifier.
    - x_test: Test latent variables.
    - save_y_pred: Predicted observations.
    - C_true: True observation matrix.
    - d_true: True bias vector.
    - predict_len: Number of steps to predict ahead.
    - interval: Interval between predictions.
    
    Returns:
    - Predicted latent variables and true latent variables.
    """
    save_x_pred = []
    save_x_true = []
    
    if os.path.exists(path + str(model_id) + ".dill"):
        model, train_elbos, q = util.load_model(path + str(model_id) + ".dill")
    else:
        return FileNotFoundError
    
    C = model.emissions.Cs[0]
    d = model.emissions.ds[0]
    
    M_inv = np.linalg.pinv(np.linalg.pinv(C) @ C_true)
    n = np.linalg.pinv(C) @ (d_true - d)
    
    for i in range(len(x_test)): 
        x_preds = []
        x_trues = []
        index = list(range(1, x_test[i].shape[0]-predict_len, interval))     
        for j, t in enumerate(index):
            if hasattr(model.emissions, "mean"):
                x_pred = (np.log(np.exp(save_y_pred[i][j]) - 1)  - d) @ np.linalg.pinv(C).T
            else:
                x_pred = (save_y_pred[i][j] - d) @ np.linalg.pinv(C).T
            x_orig = (x_pred - n) @ M_inv.T
            x_preds.append(x_orig)
            x_trues.append(x_test[i][t:t+predict_len, :])
                
        save_x_pred.append(x_preds)
        save_x_true.append(x_trues)
            
    return save_x_pred, save_x_true

def make_prediction(path, model_id, y_test, predict_len=10, interval=1, cache=1):
    """
    Make predictions using the model.
    
    Parameters:
    - path: Directory path.
    - model_id: Model identifier.
    - y_test: Test data.
    - predict_len: Number of steps to predict ahead.
    - interval: Interval between predictions.
    - cache: Whether to use cached predictions.
    
    Returns:
    - Predicted values, true values, training ELBOs, and test ELBOs.
    """
    if os.path.exists(path + str(model_id) + ".dill"):
        model, train_elbos, q = util.load_model(path + str(model_id) + ".dill")
    else:
        return FileNotFoundError
    
    if cache == 1 and os.path.exists(path + 'prediction_cache.dill'):
        with open(path + 'prediction_cache.dill', 'rb') as f:
            save_y_pred = dill.load(f)
            save_y_true = dill.load(f)
            train_elbos = dill.load(f)
            test_elbos = dill.load(f)
    else:
        if model.K == 1:
            num_iters = 10
        else:
            num_iters = 200
        test_elbos = []
        save_y_pred = []
        save_y_true = []
        for i in range(len(y_test)): 
            elbo_test, _ = model.approximate_posterior(y_test[i],                                                       
                                                       num_iters=num_iters,
                                                       continuous_tolerance=1e-12,
                                                       continuous_maxiter=400,
                                                       verbose=0)
            test_elbos.append(elbo_test[-1])

            y_preds = []
            y_trues = []
            index = list(range(1, y_test[i].shape[0]-predict_len, interval))     
            for t in index:
                elbo_test, q_test = model.approximate_posterior(y_test[i][:t],                                                       
                                                               num_iters=num_iters,
                                                               continuous_tolerance=1e-12,
                                                               continuous_maxiter=400,
                                                               verbose=0)

                x_infer = q_test.mean_continuous_states[0]
                z_infer = model.most_likely_states(x_infer, y_test[i][:t])
                prefix = [z_infer, x_infer, y_test[i][:t]]
                z_pred, x_pred, _ = model.sample(predict_len, prefix=prefix, with_noise=False)

                if hasattr(model.emissions, "mean"):
                    r_pred = model.emissions.mean(np.matmul(model.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] 
                                                  + model.emissions.ds).squeeze()
                else:
                    r_pred = (np.matmul(model.emissions.Cs[None, ...], x_pred[:, None, :, None])[:, :, :, 0] + model.emissions.ds)[:, 0, :]

                y_preds.append(r_pred)
                y_trues.append(y_test[i][t:t+predict_len, :])
                
            save_y_pred.append(y_preds)
            save_y_true.append(y_trues)

        if cache == 1:
            with open(path + 'prediction_cache.dill', 'wb') as f:
                dill.dump(save_y_pred, f)
                dill.dump(save_y_true, f)
                dill.dump(train_elbos, f)
                dill.dump(test_elbos, f)
    
    if not isinstance(y_test[0][0, 0], np.int64) and hasattr(model.emissions, "mean"):
        save_y_true = []
        for i in range(len(y_test)): 
            y_trues = []
            index = list(range(1, y_test[i].shape[0]-predict_len, interval))     
            for t in index:
                y_trues.append(y_test[i][t:t+predict_len, :])
            save_y_true.append(y_trues)

    return save_y_pred, save_y_true, train_elbos, test_elbos

def get_across_trial_evaluation(path, model_id, y_test, cache=True, latent=False, x_test=None, C_true=None, d_true=None):
    """
    Evaluate the model across trials.
    
    Parameters:
    - path: Directory path.
    - model_id: Model identifier.
    - y_test: Test data.
    - cache: Whether to use cached predictions.
    - latent: Whether to evaluate latent variables.
    - x_test: Test latent variables.
    - C_true: True observation matrix.
    - d_true: True bias vector.
    
    Returns:
    - Evaluation metrics.
    """
    if model_id is None:
        save_y_pred, save_y_true = make_last_value_prediction(y_test)
        train_elbos, test_elbos = [0], [0]
    else:
        save_y_pred, save_y_true, train_elbos, test_elbos = make_prediction(path, model_id, y_test, cache=cache)
        if latent:
            save_y_pred, save_y_true = make_latent_variable_prediction(path, model_id, x_test, save_y_pred, C_true, d_true)

    y_trues = []
    y_preds = []
    for i in range(len(save_y_pred)): 
        for t in range(len(save_y_pred[i])): 
            y_preds.append(save_y_pred[i][t])
            y_trues.append(save_y_true[i][t])

    y_trues = np.stack(y_trues, axis=0)
    y_preds = np.stack(y_preds, axis=0)
    mae = np.mean(np.linalg.norm(y_preds - y_trues, ord=2, axis=-1), axis=0)
    test_elbo = np.mean(test_elbos)

    if isinstance(y_test[0][0, 0], np.int64) and not latent:
        y_means = np.mean(y_trues[:, :, :], axis=0)
        _numerator = 0
        for i in range(y_trues.shape[0]):
            for n in range(y_trues.shape[-1]):
                _numerator += compute_approximate_sum((y_preds[i, 0, n]))
        
        numerator = np.sum(np.sum(y_trues * np.log(y_trues / (y_preds + 1e-8) + 1e-8) - (y_trues - y_preds), axis=-1), axis=0)
        denominator = np.sum(np.sum(y_trues * np.log(y_trues / (y_means + 1e-8)  + 1e-8), axis=-1), axis=0)
        eR2 = 1 - _numerator / (denominator + 1e-8) 
        R2 = 1 - numerator / (denominator + 1e-8) 
    else:
        numerator = np.linalg.norm(y_preds - y_trues, ord=2, axis=-1)
        numerator = np.sum(numerator * numerator, axis=0)
        denominator = np.linalg.norm(y_trues - np.average(y_trues, axis=0), ord=2, axis=-1)
        denominator = np.sum(denominator * denominator, axis=0)
        R2 = 1 - numerator / (denominator + 1e-8)
        eR2 = [0]

    return model_id, train_elbos[-1], test_elbo, eR2, R2, mae

def get_individual_trial_evaluation(path, model_id, y_test, latent=False, x_test=None, C_true=None, d_true=None):
    """
    Evaluate the model for each individual trial.
    
    Parameters:
    - path: Directory path.
    - model_id: Model identifier.
    - y_test: Test data.
    - latent: Whether to evaluate latent variables.
    - x_test: Test latent variables.
    - C_true: True observation matrix.
    - d_true: True bias vector.
    
    Returns:
    - Evaluation metrics for each trial.
    """
    if model_id is None:
        save_y_pred, save_y_true = make_last_value_prediction(y_test)
        train_elbos, test_elbos = [0], [0]
    else:
        save_y_pred, save_y_true, train_elbos, test_elbos = make_prediction(path, model_id, y_test, interval=10)
        if latent:
            save_y_pred, save_y_true = make_latent_variable_prediction(path, model_id, x_test, save_y_pred, C_true, d_true)

    eR2s = []
    R2s = []
    maes = []
    for i in range(len(save_y_pred)):
        y_preds = np.stack(save_y_pred[i], axis=0)
        y_trues = np.stack(save_y_true[i], axis=0)
        y_means = np.mean(y_trues, axis=0)

        if isinstance(y_test[0][0, 0], np.int64) and not latent:
            _numerator = 0
            for j in range(y_trues.shape[0]):
                for n in range(y_trues.shape[-1]):
                    _numerator += compute_approximate_sum((y_preds[j, 0, n]))
            
            numerator = np.sum(np.sum(y_trues * np.log(y_trues / (y_preds + 1e-8) + 1e-8) - (y_trues - y_preds), axis=-1), axis=0)
            denominator = np.sum(np.sum(y_trues * np.log(y_trues / (y_means + 1e-8)  + 1e-8), axis=-1), axis=0)
            eR2 = 1 - _numerator / (denominator + 1e-8) 
            R2 = 1 - numerator / (denominator + 1e-8) 
        else:
            numerator = np.linalg.norm(y_preds - y_trues, ord=2, axis=-1)
            numerator = np.sum(numerator * numerator, axis=0)
            denominator = np.linalg.norm(y_trues - y_means, ord=2, axis=-1)
            denominator = np.sum(denominator * denominator, axis=0)
            R2 = 1 - numerator / (denominator + 1e-8)
            eR2 = [0]
        
        mae = np.mean(np.linalg.norm(y_preds - y_trues, ord=2, axis=-1), axis=0)
        
        eR2s.append(eR2)
        R2s.append(R2)
        maes.append(mae)

    eR2s = np.stack(eR2s, axis=0)
    R2s = np.stack(R2s, axis=0)
    maes = np.stack(maes, axis=0)
    
    return test_elbos, eR2s, R2s, maes

def select_best_model(path, y_val, model_num, pool_size):
    """
    Select the best model based on validation data.
    
    Parameters:
    - path: Directory path.
    - y_val: Validation data.
    - model_num: Number of models.
    - pool_size: Number of parallel processes.
    
    Returns:
    - Identifier of the best model.
    """
    result_path = path + "all_model_evaluation.npy"
    if not os.path.exists(result_path):
        return_packs = []
        with get_context("spawn").Pool(processes=pool_size) as pool:
            for i in range(model_num):
                return_packs.append(pool.apply_async(get_across_trial_evaluation, args=(path, i, y_val, False)))
            pool.close()
            pool.join()

        results = []
        for res in return_packs:
            results.append(res.get())

        np.save(result_path, np.array(results))
    else:
        results = list(np.load(result_path, allow_pickle=True))

    sorted_results = sorted(results, key=lambda x:x[4][0], reverse=True)
    best_model_id = sorted_results[0][0]
    
    return best_model_id

def evaluate_inferred_dynamic(model, C_true, d_true):
    """
    Evaluate the inferred dynamics.
    
    Parameters:
    - model: Model object.
    - C_true: True observation matrix.
    - d_true: True bias vector.
    
    Returns:
    - Evaluation score.
    """
    diff = 0
    atts_true = np.array([[6, 1], [1, 6]])
    for k in range(model.K):
        att = util.find_attractor(model)[k]
        att_orginal = util.compute_original_x(model, att, C_true, d_true)
        if np.min(np.linalg.norm(att_orginal - atts_true, ord=2, axis=1)) > 1:
            att_norm = np.linalg.norm(att_orginal, ord=2)
            diff += (np.abs(att_orginal[0] - att_orginal[1]) / np.sqrt(2) / att_norm) \
                + 2 * util.tanh(50 / (att_norm ** 2))
        else:
            diff += np.min(np.linalg.norm(att_orginal - atts_true, ord=2, axis=1))
    diff /= model.K

    offset = 0
    A_true = [np.eye(model.D), 0.5 * np.eye(model.D), 0.5 * np.eye(model.D)]
    b_true = np.array([[0.05, 0.05], [-0.5, -3], [-3, -0.5]])
    eigval_true = np.stack([np.linalg.eigvals(A_true[k]) for k in range(model.K)])
    for k in range(model.K):
        eigval_est = np.linalg.eigvals(model.dynamics.As[k])
        offset += np.min(np.linalg.norm(eigval_est - eigval_true, ord=2, axis=1))
    
    offset /= model.K

    error = 0 
    C_est_inv = np.linalg.pinv(model.emissions.Cs[0]).squeeze()
    M = C_est_inv.dot(C_true).squeeze()
    n = C_est_inv.dot((d_true - model.emissions.ds[0])).squeeze()
    
    R = model.transitions.Rs.dot(M)
    r = model.transitions.Rs.dot(n) + model.transitions.r
    
    bias_true = np.array([-1, 0, 1])
    for i in range(model.K):
        for j in range(i + 1, model.K):
            slope = (R[j, 0] - R[i, 0]) / (R[i, 1] - R[j, 1])
            bias = (r[i] - r[j]) / (R[i, 1] - R[j, 1])
            slope_err = np.min([np.abs(slope - 1), 5])
            bias_err = np.min([np.min(np.abs(bias - bias_true)), 5])
            error += slope_err + bias_err
    if model.K > 1:
        error /= (model.K * model.K - model.K)
    score = diff + offset + error
    return score