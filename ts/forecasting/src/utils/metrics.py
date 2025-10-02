import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    true = np.where(true == 0, 1e-5, true)
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    true = np.where(true == 0, 1e-5, true)
    return np.mean(np.square((pred - true) / true))

def cal_metric(pred, true, digits=4):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)

    metric_dict = {
        'MAE': round(mae, digits),
        'MSE': round(mse, digits),
        'RMSE': round(rmse, digits),
        'MAPE': round(mape, digits),
        'MSPE': round(mspe, digits),
        'RSE': round(rse, digits),
    }

    return metric_dict

def log_metrics(modelname, dataname, metrics, logger):
    logger.info(f"{modelname} model, {dataname} dataset")

    for k, v in metrics.items():
        logger.info(f"  - {k:<6}: {v:.4f}" if isinstance(v, float) else f"  - {k:<6}: {v:.4f}")