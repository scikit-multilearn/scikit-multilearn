import math

def log_likelihood_loss(y, y_pred):
    log_likelihood_loss = 0
    for row in range(y.shape[0]):
        if y.ndim == 1 :
            if y[row] == 1:
                log_likelihood_loss += (math.log(y_pred[row]) * -1)
            else:
                log_likelihood_loss += (math.log(1 - y_pred[row]) * -1)
        elif y.ndim == 2 :
            for column in range(y.shape[1]):
                if y[row, column] == 1:
                    log_likelihood_loss += (math.log(y_pred[row, column]) * -1)
                else:
                    log_likelihood_loss += (math.log(1 - y_pred[row, column]) * -1)
    return log_likelihood_loss


def exact_match_accuracy(y, y_pred):
    ema = 0
    for row in range(y.shape[0]):
        if y.ndim == 1 :
            if y[row, column] == y_pred[row, column]:
                ema += 1

        elif y.ndim == 2 :
            flag = 0
            for column in range(y.shape[1]):
                if y[row, column] == y_pred[row, column] :
                    continue
                else :
                    flag += 1
                    break
            if flag == 0 :
                ema += 1
    ema = ema/y.shape[0]
    return ema