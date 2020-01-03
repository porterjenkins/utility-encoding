import torch


def loss_mse(y_true, y_hat):
    err = y_true - y_hat
    mse = torch.mean(torch.pow(err, 2))

    return mse


def utility_loss(y_hat, y_hat_c, y_hat_s, y_true, y_true_c, y_true_s):
    err = y_true - y_hat
    err_c = y_true_c - y_hat_c
    err_s = y_true_s - y_hat_s

    err_all = torch.cat((err.flatten(), err_c.flatten(), err_s.flatten()))
    return torch.mean(torch.pow(err_all, 2))


def mrs_loss(utility_loss, x_grad, x_c_grad, x_s_grad, lmbda=1):
    mrs_c = -(x_grad / x_c_grad)
    mrs_s = -(x_grad / x_s_grad)

    c_norm = torch.norm(mrs_c, dim=1)
    s_norm = torch.log(torch.norm(mrs_s, dim=1))

    mrs_loss = torch.mean(c_norm - s_norm)

    loss = mrs_loss + lmbda * utility_loss

    return loss

