import torch


def enc(x):
    return int(x)


def dec(c_x):
    return c_x


def negative(c_x):
    return -c_x


def add(c_x, c_y):
    return c_x + c_y


def add_tensor(c_x, c_y):
    return c_x + c_y


def add_const(c_x, y):
    return c_x + y


def mult(c_x, c_y):
    return c_x * c_y


def mult_tensor(c_x, c_y):
    return c_x * c_y


def mult_const(c_x, y):
    return c_x * y


def pow(c_x, power=2):  # positive integer
    return c_x * pow(c_x, power=power-1)


def pow_const(x, power=2):  # positive integer
    return x * pow(x, power=power-1)


def enc_vec(x):
    c_x = torch.empty_like(x)
    for i in range(x.shape[0]):
        c_x[i] = enc(x[i])
    return c_x


def dec_vec(c_x):
    x = torch.empty_like(c_x)
    for i in range(c_x.shape[0]):
        x[i] = dec(c_x[i])
    return c_x


def add_vec(c_x, c_y):
    c_z = torch.empty_like(c_x)
    for i in range(c_x.shape[0]):
        c_z[i] = add(c_x[i], c_y[i])
    return c_z


def mult_vec(c_x, c_y):
    c_z = torch.empty_like(c_x)
    for i in range(c_x.shape[0]):
        c_z[i] = mult(c_x[i], c_y[i])
    return c_z


def dot(c_x, c_y):
    mults = torch.empty_like(c_x)
    for i in range(c_x.shape[0]):
        mults[i] = mult(c_x[i], c_y[i])
    sum = add(mults[0], mults[1])
    for i in range(2, c_x.shape[0]):
        sum = add(sum, mults[i])
    return sum


def enc_mat(x):
    c_x = torch.empty_like(x)
    for i in range(x.shape[0]):
        c_x[i] = enc_vec(x[i])
    return c_x


def dec_mat(c_x):
    x = torch.empty_like(c_x)
    for i in range(c_x.shape[0]):
        x[i] = dec_vec(c_x[i])
    return x


def add_tensor(c_x, c_y):
    c_z = torch.empty_like(c_x)
    for i in range(c_x.shape[0]):
        c_z[i] = add_vec(c_x[i], c_y[i])
    return c_z


def add_matrix_vector(c_x, c_y): # add vector to rows
    c_z = torch.empty_like(c_x)
    for i in range(c_x.shape[0]):
        c_z[i] = add_vec(c_x[i], c_y)
    return c_z


def mat_mult(c_x, c_y):
    m = c_x.shape[0]
    n = c_y.shape[1]
    out = torch.empty((m, n))
    for i in range(m):
        for j in range(n):
            out[i][j] = dot(c_x[i], c_y[:, j])
    return out
