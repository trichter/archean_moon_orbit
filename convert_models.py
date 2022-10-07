# Copyright 2022 Tom Eulenfeld, MIT license

import numpy as np


r0 = 3.844e8
ts0 = 86400
Y = 365.24 * ts0

def _load_webb1982():
    def _convert(age, radius):
        return -float(age) / 1000, float(radius) * 6371e3 / r0
    with open('data/webb1982.txt') as f:
        data = f.read()
    lines = data.splitlines()
    model_lines = [('a', 6, 48),
                   ('b', 52, 102),
                   ('c', 106, 174),
                   ('d', 178, 240)]
    models = {f'Webb 1982 curve {m}': list(zip(*[_convert(*l.split()) for l in lines[i1:i2]]))
              for m, i1, i2 in model_lines}
    return models

def _load_daher2021():
    from scipy.io import loadmat
    fname = 'data/daher2021/integration_results_use_Schindelegger_{}_experiments_ode45_nodeLF_v9.mat'
    models = {}
    for age in ('PD', '55Ma', '116Ma', '252Ma'):
        data = loadmat(fname.format(age))
        k1 = 'timestep_vector_{}_ode45'.format(age)
        k2 = 'StateVector_{}_ode45'.format(age)
        models[f'Daher et al. 2021 {age}'] = [data[k1][:, 0] / Y / 1e9, data[k2][:, 2] / r0]
    return models

def _load_tyler2021():
    data = np.loadtxt('data/tyler2021.txt')
    age = -data[:, 0] / 1e9
    model = {}
    for i in range(3):
        Tdis = [30, 40, 50][i]
        model[f'Tyler 2021 T={Tdis}'] = [age, data[:, i+1] / r0]
    return model

def _load_farhat2022():
    data = np.loadtxt('data/farhat2022.txt', skiprows=7)
    age = data[:, 0]
    model = {'Farhat et al. 2022': [age, data[:, 1] / 60.142611],
             'Farhat et al. 2022 2sigma': [age, data[:, 2] / 60.142611, data[:, 3] / 60.142611]}
    return model

def save_models():
    models = dict(list(_load_webb1982().items()) +
                  list(_load_daher2021().items()) +
                  list(_load_tyler2021().items()) +
                  list(_load_farhat2022().items()))
    np.savez('data/orbit_models.npz', **models)


save_models()
