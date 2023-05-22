import subprocess
import tempfile
from pathlib import Path

import numpy as np
from phe import paillier

public_key, private_key = paillier.generate_paillier_keypair()


def weighted_average(data, weights=None):
    if weights is None:
        weights = [1 / len(data)] * len(data)
    assert abs(sum(weights) - 1) < 10**-6
    assert len(data) == len(weights)
    assert all([isinstance(x, np.ndarray) for x in data])
    assert all([len(x.shape) == 1 for x in data])

    data = [[public_key.encrypt(y) * w for y in x]
            for x, w in zip(data, weights)]
    data = [private_key.decrypt(sum(x)) for x in zip(*data)]
    data = np.array(data)
    return data


def kmeans(data, num_cluster):
    assert isinstance(data, np.ndarray) and len(data.shape) == 2

    scale = min(6, max([str(x)[::-1].find('.') for x in data.flatten()]))
    data = (data * (10**scale)).astype(int)

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'w') as f:
        f.write(f'{data.shape[1]} {data.shape[0]} {num_cluster}\n')
        for i in range(data.shape[0]):
            f.write(' '.join(map(str, data[i])) + '\n')

    executable_dir = Path(
        __file__).parent.parent / 'homomorphic-encryption' / 'bin'
    subprocess.Popen([executable_dir / 'server1'], cwd=executable_dir)
    subprocess.Popen([executable_dir / 'server2'], cwd=executable_dir)
    output = subprocess.check_output([executable_dir / 'client', tmp.name],
                                     text=True,
                                     cwd=executable_dir)
    return np.array(list(map(int, output.split())))


if __name__ == '__main__':
    print('Test weighted sum...')
    data = [
        np.array([0.3, 0.2, -0.1, 0.5]),
        np.array([0, 0.1, -0, 6.5]),
        np.array([0.5, 0.88, 0.1, 7]),
    ]
    weights = [0.5, 0.25, 0.25]
    print(weighted_average(data, weights=weights))
    print(np.average(data, weights=weights, axis=0))

    print('Test K-means...')
    data = np.array([
        [0.6, 0.5, 2., 1.8, 0.5],
        [1.6, 2.1, 1.3, 1.1, 1.4],
        [1.4, 1.1, 1.9, 0.6, 1.9],
        [2., 1.9, 1.8, 1.3, 0.2],
        [2.2, 2., 0.3, 1.3, 1.1],
        [0.5, 0.1, 0.6, 0.7, 1.1],
        [0.1, 0.5, 1.8, 1.8, 1.],
        [1.8, 0.2, 1.8, 1.8, 2.1],
        [2.2, 0.3, 1.8, 0.9, 1.4],
        [2.2, 0.5, 0., 1.6, 0.1],
    ])
    print(kmeans(data, num_cluster=3))
