import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder_path = '/home/chiroptera/QCThesis/datasets/gauss10e6_separated/'
data = np.load('{}{}'.format(folder_path, 'data.npy'))
gt = np.load('{}{}'.format(folder_path, 'gt.npy'))

# number of samples
cardinality = [1e2,2.5e2,5e2,7.5e2,
               1e3,2.5e3,5e3,7.5e3,
               1e4,2.5e4,5e4,7.5e4,
               1e5,2.5e5,5e5,7.5e5,
               1e6,2.5e6]
cardinality = map(int,cardinality)

total_n = data.shape[0]
div = map(lambda n: total_n / n, cardinality)

for n_samples, d in zip(cardinality, div):
    print n_samples
    fig=plt.Figure()
    data_s = data[::d]
    gt_s = gt[::d]
    for l in np.unique(gt_s):
        idx = gt_s == l
        plt.plot(data_s[idx, 0], data_s[idx, 1], '.')
    plt.savefig('{}data_plot_{}.png'.format(folder_path, n_samples))
    plt.close()