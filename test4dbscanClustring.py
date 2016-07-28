import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy import io as io

print(__doc__)


##############################################################################
# Load patterns and scale them
X = io.loadmat('/media/ohadfel/New_Volume/patterns4clustering.mat')
X = StandardScaler().fit_transform(X['smoothedPatternsAsRows'])
##############################################################################

# Compute DBSCAN
cur_eps = 1
cur_min_samples = 5
epsess = [16*2**ii for ii in range(15)]

for cur_eps in epsess:
    db = DBSCAN(eps=cur_eps, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('num of clusters = '+str(n_clusters_)+' num of un-cluster = '+str(np.sum(labels == -1))+' ,esp = '+str(cur_eps)+' ,min_samples= ' + str(cur_min_samples))