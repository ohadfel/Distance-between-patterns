import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy import io as io
import h5py
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
from sklearn import manifold

print(__doc__)


##############################################################################
# Load patterns and scale them
# X = io.loadmat('/media/ohadfel/New_Volume/patterns4clustering.mat')
# X = h5py.File('/media/ohadfel/New_Volume/personSimilarity.mat')
# X = np.array(X['similarity_mat'])
X = h5py.File('/media/ohadfel/New_Volume/pearson_Woffset_symetric.mat')
X = np.array(X['similarity_mat2'])
distArray = ssd.squareform(X)
# X = StandardScaler().fit_transform(X['smoothedPatternsAsRows'])
##############################################################################
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0,
                   dissimilarity="precomputed", n_jobs=4,verbose=5)
pos = mds.fit(X).embedding_
plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')
plt.show()



# Compute DBSCAN
cur_eps = 1
cur_min_samples = 5
# epsess = [32+ii*5 for ii in range(15)]
epsess = [0.001*ii for ii in range(1,600)]
min_samples = [5*ii for ii in range(1, 50)]
min_samples = [15]
num_of_clusters = []
num_of_unclustered = []
final_epsess = []
best_labels = []

plt.ion()
print('clustering started!')
for cur_min_samples in min_samples:
    for cur_eps in epsess:
        dbRaw = DBSCAN(eps=cur_eps, min_samples=5, metric="precomputed")
        db = dbRaw.fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('num of clusters = '+str(n_clusters_)+' num of un-cluster = '+str(np.sum(labels == -1))+' ,esp = '+str(cur_eps)+' ,min_samples= ' + str(cur_min_samples))
        num_of_clusters.append(n_clusters_)
        num_of_unclustered.append(np.sum(labels == -1))
        final_epsess.append(cur_eps)
        if max(num_of_clusters) == n_clusters_:
            best_labels = labels
        plt.plot(final_epsess, num_of_clusters, 'b')
        plt.pause(0.02)
        # plt.draw()
        if np.sum(labels == -1) == 0:
            break

fig, ax1 = plt.subplots()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(final_epsess, num_of_clusters, 'b-')
ax1.set_xlabel('epsses', fontsize=18)

# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('exp', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax1.set_ylabel('# of clusters',fontsize=16)


ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(final_epsess, num_of_unclustered, 'r-')
ax2.set_ylabel('sin', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
ax2.set_ylabel('# of unclustered',fontsize=16)
fig.suptitle('DBSCAN # clusters vs # unclusterd patterns \n as function of epsses', fontsize=16)
plt.tight_layout()
plt.show()
print('clustering finished!')