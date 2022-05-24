import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import time
from sklearn import metrics
digits=load_digits()  ###read the data
data=scale(digits.data)
y=digits.target        ###read the target

k=10                  ###number of clusters
samples,features=data.shape

def bench_kmeans(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name,  estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_)))


clf=KMeans(n_clusters=k,init='random',n_init=10)
bench_kmeans(clf,"1",data)

### the document is here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
