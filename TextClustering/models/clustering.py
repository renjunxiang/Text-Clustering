from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
import numpy as np


def clustering(X=None,
               model_name='KMeans',
               n_clusters=3,
               **param):
    if model_name == 'KMeans':  # K均值
        model = KMeans(n_clusters=n_clusters, **param)
    elif model_name == 'SpectralClustering':  # 谱聚类
        model = SpectralClustering(n_clusters=n_clusters, **param)
    elif model_name == 'AffinityPropagation':  # AP聚类
        model = AffinityPropagation(**param)
    model.fit(X=X)
    labels = model.labels_
    return model, labels


if __name__ == '__main__':
    # np.random.rand(10)
    x = [np.random.rand(10) for i in range(20)]
    y = [np.random.randint(0, 3) for i in range(20)]
    model, labels = clustering(X=x,
                               model_name='KMeans',
                               n_clusters=3)
    print(model)
    print(labels)
