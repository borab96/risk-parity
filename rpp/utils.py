import numpy as np
import scipy.cluster.hierarchy as sch
import pandas as pd


def cluster_corr(corr_array, inplace=False):
    """
    Clusters correlation matrix. Based on https://wil.yegelwel.com/cluster-correlation-matrix/

    :param corr_array:
    :param inplace:
    :return:
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='ward')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def linkage(mat, tree=False):
    pairwise_distances = sch.distance.pdist(mat)
    link = sch.linkage(pairwise_distances, method='ward')
    if tree:
        return sch.to_tree(link)
    else:
        return link


