# -*- coding: utf-8 -*-
from pyclustering.cluster.optics import optics
from gensim.models import word2vec
import numpy as np
from collections import Counter
import random
import os



def Optics(data, radius, minPt):
    """
    :param data: vectors of anonymous interfaces
    :return: a list of tuple(interface vector index in input data, optics output order)
    """
    op = optics(data, radius, minPt)
    op.process()

    clusters = op.get_clusters()
    if len(clusters) == 1:
        clusters = clusters[0]
        clusters.remove(0)
    else:
        print("reset the radius or minPt!")
        os._exit(1)

    ordering = op.get_ordering()
    return [(clusters[i], ordering[i]) for i in range(len(clusters))]


exp = 1e-6
def prob_same(x):
    y = np.exp(-np.square(x-E_same) / (2 * S2_same)) / (np.sqrt(2 * np.pi) * S_same)
    return max(y, exp)


def prob_different(x):
    y = np.exp(-np.square(x - E_different) / (2*S2_different)) / (np.sqrt(2 * np.pi) * S_different)
    return max(y, exp)


def distance_edc(a, b):
    """calculate the distance of vectors"""
    return np.sqrt(np.sum((a-b)**2))


def clustering(model, interfaces, vectors, radius, minPt):
    """ anonymous identification method based on maximum likelihood estimation """
    temp = [int(x) // 100000 for x in interfaces]
    number_R = dict(Counter(temp).most_common())  # anonymous router R_x => the number of interfaces that belong to R_x
    wight_R = {k: v / len(interfaces) for k, v in number_R.items()}
    merged_to_R = {k: 0 for k in temp}  # anonymous router R_x  =>  the number of interfaces that are merged into R_x
    correct_R = {k: 0 for k in temp}    # anonymous router R_x  =>  the number of interfaces that are correctly merged into R_x

    ordering = Optics(vectors, radius, minPt)
    # sorted by reachability-distance
    reachability_sort = sorted(ordering, key=lambda x: x[1])
    is_processed = [False] * len(ordering)

    count_cluster = 0
    while len(reachability_sort) > 1:
        interface_id = reachability_sort[0][0]    # the interface of minimum reachability-distance to be processed
        current_index = [x[0] for x in ordering].index(interface_id)

        count_cluster += 1
        cluster = []    # the indexed of the interfaces to be merged
        cluster_index_ordering = []  # the indexes in "ordering" of interface in "cluster"
        cluster.append(interface_id)
        cluster_index_ordering.append(current_index)

        check = current_index
        left = current_index - 1
        right = current_index + 1
        stop_left = False  # if True, do not extend to the left
        stop_right = False  # if True, do not extend to the right
        flag_left = True  # True means that this time we choose the left interface
        while True:
            if left == -1 or is_processed[left]:
                stop_left = True
            if right == len(ordering) or is_processed[right]:
                stop_right = True
            if stop_left and stop_right:
                break

            def go_left(check, left, flag_left):
                check = left
                left -= 1
                flag_left = True
                return check, left, flag_left

            def go_right(check, right, flag_left):
                check = right
                right += 1
                flag_left = False
                return check, right, flag_left

            if stop_left == False and stop_right == False:
                if ordering[left][1] <= ordering[right][1]:
                    check, left, flag_left = go_left(check, left, flag_left)
                else:
                    check, right, flag_left = go_right(check, right, flag_left)
            elif stop_left:
                check, right, flag_left = go_right(check, right, flag_left)
            else:
                check, left, flag_left = go_left(check, left, flag_left)

            # check if the likelihood function increases(only consider the influenced part of likelihood function)
            vec0 = model.wv[interfaces[ordering[check][0]]]
            change = 1.0
            for index in cluster_index_ordering:
                vec1 = model.wv[interfaces[ordering[index][0]]]
                dist = distance_edc(vec0, vec1)
                change *= prob_same(dist) / prob_different(dist)
            if change > 1.0:
                cluster.append(ordering[check][0])
                cluster_index_ordering.append(check)
                is_processed[check] = True
            elif flag_left:
                stop_left = True
            else:
                stop_right = True

        merged_interfaces = [interfaces[x] for x in cluster]
        print("cluster", count_cluster, ":", merged_interfaces)
        merged_routers = [int(x) // 100000 for x in merged_interfaces]
        counter = Counter(merged_routers).most_common()
        merged_to_R[counter[0][0]] += len(merged_routers)
        correct_R[counter[0][0]] += counter[0][1]

        def dele(x):
            return x[0] not in cluster
        reachability_sort = list(filter(dele, reachability_sort))

    precision = sum(correct_R[k] / merged_to_R[k] * wight_R[k] for k in correct_R.keys() if correct_R[k] != 0)
    recall = sum(correct_R[k] / number_R[k] * wight_R[k] for k in correct_R.keys())
    recognition = (len(interfaces) - count_cluster) / len(interfaces)

    return precision, recall, recognition


def load_file():
    # load anonymous routers
    anonymous_routers = []
    anonymousFile = "anonymous.txt"
    with open(anonymousFile, encoding='utf-8') as f:
        lines = f.readlines()
        anonymous_routers = [int(x.strip()) for x in lines]

    # load node2vec model
    modelFile = "embedding.model"
    model = word2vec.Word2Vec.load(modelFile)

    # load the vectors of anonymous interfaces
    # anonymous_interface_id = router_id * 100000 + visited_times_of__anonymous_router
    # nonanonymous_interface_id = router_id * 100000
    anonymous_interfaces = []
    anonymous_vectors = []
    for vocab in model.wv.vocab:
        if int(vocab) // 100000 in anonymous_routers:
            anonymous_vectors.append(model.wv[vocab])
            anonymous_interfaces.append(vocab)
    anonymous_vectors = np.asarray(anonymous_vectors)
    print("the number of anonymous interfaces：", len(anonymous_interfaces))

    anonymous_interfaces = [str(x) for x in anonymous_interfaces]

    def dele(x):
        return x not in anonymous_routers  # anonymous_interfaces
    nonanonymous_interfaces = list(filter(dele, list(model.wv.vocab)))

    return model, nonanonymous_interfaces, anonymous_interfaces, anonymous_vectors


if __name__ == '__main__':
    # statistic values
    global E_same, S_same, S2_same, E_different, S_different, S2_different
    E_same, S_same = 2.2523, 1.0688
    S2_same = S_same ** 2
    E_different, S_different = 4.4549, 0.655
    S2_different = S_different ** 2

    model, nonanonymous_interfaces, anonymous_interfaces, anonymous_vectors = load_file()
    precision, recall, recognition = clustering(model, anonymous_interfaces, anonymous_vectors, radius=5, minPt=3)
    print("precision：", precision)
    print("recall：",  recall)
    print("recognition：", recognition)

