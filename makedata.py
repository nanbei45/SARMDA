import csv

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp


def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    #np.sum()将矩阵中所有元素加起来,得到1的个数
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy

def single_generate_graph_adj_and_feature(network, feature):
    '''.todense()将稀疏矩阵转为稠密矩阵。
    .tocoo()将稠密矩阵转为稀疏矩阵。
    csr是按行压缩的稀疏矩阵。
    network是微生物/疾病相似性的01矩阵
    fearture是行为微生物，列为疾病的相关性01矩阵 /行为疾病，列为微生物的相关性01矩阵
    '''
    #存储为稀疏格式
    features = sp.csr_matrix(feature).tolil().todense()
    #将邻接矩阵转换成图
    graph = nx.from_numpy_array(network)
    adj = nx.adjacency_matrix(graph)
    #存储为稀疏格式
    adj = sp.coo_matrix(adj)
    return adj, features


#python2可以用file替代open
def makefiledata(file,filepath):
    with open(filepath, "w") as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        #writer.writerow(["mirco_id", "dis_id"])
        row, col = file.shape
        for i in range(col):
            for j in range(row):
                if (file[j][i] == 1):
                    writer.writerow([i, j])


