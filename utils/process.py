import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy import sparse
from sklearn import preprocessing

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
# """
#  Prepare adjacency matrix by expanding up to a given neighbourhood.
#  This will insert loops on every node.
#  Finally, the matrix is converted to bias vectors.
#  Expected shape: [graph, nodes, nodes]
# """
# def adj_to_bias(adj, sizes, nhood=1):
#     nb_graphs = adj.shape[0]
#     mt = np.empty(adj.shape)
#     for g in range(nb_graphs):
#         mt[g] = np.eye(adj.shape[1])
#         for _ in range(nhood):
#             mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
#         for i in range(sizes[g]):
#             for j in range(sizes[g]):
#                 if mt[g][i][j] > 0.0:
#                     mt[g][i][j] = 1.0
#     return -1e9 * (1.0 - mt)


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        # for i in range(sizes[g]):
        #     for j in range(sizes[g]):
        #         if mt[g][i][j] > 0.0:
        #             mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def get_bias_mat(adj, size, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.ones((adj.shape[1], adj.shape[1]))
        # for _ in range(nhood):
        #     mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        # for i in range(sizes[g]):
        #     for j in range(sizes[g]):
        #         if mt[g][i][j] > 0.0:
        #             mt[g][i][j] = 1.0
    return mt

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
#     """Load data."""
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))

#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)

#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended

#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y)+500)

#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])

#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]

#     print(adj.shape)
#     print(features.shape)

#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_group_data():
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.datasets import fetch_20newsgroups
    from time import time
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import pairwise_distances

    # n_samples = 2000
    n_features = 1000
    n_topics = 20

    dataset = fetch_20newsgroups(shuffle=True, random_state=1, 
#						              categories=['comp.graphics', 
#                                                                        'rec.sport.baseball', 
#                                                                        'talk.politics.guns'], 
        #                                                                'alt.atheism', 
        #                                                                'misc.forsale'], 
                                remove=('headers', 'footera', 'quotes'))
    #dataset = fetch_20newsgroups(shuffle=True, random_state=1, 
    #                            remove=('headers', 'footera', 'quotes'))
    data_samples = dataset.data
    labels = OneHotEncoder().fit_transform(dataset.target.reshape(-1, 1)).toarray()

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, 
                                    # min_df=2, 
                                    max_features=n_features, 
                                    stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data_samples)

    # each edge attribute is denoted by a adj matrix
    adjs = []
    adj = cosine_similarity(tfidf)
    #adj[adj < 0.5] = 0
    adjs.append(adj)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=1).fit_transform(tfidf) # (N X n_topics) matrix
    for tid in range(n_topics):
        feature_mat = lda[:,tid].reshape(-1,1)
        lda_adj = pairwise_distances(feature_mat, metric=lambda x, y: (x+y)/2, n_jobs=8)
        # adjs.append(lda_adj)

    # nmf = NMF(n_components=n_topics, random_state=1, 
    #           alpha=0.1, l1_ratio=0.5).fit_transform(tfidf) # (N X n_topics) matrix
    # for tid in range(n_topics):
    #     feature_mat = nmf[:,tid].reshape(-1,1)
    #     nmf_adj = pairwise_distances(feature_mat, metric=lambda x, y: (x+y)/2)
    #     adjs.append(nmf_adj)

    features = tfidf

    size = tfidf.shape[0]
    train_ratio = 0.8

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size*train_ratio)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size*train_ratio):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size*train_ratio):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adjs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(edge_attr_directory, node_features_path, label_path, train_ratio):
    print('loading data')
    # Edge features
    adjs = []
    edge_attr_name = []
    for edge_attr_file in os.listdir(edge_attr_directory):
        edge_name = os.path.split(edge_attr_file)[1]
        edge_attr_name.append(edge_name)  # use the file name as edge attr name
        adj = sp.load_npz(os.path.join(edge_attr_directory, edge_attr_file))
        variance = np.var(adj.data)
        print('variance of edge {}: {}'.format(edge_name, str(variance)))
        adj.data = adj.data/variance
        adjs.append(adj)

    # Node Features
    features = pd.read_csv(node_features_path, delimiter=',').values

    # Ground Truth label
    labels = pd.read_csv(label_path, delimiter=',')
    
    # convert label to one-hot format
    labels = pd.get_dummies(data=labels, dummy_na=True, columns=['label']).set_index('id') # N X (#edge attr)
    # print(labels.columns)
    labels = labels.drop(['label_nan'], axis=1)

    size = features.shape[0]

    train_id = set()
    test_id = set()
    train_mask = np.zeros((size,)).astype(bool)
    val_mask = np.zeros((size,)).astype(bool)
    test_mask = np.zeros((size,)).astype(bool)

    np.random.seed(1)
    for column in labels.columns:
        set_of_key = set(labels[(labels[column] == 1)].index)
        train_key_set = set(np.random.choice(list(set_of_key), size=int(len(set_of_key)*train_ratio), replace=False))
        test_key_set = set_of_key - train_key_set
        train_id = train_id.union(train_key_set)
        test_id = test_id.union(test_key_set)
    train_mask[list(train_id)] = 1
    val_mask[list(test_id)] = 1
    test_mask[list(test_id)] = 1

    # labels = labels.values[:,:-1]  # convert to numpy format and remove the nan column
    y_train = np.zeros((size, labels.shape[1]))
    y_val = np.zeros((size, labels.shape[1]))
    y_test = np.zeros((size, labels.shape[1]))
    y_train[train_mask, :] = labels.loc[sorted(train_id)]
    y_val[val_mask, :] = labels.loc[sorted(test_id)]
    y_test[test_mask, :] = labels.loc[sorted(test_id)]

    # standardize node features and convert it to sparse matrix
    # print(features)
    # print(features[train_mask])
    # scaler = preprocessing.StandardScaler().fit(features)
    # print(scaler.mean_)
    # print(scaler.var_)
    # # scaler = preprocessing.MinMaxScaler(feature_range=(-2, 2), copy=True).fit(features)
    # features = scaler.transform(features)
    features = sparse.csr_matrix(features)

    print('adjs length: ', len(adjs))
    print('features shape: ', features.shape)
    print('y_train shape: ', y_train.shape)
    print('y_val shape: ', y_val.shape)
    print('y_test shape: ', y_test.shape)
    print('train_mask shape: ', train_mask.shape)
    print('val_mask shape: ', val_mask.shape)
    print('test_mask shape: ', test_mask.shape)

    return adjs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_attr_name

def load_random_data(size):

    adj = sp.random(size, size, density=0.002) # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7)) # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size/2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size/2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size/2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
  
    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_networkx_data(file_path):
    g=nx.readwrite.edgelist.read_edgelist('/home/handason/NRL/Deep-NRL/data/adj_mat.csv', delimiter=',', data=[('sum_value', float), ('avg_value', float), ('var_value', str), ('count_trans', float), ('count_call', float), ('count_mining', float), ('count_other', float)], comments='#')

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_bias(adj, to_unweighted=False):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    if to_unweighted:  # to unweighted graph
        adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape
