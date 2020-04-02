# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of preprocessing functions for different graph formats."""

import json
import time
import sys

from networkx.readwrite import json_graph
import numpy as np
import partition_utils
import scipy.sparse as sp
import sklearn.metrics
import sklearn.preprocessing
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile

import networkx as nx
import pickle as pkl 
from sklearn.model_selection import train_test_split 
flags = tf.app.flags
FLAGS = flags.FLAGS

def parse_index_file(filename):
  """Parse index file."""
  index = []
  for line in gfile.Open(filename):
    index.append(int(line.strip()))
  return index


def sample_mask(idx, l):
  """Create mask."""
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=np.bool)


def unnormlize_adj(adj):
  """Add self-loop and return, not normalize"""
  adj = adj + sp.eye(adj.shape[0])
  return adj

def sym_normalize_adj(adj):
  """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
  adj = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1)) + 1e-20
  d_inv_sqrt = np.power(rowsum, -0.5).flatten()
  d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
  adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
  return adj


def normalize_adj(adj):
  rowsum = np.array(adj.sum(1)).flatten()
  d_inv = 1.0 / (np.maximum(1.0, rowsum))
  d_mat_inv = sp.diags(d_inv, 0)
  adj = d_mat_inv.dot(adj)
  return adj


def normalize_adj_diag_enhance(adj, diag_lambda):
  """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
  adj = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1)).flatten()
  d_inv = 1.0 / (rowsum + 1e-20)
  d_mat_inv = sp.diags(d_inv, 0)
  adj = d_mat_inv.dot(adj)
  adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
  return adj


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


def calc_f1(y_pred, y_true, multilabel):
  if multilabel:
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
  else:
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
  return sklearn.metrics.f1_score(
      y_true, y_pred, average='micro'), sklearn.metrics.f1_score(
          y_true, y_pred, average='macro')


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
  """Construct feed dictionary."""
  feed_dict = dict()
  feed_dict.update({placeholders['labels']: labels})
  feed_dict.update({placeholders['labels_mask']: labels_mask})
  feed_dict.update({placeholders['features']: features})
  feed_dict.update({placeholders['support']: support})
  feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
  return feed_dict

def construct_feed_dict_afm(features,features_idx,features_val, support, labels, labels_mask, placeholders):
  """Construct feed dictionary for AFM."""
  feed_dict = dict()
  feed_dict.update({placeholders['labels']: labels})
  feed_dict.update({placeholders['labels_mask']: labels_mask})
  feed_dict.update({placeholders['features']: features})
  feed_dict.update({placeholders['features_idx']: features_idx})
  feed_dict.update({placeholders['features_val']: features_val})
  feed_dict.update({placeholders['support']: support})
  feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
  return feed_dict


def preprocess_multicluster(adj,
                            parts,
                            features,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1,
                            feat_sparse=False):
  """Generate the batch for multiple clusters."""

  features_batches = []
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  np.random.shuffle(parts)
  for _, st in enumerate(range(0, num_clusters, block_size)):
    pt = parts[st]
    for pt_idx in range(st + 1, min(st + block_size, num_clusters)): 
      pt = np.concatenate((pt, parts[pt_idx]), axis=0)  # concat [st: st+block_size] in a batch
    features_batches.append(sparse_to_tuple(features[pt, :])) if feat_sparse else features_batches.append(features[pt, :])
    y_train_batches.append(y_train[pt, :])
    support_now = adj[pt, :][:, pt]
    if diag_lambda == -1: # no diag enhance
      support_batches.append(sparse_to_tuple(normalize_adj(support_now))) # renormalize adj
    elif diag_lambda == -2: # Kipf GCN
      support_batches.append(sparse_to_tuple(sym_normalize_adj(support_now))) # renormalize adj
    elif diag_lambda == 0 and FLAGS.model =='gat_nfm':
      support_batches.append(sparse_to_tuple(support_now))
    else:
      support_batches.append(
          sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
    total_nnz += support_now.count_nonzero()

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt))) # train_mask in this batch
  return (features_batches, support_batches, y_train_batches,
          train_mask_batches)

# TODO: 增加判断是否是sparse feature input，features_batches元素为sparse tuple
def preprocess(adj,
               features,
               y_train,
               train_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1,
               sparse_input=False):
  """Do graph partitioning and preprocessing for SGD training."""

  # Do graph partitioning
  part_adj, parts = partition_utils.partition_graph(adj, visible_data,
                                                    num_clusters)
  if diag_lambda == -1:
    part_adj = normalize_adj(part_adj)
  elif diag_lambda == -2:
    part_adj = sym_normalize_adj(part_adj)
  elif diag_lambda == 0 and FLAGS.model =='gat_nfm':
    part_adj = unnormlize_adj(part_adj)
  else:
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
  parts = [np.array(pt) for pt in parts]

  features_batches = []
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  for pt in parts:
    if sparse_input:
      features_batches.append(sparse_to_tuple(features[pt, :]))
    else:
      features_batches.append(features[pt, :])
    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(sparse_to_tuple(now_part))
    y_train_batches.append(y_train[pt, :])

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt)))
  return (parts, features_batches, support_batches, y_train_batches,
          train_mask_batches)



def preprocess_train_afm(adj,
               features,
               features_idx,
               features_val,
               y_train,
               train_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1,
               sparse_input=False):
  """Do graph partitioning and preprocessing for SGD training. Patition train dataset."""
  part_adj, parts = partition_utils.partition_graph(adj, visible_data,
                                                    num_clusters)
  if diag_lambda == -1:
    part_adj = normalize_adj(part_adj)
  elif diag_lambda == -2:
    part_adj = sym_normalize_adj(part_adj)
  elif diag_lambda == 0 and FLAGS.model == 'gat_nfm':
    part_adj = unnormlize_adj(part_adj)
  else:
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
  parts = [np.array(pt) for pt in parts]

  features_batches = [[],[],[]]
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  for pt in parts:
    if sparse_input:
      features_batches[0].append(sparse_to_tuple(features[pt, :]))  # features_sp
    else:
      features_batches.append(features[pt, :])
    features_batches[1].append(features_idx[pt,:])
    features_batches[2].append(features_val[pt,:])
    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(sparse_to_tuple(now_part))
    y_train_batches.append(y_train[pt, :])

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt)))
  return (parts, features_batches, support_batches, y_train_batches,
          train_mask_batches)

# TODO: 增加一次划分得到train/test/val的代码以测试一次划分效果
def preprocess_val_test(adj,
               features,
               y_val,
               val_mask,
               y_test,
               test_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1):
  """Do graph partitioning and preprocessing for SGD training. Patition validation and test set in the same time"""

  # Do graph partitioning
  part_adj, parts = partition_utils.partition_graph(adj, visible_data,
                                                    num_clusters)
  if diag_lambda == -1:
    part_adj = normalize_adj(part_adj)
  elif diag_lambda == -2:
    part_adj = sym_normalize_adj(part_adj)
  elif diag_lambda == 0 and FLAGS.model == 'gat_nfm':
    part_adj = unnormlize_adj(part_adj)
  else:
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
  parts = [np.array(pt) for pt in parts]

  features_val_batches = []
  support_batches = []
  y_val_batches = []
  val_mask_batches = []
  y_test_batches = []
  test_mask_batches = []
  total_nnz = 0
  for pt in parts:
    features_val_batches.append(sparse_to_tuple(features[pt, :]))
    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(sparse_to_tuple(now_part))
    y_val_batches.append(y_val[pt, :])
    y_test_batches.append(y_test[pt, :])

    val_pt = []
    test_pt = []
    for newidx, idx in enumerate(pt):
      if val_mask[idx]:
        val_pt.append(newidx)
      if test_mask[idx]:
        test_pt.append(newidx)
    val_mask_batches.append(sample_mask(val_pt, len(pt)))
    test_mask_batches.append(sample_mask(test_pt, len(pt)))
  features_test_batches = features_val_batches
  return (parts, features_val_batches, features_test_batches, support_batches, y_val_batches,y_test_batches,
          val_mask_batches, test_mask_batches)

###################################
##  Data preprocess for GCN_AFM  ##
###################################

def preprocess_val_test_afm(adj,
               features,
               features_idx,
               features_val,
               y_val,
               val_mask,
               y_test,
               test_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1):
  """Do graph partitioning and preprocessing for SGD training. Patition validation and test set in the same time"""

  # Do graph partitioning
  part_adj, parts = partition_utils.partition_graph(adj, visible_data,
                                                    num_clusters)
  if diag_lambda == -1:
    part_adj = normalize_adj(part_adj)
  elif diag_lambda == -2:
    part_adj = sym_normalize_adj(part_adj)
  elif diag_lambda == 0 and FLAGS.model =='gat_nfm':
    part_adj = part_adj
  else:
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
  parts = [np.array(pt) for pt in parts]

  # TODO: feature_idx/ feature_val的计算只与验证集和测试集自身有关，无需加入训练集
  features_val_batches = [[],[],[]] # [features_sp, features_idx, features_val]
  support_batches = []
  y_val_batches = []
  val_mask_batches = []
  y_test_batches = []
  test_mask_batches = []
  total_nnz = 0
  for pt in parts:
    features_val_batches[0].append(sparse_to_tuple(features[pt, :]))  # features_sp
    features_val_batches[1].append(features_idx[pt,:])
    features_val_batches[2].append(features_val[pt,:])

    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(sparse_to_tuple(now_part))
    y_val_batches.append(y_val[pt, :])
    y_test_batches.append(y_test[pt, :])

    val_pt = []
    test_pt = []
    for newidx, idx in enumerate(pt):
      if val_mask[idx]:
        val_pt.append(newidx)
      if test_mask[idx]:
        test_pt.append(newidx)
    val_mask_batches.append(sample_mask(val_pt, len(pt)))
    test_mask_batches.append(sample_mask(test_pt, len(pt)))
  features_test_batches = features_val_batches
  return (parts, features_val_batches, features_test_batches, support_batches, y_val_batches,y_test_batches,
          val_mask_batches, test_mask_batches)


def preprocess_train_val_test_afm(adj,
                part_adj,
                parts,
               features,
               features_idx,
               features_val,
               y_val,
               val_mask,
               y_test,
               test_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1):
  """Do graph partitioning and preprocessing for SGD training. Patition validation and test set in the same time"""

  # Do graph partitioning
  part_adj, parts = partition_utils.partition_graph(adj, visible_data,
                                                    num_clusters)
  if diag_lambda == -1:
    part_adj = normalize_adj(part_adj)
  elif diag_lambda == -2:
    part_adj = sym_normalize_adj(part_adj)
  elif diag_lambda == 0 and FLAGS.model == 'nfm_gat':
    part_adj = unnormlize_adj(part_adj)
  else:
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
  parts = [np.array(pt) for pt in parts]

  # TODO: feature_idx/ feature_val的计算只与验证集和测试集自身有关，无需加入训练集
  features_val_batches = [[],[],[]] # [features_sp, features_idx, features_val]
  support_batches = []
  y_val_batches = []
  val_mask_batches = []
  y_test_batches = []
  test_mask_batches = []
  total_nnz = 0
  for pt in parts:
    features_val_batches[0].append(sparse_to_tuple(features[pt, :]))  # features_sp
    features_val_batches[1].append(features_idx[pt,:])
    features_val_batches[2].append(features_idx[pt,:])

    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(sparse_to_tuple(now_part))
    y_val_batches.append(y_val[pt, :])
    y_test_batches.append(y_test[pt, :])

    val_pt = []
    test_pt = []
    for newidx, idx in enumerate(pt):
      if val_mask[idx]:
        val_pt.append(newidx)
      if test_mask[idx]:
        test_pt.append(newidx)
    val_mask_batches.append(sample_mask(val_pt, len(pt)))
    test_mask_batches.append(sample_mask(test_pt, len(pt)))
  features_test_batches = features_val_batches
  return (parts, features_val_batches, features_test_batches, support_batches, y_val_batches,y_test_batches,
          val_mask_batches, test_mask_batches)

def preprocess_multicluster_afm(adj,
                            parts,
                            features,
                            features_idx,
                            features_val,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1,
                            feat_sparse=False):
  """Generate the batch for multiple clusters."""

  features_batches = [[],[],[]]
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  np.random.shuffle(parts)
  for _, st in enumerate(range(0, num_clusters, block_size)):
    pt = parts[st]
    # merge multiple adj block
    for pt_idx in range(st + 1, min(st + block_size, num_clusters)): 
      pt = np.concatenate((pt, parts[pt_idx]), axis=0)  # concat [st: st+block_size] in a batch
    features_batches[0].append(sparse_to_tuple(features[pt, :])) if feat_sparse else features_batches.append(features[pt, :])
    features_batches[1].append(features_idx[pt, :])
    features_batches[2].append(features_val[pt, :])

    y_train_batches.append(y_train[pt, :])
    support_now = adj[pt, :][:, pt]
    if diag_lambda == -1: # no diag enhance
      support_batches.append(sparse_to_tuple(normalize_adj(support_now))) # renormalize adj
    elif diag_lambda == -2:
      support_batches.append(sparse_to_tuple(sym_normalize_adj(support_now))) # renormalize adj
    elif diag_lambda == 0 and FLAGS.model =='gat_nfm':
      support_batches.append(sparse_to_tuple(support_now))
    else:
      support_batches.append(
          sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
    total_nnz += support_now.count_nonzero()

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt))) # train_mask in this batch
  return (features_batches, support_batches, y_train_batches,
          train_mask_batches)


def load_graphsage_data(dataset_path, dataset_str, normalize=True):
  """Load GraphSAGE data."""
  start_time = time.time()

  graph_json = json.load(
      gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,
                                          dataset_str)))
  graph_nx = json_graph.node_link_graph(graph_json)

  id_map = json.load(
      gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,
                                               dataset_str)))
  is_digit = list(id_map.keys())[0].isdigit()
  id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
  class_map = json.load(
      gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,
                                                  dataset_str)))

  is_instance = isinstance(list(class_map.values())[0], list)
  class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
               for k, v in class_map.items()}

  broken_count = 0
  to_remove = []
  for node in graph_nx.nodes():
    if node not in id_map:
      to_remove.append(node)
      broken_count += 1
  for node in to_remove:
    graph_nx.remove_node(node)
  tf.logging.info(
      'Removed %d nodes that lacked proper annotations due to networkx versioning issues',
      broken_count)

  feats = np.load(
      gfile.Open(
          '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
          'rb')).astype(np.float32)

  tf.logging.info('Loaded data (%f seconds).. now preprocessing..',
                  time.time() - start_time)
  start_time = time.time()

  edges = []
  for edge in graph_nx.edges():
    if edge[0] in id_map and edge[1] in id_map:
      edges.append((id_map[edge[0]], id_map[edge[1]]))
  num_data = len(id_map)
  # g_nodes = graph_nx.nodes()
  # node_list = [n for n in graph_nx.nodes()]
  val_data = np.array(
      [id_map[n] for n in graph_nx.nodes() if graph_nx.nodes[n]['val'] == True],
      dtype=np.int32)
  test_data = np.array(
      [id_map[n] for n in graph_nx.nodes() if graph_nx.nodes[n]['test'] == True],
      dtype=np.int32)
  is_train = np.ones((num_data), dtype=np.bool)
  is_train[val_data] = False
  is_train[test_data] = False
  train_data = np.array([n for n in range(num_data) if is_train[n]],
                        dtype=np.int32)

  train_edges = [
      (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
  ]
  edges = np.array(edges, dtype=np.int32)
  train_edges = np.array(train_edges, dtype=np.int32)

  # Process labels
  if isinstance(list(class_map.values())[0], list):
    num_classes = len(list(class_map.values())[0])
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], :] = np.array(class_map[k])
  else:
    num_classes = len(set(class_map.values()))
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], class_map[k]] = 1

  if normalize:
    train_ids = np.array([
        id_map[n]
        for n in graph_nx.nodes()
        if not graph_nx.nodes[n]['val'] and not graph_nx.nodes[n]['test']
    ])
    train_feats = feats[train_ids]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

  def _construct_adj(edges):
    adj = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
                        shape=(num_data, num_data))
    adj += adj.transpose()
    return adj

  train_adj = _construct_adj(train_edges)
  full_adj = _construct_adj(edges)

  train_feats = feats[train_data]
  test_feats = feats

  tf.logging.info('Data loaded, %f seconds.', time.time() - start_time)
  return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data



##############################
#### graph dataset loader ####
##############################


def encode_onehot(labels):
  """Encoder label from number to one-hot"""
  classes = set(labels)
  classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                  enumerate(classes)}
  labels_onehot = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int32)
  return labels_onehot


def load_ne_data_transductive(data_prefix, dataset_str, precalc, split=[0.7,0.2,0.1],normalize=True):
  """load data from graph and preprocessing: 10% train, 20% validation, 70% test"""   
  print('Loading data from graph...'.format(dataset_str))
  names = ['adj', 'feature', 'label']
  objects = []
  for i in range(len(names)):
      with open("data/{}/{}.{}.pkl".format(dataset_str, dataset_str, names[i]), 'rb') as f:
          if sys.version_info > (3, 0):
              objects.append(pkl.load(f, encoding='latin1'))
          else:
              objects.append(pkl.load(f))

  adj, features, labels = tuple(objects)
        
  num_data = features.shape[0]
  idx = range(num_data)
  # split train / val / test nodes
  idx_, test_data = train_test_split(idx, test_size=split[2],random_state=FLAGS.seed)
  train_data, val_data = train_test_split(idx_, test_size=split[1]/(split[0]+split[1]),random_state=FLAGS.seed)
  
  is_train = np.ones((num_data), dtype=np.bool)
  is_train[val_data] = False
  is_train[test_data] = False

  y_train = np.zeros(labels.shape)
  y_val = np.zeros(labels.shape)
  y_test = np.zeros(labels.shape)
  y_train[train_data, :] = labels[train_data, :]
  y_val[val_data, :] = labels[val_data, :]
  y_test[test_data, :] = labels[test_data, :]

  train_mask = sample_mask(train_data, labels.shape[0])
  val_mask = sample_mask(val_data, labels.shape[0])
  test_mask = sample_mask(test_data, labels.shape[0])

  if normalize:
    # Row-normalize feature matrix
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)    
    
  features = features.todense()
  train_feats = features
  test_feats = features

  if precalc:
    train_feats = adj.dot(train_feats)
    train_feats = np.hstack((train_feats, features))
    test_feats = train_feats

  return (adj, train_feats, test_feats, y_train, y_val, y_test,
          train_mask, val_mask, test_mask, train_data, val_data, test_data,
          num_data)


def load_ne_data_transductive_sparse(data_prefix, dataset_str, precalc, split=[0.7,0.2,0.1],normalize=True):
  """load data from graph and preprocessing: 10% train, 20% validation, 70% test"""   
  print('Loading data from graph...'.format(dataset_str))
  print('split: ' + str(split))
  names = ['adj', 'feature', 'label']
  objects = []
  for i in range(len(names)):
      with open("data/{}/{}.{}.pkl".format(dataset_str, dataset_str, names[i]), 'rb') as f:
          if sys.version_info > (3, 0):
              objects.append(pkl.load(f, encoding='latin1'))
          else:
              objects.append(pkl.load(f))

  adj, features, labels = tuple(objects)
        
  num_data = features.shape[0]
  idx = range(num_data)
  # split train / val / test nodes
  idx_, test_data = train_test_split(idx, test_size=split[2],random_state=FLAGS.seed)
  train_data, val_data = train_test_split(idx_, test_size=split[1]/(split[0]+split[1]),random_state=FLAGS.seed)
  print('dataset: ' + dataset_str + 'train: {} val: {} test: {}'.format(len(train_data), len(val_data), len(test_data)))
  is_train = np.ones((num_data), dtype=np.bool)
  is_train[val_data] = False
  is_train[test_data] = False

  y_train = np.zeros(labels.shape)
  y_val = np.zeros(labels.shape)
  y_test = np.zeros(labels.shape)
  y_train[train_data, :] = labels[train_data, :]
  y_val[val_data, :] = labels[val_data, :]
  y_test[test_data, :] = labels[test_data, :]

  train_mask = sample_mask(train_data, labels.shape[0])
  val_mask = sample_mask(val_data, labels.shape[0])
  test_mask = sample_mask(test_data, labels.shape[0])

  if normalize:
    # Row-normalize feature matrix
    normalize_features(features)

  return (adj, features, y_train, y_val, y_test,
          train_mask, val_mask, test_mask, train_data, val_data, test_data,
          num_data)


def tab_printer(args):
  """
  Function to print the logs in a nice tabular format.
  :param args: Parameters used for the model.
  """
  keys = sorted(args.keys())
  from texttable import Texttable
  t = Texttable()
  t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
  print(t.draw())


def normalize_features(features):
    """Row-normalize feature matrix into sparse matrix format"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features_nonzero(features,normalize=True,threshold=100):
    """Process nonzero feature index and value, trunc or pad for fixed length"""
    if normalize == True:
      features = normalize_features(features).tolil()
    else:
      features = features.tolil()
    width = min(max(map(lambda x: len(x), features.rows)), threshold) # max nonzero number
    idx = features.rows
    val = features.data
    idx_pad = []
    val_pad = []
    for row, d in zip(idx, val):
      if len(row) <= width:
        # padding
        idx_pad.append  (np.pad(np.array(row)+1,(0,width - len(row)), 'constant', constant_values=(0.))) # bias=1 for zero vector
        val_pad.append(np.pad(d, (0,width - len(row)), 'constant', constant_values=(0.)))
      else:
        # choose width number of nonzero feature
        choice_id = np.random.choice(range(len(row)), width, replace=False)
        idx_pad.append(np.array(row)[choice_id] + 1)
        val_pad.append(np.array(d)[choice_id])
    return np.array(idx_pad, dtype=np.int32), np.array(val_pad, dtype=np.float32), width