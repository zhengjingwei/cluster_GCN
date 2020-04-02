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

"""Collections of different Models."""

import layers
import metrics
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

class Model(object):
  """Model class to be inherited."""

  def __init__(self, **kwargs):
    allowed_kwargs = {
        'name', 'logging', 'multilabel', 'norm', 'precalc', 'num_layers', 'residual', 'sparse_inputs',
        'valid_dimension', 'act', 'attn_reg', 'gat_layers'
    }
    for kwarg, _ in kwargs.items():
      assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
    name = kwargs.get('name')
    if not name:
      name = self.__class__.__name__.lower()
    self.name = name

    logging = kwargs.get('logging', False)
    self.logging = logging

    self.vars = {}
    self.placeholders = {}

    self.layers = []
    self.activations = []

    self.inputs = None
    self.outputs = None

    self.loss = 0
    self.accuracy = 0
    self.pred = 0
    self.optimizer = None
    self.opt_op = None
    self.multilabel = kwargs.get('multilabel', False)
    self.residual = kwargs.get('residual', False) # True
    self.norm = kwargs.get('norm', False)
    self.precalc = kwargs.get('precalc', False)
    self.num_layers = kwargs.get('num_layers', 2)
    self.sparse_inputs = kwargs.get('sparse_inputs', False)
    self.valid_dimension = kwargs.get('valid_dimension', 0) # max nonzero num of features
    self.attn_reg = kwargs.get('attn_reg', 0)

    # GAT
    self.gat_layers = kwargs.get('gat_layers', [16])

  def _build(self):
    raise NotImplementedError

  def build(self):
    """Wrapper for _build()."""
    with tf.variable_scope(self.name):
      self._build()

    # Build sequential layer model
    self.activations.append(self.inputs)
    for layer in self.layers:
      hidden = layer(self.activations[-1])
      if isinstance(hidden, tuple):
        tf.logging.info('{} shape = {}'.format(layer.name,
                                               hidden[0].get_shape()))
      else:
        tf.logging.info('{} shape = {}'.format(layer.name, hidden.get_shape()))
      self.activations.append(hidden)
    self.outputs = self.activations[-1]

    # Store model variables for easy access
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = variables
    for k in self.vars:
      tf.logging.info((k.name, k.get_shape()))

    # Build metrics
    self._loss()
    self._accuracy()
    self._predict()

    self.opt_op = self.optimizer.minimize(self.loss)

  def _loss(self):
    """Construct the loss function."""
    # Weight decay loss
    if FLAGS.weight_decay > 0.0:
      for var in self.layers[0].vars.values():
        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    if self.attn_reg > 0:
      self.loss += self.attn_reg * tf.nn.l2_loss(self.weights['attention_W'])

    # Cross entropy error
    if self.multilabel:
      self.loss += metrics.masked_sigmoid_cross_entropy(
          self.outputs, self.placeholders['labels'],
          self.placeholders['labels_mask'])
    else:
      self.loss += metrics.masked_softmax_cross_entropy(
          self.outputs, self.placeholders['labels'],
          self.placeholders['labels_mask'])

  def _accuracy(self):
    if self.multilabel:
      self.accuracy = metrics.masked_accuracy_multilabel(
          self.outputs, self.placeholders['labels'],
          self.placeholders['labels_mask'])
    else:
      self.accuracy = metrics.masked_accuracy(self.outputs,
                                              self.placeholders['labels'],
                                              self.placeholders['labels_mask'])

  def _predict(self):
    if self.multilabel:
      self.pred = tf.nn.sigmoid(self.outputs)
    else:
      self.pred = tf.nn.softmax(self.outputs)

  def save(self, sess=None):
    if not sess:
      raise AttributeError('TensorFlow session not provided.')
    saver = tf.train.Saver(self.vars)
    save_path = saver.save(sess, 'tmp/%s.ckpt' % self.name)
    tf.logging.info('Model saved in file:', save_path)

  def load(self, sess=None):
    if not sess:
      raise AttributeError('TensorFlow session not provided.')
    saver = tf.train.Saver(self.vars)
    save_path = 'tmp/%s.ckpt' % self.name
    saver.restore(sess, save_path)
    tf.logging.info('Model restored from file:', save_path)


class GCN(Model):
  """Implementation of GCN model."""

  def __init__(self, placeholders, input_dim, **kwargs):
    super(GCN, self).__init__(**kwargs)

    self.inputs = placeholders['features']
    self.input_dim = input_dim
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    self.build()

  def _build(self):

    self.layers.append(
        layers.GraphConvolution(
            input_dim=self.input_dim if not self.precalc else self.input_dim * 2,
            output_dim=FLAGS.hidden1,
            placeholders=self.placeholders,
            act=tf.nn.relu,
            dropout=True,
            sparse_inputs=self.sparse_inputs, # False
            logging=self.logging,
            norm=self.norm,
            precalc=self.precalc,
            residual=self.residual))

    for _ in range(self.num_layers - 2):
      self.layers.append(
          layers.GraphConvolution(
              input_dim=FLAGS.hidden1 if not self.residual else FLAGS.hidden1*2,
              output_dim=FLAGS.hidden1,
              placeholders=self.placeholders,
              act=tf.nn.relu,
              dropout=True,
              sparse_inputs=False,
              logging=self.logging,
              norm=self.norm,
              precalc=False,
              residual=self.residual))

    self.layers.append(
        layers.GraphConvolution(
            input_dim=FLAGS.hidden1 if not self.residual else FLAGS.hidden1*2,
            output_dim=self.output_dim,
            placeholders=self.placeholders,
            act=lambda x: x,
            dropout=True,
            logging=self.logging,
            norm=False,
            precalc=False,
            residual=self.residual))



class GCN_NFM(Model):
  """Implementation of GCN_NFM model."""

  def __init__(self, placeholders, input_dim, **kwargs):
    super(GCN_NFM, self).__init__(**kwargs)
    assert FLAGS.precalc == False
    self.inputs = placeholders['features']
    self.fm_dropout = placeholders['fm_dropout']

    self.input_dim = input_dim
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    
    self.build()

  def _build(self):

    self.layers.append(
        layers.GraphConvolution(
            input_dim=self.input_dim if not self.residual else self.input_dim*2,
            output_dim=FLAGS.hidden1,
            placeholders=self.placeholders,
            act=tf.nn.relu,
            dropout=True,
            sparse_inputs=self.sparse_inputs,
            logging=self.logging,
            norm=self.norm,
            precalc=self.precalc,
            residual=self.residual))

    for _ in range(self.num_layers - 1):
      self.layers.append(
          layers.GraphConvolution(
              input_dim=FLAGS.hidden1 if not self.residual else FLAGS.hidden1*2,
              output_dim=FLAGS.hidden1,
              placeholders=self.placeholders,
              act=tf.nn.relu,
              dropout=True,
              sparse_inputs=False,
              logging=self.logging,
              norm=self.norm,
              precalc=False,
              residual=self.residual))

    self.project_layer = layers.Dense(
            input_dim=FLAGS.hidden1 * 2,
            output_dim=self.output_dim,
            placeholders=self.placeholders,
            act=lambda x: x,
            dropout=True,
            logging=self.logging,
            norm=False)
    
  
  def build(self):
    """Wrapper for _build()."""
    with tf.variable_scope(self.name):
      self._build()
    
    # FM
    self.fm_embedding = tf.get_variable('fm_embedding', [self.input_dim, FLAGS.hidden1])
    squared_feature_emb = tf.square(self.fm_embedding)

    if self.sparse_inputs:
      summed_feature_emb = tf.sparse_tensor_dense_matmul(self.inputs, self.fm_embedding)
      squared_feature_emb_sum = tf.sparse_tensor_dense_matmul(self.inputs, squared_feature_emb)
    else:
      summed_feature_emb = tf.matmul(self.inputs, self.fm_embedding)
      squared_feature_emb_sum = tf.matmul(self.inputs, squared_feature_emb)
    
    summed_feature_emb_square = tf.square(summed_feature_emb)

    
    self.NFM = 0.5 * tf.subtract(summed_feature_emb_square, squared_feature_emb_sum,name='nfm')
    self.NFM = tf.nn.dropout(self.NFM, 1 - self.fm_dropout)

    # Build sequential layer model
    self.activations.append(self.inputs)
    for layer in self.layers:
      hidden = layer(self.activations[-1])
      if isinstance(hidden, tuple):
        tf.logging.info('{} shape = {}'.format(layer.name,
                                               hidden[0].get_shape()))
      else:
        tf.logging.info('{} shape = {}'.format(layer.name, hidden.get_shape()))
      self.activations.append(hidden)
    self.GCN = self.activations[-1]


    self.outputs = self.project_layer(tf.concat([self.GCN, self.NFM], axis=1))

    # Store model variables for easy access
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = variables
    for k in self.vars:
      tf.logging.info((k.name, k.get_shape()))

    # Build metrics
    self._loss()
    self._accuracy()
    self._predict()

    self.opt_op = self.optimizer.minimize(self.loss)




class GCN_AFM(Model):
  """Implementation of GCN+AFM model."""

  def __init__(self, placeholders, input_dim, **kwargs):
    super(GCN_AFM, self).__init__(**kwargs)
    assert FLAGS.precalc == False
    self.inputs = placeholders['features']
    self.fm_dropout = placeholders['fm_dropout']
    self.attn_dropout = placeholders['attn_dropout']
    self.input_dim = input_dim
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # AFM 
    self.input_idx = placeholders['features_idx']
    self.input_val = placeholders['features_val']
    self.hidden_factor = [FLAGS.attn_dims, FLAGS.hidden1]

    if FLAGS.act == 'relu':
        self.act = tf.nn.relu
    elif FLAGS.act == 'tanh':
        self.act = tf.nn.tanh
    elif FLAGS.act == 'identity':
        self.act = lambda x: x
    else:
        raise ValueError('FLAGS.act value error: ' + str(FLAGS.act))
    
    self.build()

  def _build(self):

    self.layers.append(
        layers.GraphConvolution(
            input_dim=self.input_dim if not self.residual else self.input_dim*2,
            output_dim=FLAGS.hidden1,
            placeholders=self.placeholders,
            act=tf.nn.relu,
            dropout=True,
            sparse_inputs=self.sparse_inputs,
            logging=self.logging,
            norm=self.norm,
            precalc=self.precalc,
            residual=self.residual))

    for _ in range(self.num_layers - 1):
      self.layers.append(
          layers.GraphConvolution(
              input_dim=FLAGS.hidden1 if not self.residual else FLAGS.hidden1*2,
              output_dim=FLAGS.hidden1,
              placeholders=self.placeholders,
              act=tf.nn.relu,
              dropout=True,
              sparse_inputs=False,
              logging=self.logging,
              norm=self.norm,
              precalc=False,
              residual=self.residual))

    self.project_layer = layers.Dense(
            input_dim=FLAGS.hidden1 * 2,
            output_dim=self.output_dim,
            placeholders=self.placeholders,
            act=lambda x: x,
            dropout=True,
            logging=self.logging,
            norm=False)
    
  
  def build(self):
    """Wrapper for _build()."""
    with tf.variable_scope(self.name):
      self._build()
    
    # FM weight init
    self.weights = {}
    self.fm_embedding = tf.get_variable('fm_embedding', [self.input_dim, FLAGS.hidden1])
    self.zero = tf.zeros([1,self.hidden_factor[1]], tf.float32)
    self.fm_embedding = tf.concat([self.zero, self.fm_embedding], axis=0)

    if FLAGS.attention == 'additive':
      self.weights['attention_W'] = tf.get_variable('W_attention', [self.hidden_factor[1], self.hidden_factor[0]],initializer=tf.glorot_uniform_initializer) # (k, t) k=fm_embedding_dim, t=attention_dim
      self.weights['attention_b'] = tf.get_variable('b_attention', [1, self.hidden_factor[0]],initializer=tf.glorot_uniform_initializer) # ,  initializer=tf.zeros_initializer()
      self.weights['attention_p'] = tf.get_variable('p_attention', [self.hidden_factor[0]],initializer=tf.glorot_uniform_initializer)        
    elif FLAGS.attention == 'dot' or FLAGS.attention == 'scaled-dot':
      self.weights['attention_p'] = tf.get_variable('p_attention', [self.hidden_factor[1]])

    """
    squared_feature_emb = tf.square(self.fm_embedding)
    if self.sparse_inputs:
      summed_feature_emb = tf.sparse_tensor_dense_matmul(self.inputs, self.fm_embedding)
      squared_feature_emb_sum = tf.sparse_tensor_dense_matmul(self.inputs, squared_feature_emb)
    else:
      summed_feature_emb = tf.matmul(self.inputs, self.fm_embedding)
      squared_feature_emb_sum = tf.matmul(self.inputs, squared_feature_emb)
    
    summed_feature_emb_square = tf.square(summed_feature_emb)
    self.NFM = 0.5 * tf.subtract(summed_feature_emb_square, squared_feature_emb_sum,name='nfm')
    """

    # ____GCN model____
    self.activations.append(self.inputs)
    for layer in self.layers:
      hidden = layer(self.activations[-1])
      if isinstance(hidden, tuple):
        tf.logging.info('{} shape = {}'.format(layer.name,
                                               hidden[0].get_shape()))
      else:
        tf.logging.info('{} shape = {}'.format(layer.name, hidden.get_shape()))
      self.activations.append(hidden)
    self.GCN = self.activations[-1]

    # ____FM model____
    self.nonzero_embeddings = tf.nn.embedding_lookup(self.fm_embedding, self.input_idx)
    self.nonzero_embeddings = tf.multiply(self.nonzero_embeddings, self.input_val[:,:,tf.newaxis]) # N * M * K


    element_wise_product_list = []
    bias_element_wise_product_list = []
    count = 0
    for i in range(0, self.valid_dimension):
        for j in range(i+1, self.valid_dimension):
            element_wise_product_list.append(tf.multiply(self.nonzero_embeddings[:,i,:], self.nonzero_embeddings[:,j,:]))
            bias_element_wise_product_list.append(tf.to_float(tf.equal(tf.multiply(self.input_val[:,i], self.input_val[:,j]), \
                tf.constant(0.))) * tf.constant(-1e9) ) # zero interaciton set -1e9
            count += 1
    self.bias_element_wise_product = tf.stack(bias_element_wise_product_list) # (M* (M-1)) * N
    # self.LOG.debug('bias_element_wise_product.shape: ' + str(self.bias_element_wise_product.shape))
    self.bias_element_wise_product = tf.transpose(self.bias_element_wise_product, perm=[1,0], name="bias_element_wise_product") # N * (M* (M-1)) 
    self.num_interactions = tf.reduce_sum(tf.to_float(tf.greater(self.bias_element_wise_product, tf.constant(-1e8))), axis=-1, keep_dims=True) # N
    self.num_interactions = self.num_interactions + tf.to_float(tf.equal(self.num_interactions, tf.constant(0.))) # set 0 to 1 for divide op
    self.element_wise_product = tf.stack(element_wise_product_list) # (M* (M-1)) * N * K
    self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1,0,2], name="element_wise_product") # N * (M* (M-1)) * K

    # _________ MLP Layer / attention part _____________
    num_interactions = self.valid_dimension*(self.valid_dimension-1) // 2
    if FLAGS.fm == 'afm':
        # query vector
        if FLAGS.query == 'gcn':
            self.query = self.GCN[:,tf.newaxis,:]
        else:
            self.query = self.weights['attention_p']
        # attention 
        if FLAGS.attention == 'additive':
            self.attention_mul = tf.reshape(tf.matmul(tf.reshape(self.element_wise_product, shape=[-1, self.hidden_factor[1]]), \
                self.weights['attention_W']), shape=[-1, num_interactions, self.hidden_factor[0]]) # None * (M'*(M'-1)) * K
            self.att_1 = self.attention_mul + self.weights['attention_b']
            self.att_2 = self.act(self.att_1)
            if FLAGS.attn_bias == True:
                self.attention_unnormalized = tf.reduce_sum(tf.multiply(self.query, \
                    self.act(self.attention_mul + self.weights['attention_b'])), axis=2) + self.bias_element_wise_product    # None * (M'*(M'-1))
            else:
                self.attention_unnormalized = tf.reduce_sum(tf.multiply(self.query, \
                    self.act(self.attention_mul)), axis=2) + self.bias_element_wise_product    # None * (M'*(M'-1))
            self.attention_unnormalized = self.attention_unnormalized / FLAGS.attn_scale
        elif FLAGS.attention == 'dot':
            # dot-product attention
            self.attention_unnormalized = tf.reduce_sum(tf.multiply(self.query, self.element_wise_product), axis=-1) + self.bias_element_wise_product
        elif FLAGS.attention == 'scaled-dot':
            self.attention_unnormalized = tf.reduce_sum(tf.multiply(self.query, self.element_wise_product), axis=-1) / tf.sqrt(tf.constant(self.hidden_factor[0])) + self.bias_element_wise_product
        elif FLAGS.attention == 'kv':
            # key-value attention
            pass
        self.attention_out = tf.nn.softmax(self.attention_unnormalized)
        if FLAGS.pooling == 'sum':
            self.attention_out = tf.multiply(self.attention_out, self.num_interactions) # N * (M*(M-1))
        self.attention_out = tf.nn.dropout(self.attention_out, 1 - self.attn_dropout) # no dropout

    # _________ Attention-aware Pairwise Interaction Layer _____________
    if FLAGS.fm == 'afm':
        # self.AFM = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), 1, name="afm") # None * K
        self.AFM = tf.reduce_sum(tf.multiply(self.attention_out[:,:,tf.newaxis], self.element_wise_product), 1 ,name="afm")
    else:
        self.AFM = tf.reduce_sum(self.element_wise_product, 1, name="afm") # None * K
    ##self.AFM_FM = tf.reduce_sum(self.element_wise_product, 1, name="afm_fm") # None * K
    ##self.AFM_FM = self.AFM_FM / num_interactions
    self.AFM = tf.nn.dropout(self.AFM, 1 - self.fm_dropout) # dropout

    # ____concate and project____
    self.outputs = self.project_layer(tf.concat([self.GCN, self.AFM], axis=1))

    # Store model variables for easy access
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = variables
    for k in self.vars:
      tf.logging.info((k.name, k.get_shape()))

    # Build metrics
    self._loss()
    self._accuracy()
    self._predict()

    self.opt_op = self.optimizer.minimize(self.loss)

class GAT_NFM(Model):
  """Implementation of GAT_NFM model."""

  def __init__(self, placeholders, input_dim, **kwargs):
    super(GAT_NFM, self).__init__(**kwargs)
    assert FLAGS.precalc == False
    self.inputs = placeholders['features']
    self.fm_dropout = placeholders['fm_dropout']
    self.input_dim = input_dim
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    self.hidden_dims = [input_dim] + self.gat_layers # [input_dim, 64, 16]
    self.num_layers = len(self.hidden_dims) -1
    self.define_weights(self.hidden_dims)
    self.dropout = placeholders['dropout']
    self.fm_dropout = placeholders['fm_dropout']
    self.build()


  def define_weights(self, hidden_dims):
    W = {}
    for i in range(self.num_layers):
        W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))
    Ws_att = {} # attention weight matrix
    for i in range(len(hidden_dims)-1):
        v = {}
        v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
        v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))
        Ws_att[i] = v
    self.W , self.v = W, Ws_att
    self.C = {}

    self.project_layer = layers.Dense(
            input_dim=self.hidden_dims[-1] + FLAGS.fm_dims,
            output_dim=self.output_dim,
            placeholders=self.placeholders,
            act=lambda x: x,
            dropout=True,
            logging=self.logging,
            norm=False)
    

  def __encoder(self, A, H, layer):
        # H = tf.matmul(H, self.W[layer])
        if layer == 0:
          H = layers.sparse_dropout(H, 1 - self.dropout, self.placeholders['num_features_nonzero'])
          H = tf.sparse_tensor_dense_matmul(H, self.W[layer])
        else:
          H = tf.nn.dropout(H, 1 - self.dropout)
          H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer) # attention value
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)


  def graph_attention_layer(self, A, M, v, layer):

    with tf.variable_scope("layer_%s"% layer):
      # drop
      # M = tf.nn.dropout(M, 1 - self.dropout)
      f1 = tf.matmul(M, v[0]) # (?,1)
      f1 = A * f1             # (?,) element-wise product
      f2 = tf.matmul(M, v[1]) # (?,1)
      f2 = A * tf.transpose(f2, [1, 0]) # (?,?) transpose: for the coefficient of h_j 
      logits = tf.sparse_add(f1, f2)      # (?,)

      unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                    values=tf.nn.sigmoid(logits.values),
                                    dense_shape=logits.dense_shape)
      attentions = tf.sparse_softmax(unnormalized_attentions) # (?,)

      attentions = tf.SparseTensor(indices=attentions.indices,
                                    values=attentions.values,
                                    dense_shape=attentions.dense_shape)    # (?,)
      
      # attention dropout, each node is exposed to a stochastically sampled neighborhood
      if FLAGS.gat_dropout > 0:
        attentions = tf.SparseTensor(indices=attentions.indices,
                                    values=tf.nn.dropout(attentions.values, 1.0 - self.placeholders['gat_dropout']),
                                    dense_shape=attentions.dense_shape)    # (?,)

      return attentions


  def build(self):
    """Wrapper for _build()."""
    with tf.variable_scope(self.name):
      self._build()


    # FM
    self.fm_embedding = tf.get_variable('fm_embedding', [self.input_dim, FLAGS.fm_dims])
    squared_feature_emb = tf.square(self.fm_embedding)

    if self.sparse_inputs:
      summed_feature_emb = tf.sparse_tensor_dense_matmul(self.inputs, self.fm_embedding)
      squared_feature_emb_sum = tf.sparse_tensor_dense_matmul(self.inputs, squared_feature_emb)
    else:
      summed_feature_emb = tf.matmul(self.inputs, self.fm_embedding)
      squared_feature_emb_sum = tf.matmul(self.inputs, squared_feature_emb)
    
    summed_feature_emb_square = tf.square(summed_feature_emb)

    self.NFM = 0.5 * tf.subtract(summed_feature_emb_square, squared_feature_emb_sum,name='nfm')
    self.NFM = tf.nn.dropout(self.NFM, 1 - self.fm_dropout)

    self.outputs = self.project_layer(tf.concat([self.GAT, self.NFM], axis=1))

    # Store model variables for easy access
    variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = variables
    for k in self.vars:
      tf.logging.info((k.name, k.get_shape()))

    # Build metrics
    self._loss()
    self._accuracy()
    self._predict()

    self.opt_op = self.optimizer.minimize(self.loss)


  def _build(self):
    H = self.inputs
    for layer in range(self.num_layers):
      H = self.__encoder(self.placeholders['support'], H, layer)
    self.GAT = H


    







