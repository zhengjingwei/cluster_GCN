##############
# Transductive settings
# GCN_AFM:  input sparse feature for GCN
#           input padding nonzero index and values for AFM
###############
"""Main script for training GCN models."""

import time
import models
import numpy as np
import partition_utils
import tensorflow.compat.v1 as tf
import utils

tf.logging.set_verbosity(tf.logging.INFO)
# Set random seed


# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('save_name', './mymodel.ckpt', 'Path for saving model')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.') # ppi
flags.DEFINE_string('data_prefix', 'data', 'Datapath prefix.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('num_clusters', 10, 'Number of clusters.')
flags.DEFINE_integer('bsize', 2, 'Number of clusters for each batch.')
flags.DEFINE_integer('num_clusters_val', 10,
                     'Number of clusters for validation.')
flags.DEFINE_integer('num_clusters_test', 1, 'Number of clusters for test.')
flags.DEFINE_integer('num_layers', 2, 'Number of GCN layers.')
flags.DEFINE_float(
    'diag_lambda', 1,
    'A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement'
)
flags.DEFINE_bool('multilabel', False, 'Multilabel or multiclass.')
flags.DEFINE_bool('layernorm', True, 'Whether to use layer normalization.')
flags.DEFINE_bool(
    'precalc', False,
    'Whether to pre-calculate the first layer (AX preprocessing).')
flags.DEFINE_bool('validation', True,
                  'Print validation accuracy after each epoch.')

# new settings
flags.DEFINE_list('split', [0.7, 0.2, 0.1], 'Data split.')
flags.DEFINE_integer('seed',100, 'Random seed.')
flags.DEFINE_string('model','gcn_afm', 'model name.')
flags.DEFINE_bool('residual', False, 'Whether to use residual connection.')

# FM settings
flags.DEFINE_string('attention', 'additive', 'Attention manner.')  # 'additive', 'dot', 'scaled-dot'
flags.DEFINE_integer('fm_embedding_dims', 16, 'FM embedding dimensions.')
flags.DEFINE_float('fm_dropout', 0, 'FM dropout rate (1 - keep probability).')
flags.DEFINE_float('attn_dropout', 0, 'Attention network dropout rate (1 - keep probability).')
flags.DEFINE_integer('attn_dims', 16 , 'Attention vector dimensions.')
flags.DEFINE_float('attn_scale', 1.0 , 'Attention scale factor.')
flags.DEFINE_boolean('attn_bias',True, 'Attention bias vector.')
flags.DEFINE_float('attn_reg', 10, 'Weight for L2 loss on attention matrix.')
flags.DEFINE_string('fm', 'afm', 'FM model.')  # 'afm', 'nfm'
flags.DEFINE_string('query', 'h', 'Aligment vector for attention.')
flags.DEFINE_boolean('gpu', True, 'Use GPU.')  #
flags.DEFINE_boolean('debug', False, 'Debug mode.')  #
flags.DEFINE_string('act', 'relu', '.')  # 
flags.DEFINE_string('pooling', 'sum', 'Interaction vector pooling manner.') # sum, average, max


seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

# Define model evaluation function
def evaluate(sess, model, val_features_batches, val_support_batches,
             y_val_batches, val_mask_batches, val_data, placeholders):
  """evaluate GCN model."""
  total_pred = []
  total_lab = []
  total_loss = 0
  total_acc = 0

  num_batches = len(val_features_batches[0])
  for i in range(num_batches):
    features_sp_b = val_features_batches[0][i]
    features_idx_b = val_features_batches[1][i]
    features_val_b = val_features_batches[2][i]
    support_b = val_support_batches[i]
    y_val_b = y_val_batches[i]
    val_mask_b = val_mask_batches[i]
    num_data_b = np.sum(val_mask_b)
    if num_data_b == 0:
      continue
    else:
      feed_dict = utils.construct_feed_dict_afm(features_sp_b,features_idx_b,features_val_b, support_b, y_val_b,
                                            val_mask_b, placeholders)
      outs = sess.run([model.loss, model.accuracy, model.outputs],
                      feed_dict=feed_dict)

    total_pred.append(outs[2][val_mask_b])
    total_lab.append(y_val_b[val_mask_b])
    total_loss += outs[0] * num_data_b
    total_acc += outs[1] * num_data_b

  total_pred = np.vstack(total_pred)
  total_lab = np.vstack(total_lab)
  loss = total_loss / len(val_data)
  acc = total_acc / len(val_data)

  micro, macro = utils.calc_f1(total_pred, total_lab, FLAGS.multilabel)
  return loss, acc, micro, macro


def main(unused_argv):
  """Main function for running experiments."""
  # Load data
  utils.tab_printer(FLAGS.flag_values_dict())
  (full_adj, feats, y_train, y_val, y_test,
          train_mask, val_mask, test_mask, train_data, val_data, test_data,
          num_data) = utils.load_ne_data_transductive_sparse(FLAGS.data_prefix, FLAGS.dataset, list(map(float,FLAGS.split)), normalize=False)
  feats_idx, feats_val, max_nnz = utils.preprocess_features_nonzero(feats,normalize=False,threshold=100) # not normalized for FM
  print('max nnz: ', max_nnz)
  feats = utils.normalize_features(feats) # normalized for GCN
  # TODO: AX预计算，省去输入特征，
  # Partition graph and do preprocessing
  if FLAGS.bsize > 1: # multi cluster per epoch
    _, parts = partition_utils.partition_graph(full_adj, np.arange(num_data),
                                               FLAGS.num_clusters)

    parts = [np.array(pt) for pt in parts]
  else:
    (parts, features_batches, support_batches, y_train_batches,
     train_mask_batches) = utils.preprocess(full_adj, feats, y_train,
                                            train_mask, np.arange(num_data),
                                            FLAGS.num_clusters,
                                            FLAGS.diag_lambda)
  # valid & test in the same time
  # validation set
  
  (_, val_features_batches, test_features_batches, val_support_batches, y_val_batches,y_test_batches,
          val_mask_batches, test_mask_batches) = utils.preprocess_val_test_afm(full_adj, feats,feats_idx,feats_val,
                                                                  y_val,val_mask, y_test, test_mask,
                                                                  np.arange(num_data),
                                                                  FLAGS.num_clusters_val,
                                                                  FLAGS.diag_lambda)

  idx_parts = list(range(len(parts)))

  # Define placeholders
  placeholders = {
      'support':
          tf.sparse_placeholder(tf.float32),
      # 'features':
      #     tf.placeholder(tf.float32),
      'features':
          tf.sparse_placeholder(tf.float32),
      'labels':
          tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
      'labels_mask':
          tf.placeholder(tf.int32),
      'dropout':
          tf.placeholder_with_default(0., shape=()),
      'fm_dropout':
          tf.placeholder_with_default(0., shape=()),
      'attn_dropout':
          tf.placeholder_with_default(0., shape=()),
      'num_features_nonzero':
          tf.placeholder(tf.int32)  # helper variable for sparse dropout
  }


  # Create model
  if FLAGS.model == 'gcn_afm':
    placeholders['features_idx'] = tf.placeholder(tf.int32)
    placeholders['features_val'] = tf.placeholder(tf.float32)
    model = models.GCN_AFM(placeholders,
                      input_dim=feats.shape[1],
                      logging=True,
                      multilabel=FLAGS.multilabel,
                      norm=FLAGS.layernorm,
                      precalc=FLAGS.precalc,
                      num_layers=FLAGS.num_layers,
                      residual=False,
                      sparse_inputs=True,
                      valid_dimension=max_nnz,
                      attn_reg=FLAGS.attn_reg)
  else:
    raise ValueError(str(FLAGS.model))

  # Initialize session
  sess = tf.Session()
  
  # Init variables
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  cost_val = []
  acc_val = []
  total_training_time = 0.0
  
  # Train model
  for epoch in range(FLAGS.epochs):
    t = time.time()
    np.random.shuffle(idx_parts)
    if FLAGS.bsize > 1:
      (features_batches, support_batches, y_train_batches,
       train_mask_batches) = utils.preprocess_multicluster_afm(
           full_adj, parts, feats,feats_idx,feats_val, y_train, train_mask,
           FLAGS.num_clusters, FLAGS.bsize, FLAGS.diag_lambda, True)
      for pid in range(len(features_batches[0])):
        # Use preprocessed batch data
        # features_sp_b = features_batches[0][pid], features_batches[1][pid], features_batches[2][pid]
        support_b = support_batches[pid]
        y_train_b = y_train_batches[pid]
        train_mask_b = train_mask_batches[pid]
        # Construct feed dictionary
        feed_dict = utils.construct_feed_dict_afm(features_batches[0][pid], features_batches[1][pid],features_batches[2][pid],
                                              support_b, y_train_b,
                                              train_mask_b, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['fm_dropout']: FLAGS.fm_dropout})
        feed_dict.update({placeholders['attn_dropout']: FLAGS.attn_dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy],
                        feed_dict=feed_dict)
      np.random.shuffle(idx_parts)
    else:
      raise ValueError('FLAGS.bsize must > 1')
      for pid in idx_parts:
        # Use preprocessed batch data
        features_b = features_batches[pid]
        support_b = support_batches[pid]
        y_train_b = y_train_batches[pid]
        train_mask_b = train_mask_batches[pid]
        # Construct feed dictionary
        feed_dict = utils.construct_feed_dict_afm(features_b, support_b, y_train_b,
                                              train_mask_b, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy],
                        feed_dict=feed_dict)

    total_training_time += time.time() - t
    print_str = 'Epoch: %04d ' % (epoch + 1) + 'training time: {:.5f} '.format(
        total_training_time) + 'train_acc= {:.5f} '.format(outs[2])

    # Validation
    if FLAGS.validation:
      cost, acc, micro, macro = evaluate(sess, model, val_features_batches,
                                         val_support_batches, y_val_batches,
                                         val_mask_batches, val_data,
                                         placeholders)
      cost_val.append(cost)
      acc_val.append(acc)
      print_str += 'val_acc= {:.5f} '.format(
          acc) + 'mi F1= {:.5f} ma F1= {:.5f} '.format(micro, macro)

    if epoch == 10:
      db_sz = 20
      nnz_emb, fm_emb, query,att_w,att_b, interaction, num_interaction, att_unnorm, att_out, att_mul,bias, att_1, att_2  = sess.run([model.nonzero_embeddings, \
          model.fm_embedding, model.query, model.weights['attention_W'], model.weights['attention_b'], \
              model.element_wise_product[:db_sz],model.num_interactions, model.attention_unnormalized[:db_sz],model.attention_out[:db_sz], \
          model.attention_mul[:db_sz], model.bias_element_wise_product[:db_sz], model.att_1[:db_sz], model.att_2[:db_sz]],\
                feed_dict=feed_dict)
      print('debug')

    # tf.logging.info(print_str)
    print(print_str)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
        cost_val[-(FLAGS.early_stopping + 1):-1]):
      tf.logging.info('Early stopping...')
      break

    ### use acc early stopping, lower performance than using loss
    # if epoch > FLAGS.early_stopping and acc_val[-1] < np.mean(
    #     acc_val[-(FLAGS.early_stopping + 1):-1]):
    #   tf.logging.info('Early stopping...')
    #   break

  tf.logging.info('Optimization Finished!')

  # Save model
  saver.save(sess, FLAGS.save_name)

  # Load model (using CPU for inference)
  with tf.device('/cpu:0'):
    sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    sess_cpu.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess_cpu, FLAGS.save_name)
    # Testing
    test_cost, test_acc, micro, macro = evaluate(
        sess_cpu, model, test_features_batches, val_support_batches,
        y_test_batches, test_mask_batches, test_data, placeholders)
    print_str = 'Test set results: ' + 'cost= {:.5f} '.format(
        test_cost) + 'accuracy= {:.5f} '.format(
            test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
    tf.logging.info(print_str)


if __name__ == '__main__':
  tf.app.run(main)
