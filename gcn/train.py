import tensorflow as tf
from gcn.utils import *
from gcn.models import GCN
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.25, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

def load_data(dataset):
    """
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    # Find isolated nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = create_mask(idx_train, labels.shape[0])
    val_mask = create_mask(idx_val, labels.shape[0])
    test_mask = create_mask(idx_test, labels.shape[0])

    return adj, features, train_mask, val_mask, test_mask, labels

def give_predictions(features, support, labels, placeholders, sess, model, test_mask):
        feed_dict_val = construct_feed_dict(features, support, labels, test_mask, placeholders)
        predicted_probabilities = sess.run(model.predict(), feed_dict=feed_dict_val)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)
        num_classes = predicted_probabilities.shape[1]
        one_hot_predicted_labels = np.eye(num_classes)[predicted_labels]
        return one_hot_predicted_labels

def train_gcn(model_info, adj, features, train_mask, val_mask, test_mask, labels, correct_labels):
    train_labels, val_labels, test_labels = np.zeros(labels.shape), np.zeros(correct_labels.shape), np.zeros(correct_labels.shape)
    train_labels[train_mask, :] = correct_labels[train_mask, :]
    val_labels[val_mask, :] = correct_labels[val_mask, :]
    test_labels[test_mask, :] = correct_labels[test_mask, :]

    # Running first time, model is initialized
    if model_info['model'] is None:
        features_pro = preprocess_features(features)
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN

        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_pro[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, train_labels.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        model = model_func(placeholders, input_dim=features_pro[2][1], logging=True)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    else:  # If model is provided, continue training it
        sess = model_info['session']
        placeholders = model_info['placeholders']
        model = model_info['model']
        support = model_info['support']
        features_pro = model_info['features']
    
    def evaluate(features_pro, support, labels, mask, placeholders):
        feed_dict_val = construct_feed_dict(features_pro, support, labels, mask, placeholders)
        loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return loss, accuracy

    cost_val = []
    for epoch in range(FLAGS.epochs):
        feed_dict = construct_feed_dict(features_pro, support, train_labels, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc = evaluate(features_pro, support, val_labels, val_mask, placeholders)
        cost_val.append(cost)

        print(f'Epoch {epoch + 1}/{FLAGS.epochs}', "train_acc=", "{:.5f}".format(outs[2]), "train_loss=", "{:.5f}".format(outs[1]), "val_acc=", "{:.5f}".format(acc))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    # Testing
    test_cost, test_acc= evaluate(features_pro, support, test_labels, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc))
    
    pseudo_labels = give_predictions(features_pro, support, labels, placeholders, sess, model, test_mask)
    model_info = {
        'session': sess,
        'placeholders': placeholders,
        'model': model,
        'support':support,
        'features':features_pro
    }
    del train_labels, val_labels, test_labels
    return model_info, pseudo_labels
