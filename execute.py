import time
import numpy as np
import tensorflow as tf
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from models import GAT
from models import SpGAT
from utils import process
# from utils.tensorboard_logging import Logger


def train(sparse, epochs, lr, patience, l2_coef, hid_units, n_heads, residual, attention_drop, edge_attr_directory, node_features_path, label_path, log_path, train_ratio):
    # flags = tf.app.flags
    # FLAGS = flags.FLAGS
    nb_epochs = epochs

    # flags.DEFINE_string('summaries_dir', log_path, 'Summaries directory')
    if tf.gfile.Exists(log_path):
        tf.gfile.DeleteRecursively(log_path)
    tf.gfile.MakeDirs(log_path)

    checkpt_file = 'pre_trained/mod_test.ckpt'

    dataset = 'know'

    # training params
    batch_size = 1
    nonlinearity = tf.nn.elu
    if sparse:
        model = SpGAT
    else:
        model = GAT
    nhood = 1
    in_drop = attention_drop

    print('Dataset: ' + dataset)
    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))

    adjs, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_attr_name = process.load_data(edge_attr_directory, node_features_path, label_path, train_ratio)
    features, spars = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    features = features[np.newaxis]
    # adj = adj[np.newaxis]
    # adjs = [adj[np.newaxis] for adj in adjs]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    if sparse:
        biases = process.preprocess_adj_bias(adjs[0], to_unweighted=True)  # sparse (indices, values, dense_shape), the graph topologies (unweighted)
        adjs = [tf.SparseTensor(*process.preprocess_adj_bias(adj, to_unweighted=False)) for adj in adjs]
    else:
        biases = process.adj_to_bias(adjs[0], [nb_nodes], nhood=nhood)
    # biases = process.get_bias_mat(adjs[0], [nb_nodes], nhood=nhood)
    print(biases)

    # with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        if sparse:
            # bias_in = tf.sparse_placeholder(dtype=tf.float32)
            bias_in = tf.SparseTensor(*biases)
        else:
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(inputs=ftr_in, 
                            edge_adjs=adjs, 
                            nb_classes=nb_classes, 
                            nb_nodes=nb_nodes, 
                            training=is_train,
                                attn_drop=attn_drop, 
                                ffd_drop=ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity, 
                                edge_attr_name=edge_attr_name)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    micro_f1 = model.micro_f1(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    vmf1_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(log_path + '/train')
        test_summary_writer = tf.summary.FileWriter(log_path + '/test')
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        train_microf1_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0
        val_microf1_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[tr_step*batch_size:(tr_step+1)*batch_size]

                _, summary, loss_value_tr, acc_tr, micro_f1_tr = sess.run([train_op, merged, loss, accuracy, micro_f1],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        # bias_in: bbias,
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: attention_drop, ffd_drop: in_drop})
                print(loss_value_tr)
                train_microf1_avg += micro_f1_tr
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1
                train_summary_writer.add_summary(summary, epoch)

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[vl_step*batch_size:(vl_step+1)*batch_size]
                summary, loss_value_vl, acc_vl, micro_f1_vl = sess.run([merged, loss, accuracy, micro_f1],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        # bias_in: bbias,
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_microf1_avg += micro_f1_vl
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
                test_summary_writer.add_summary(summary, epoch)

            print('Training: loss = %.5f, acc = %.5f, micro_f1 = %.5f | Val: loss = %.5f, acc = %.5f, micro_f1 = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step, train_microf1_avg/tr_step, 
                    val_loss_avg/vl_step, val_acc_avg/vl_step, val_microf1_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vmf1_mx = np.max((val_microf1_avg/vl_step, vmf1_mx))
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx, ', Max Micro-f1', vmf1_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            train_microf1_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0
            val_microf1_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            if sparse:
                bbias = biases
            else:
                bbias = biases[ts_step*batch_size:(ts_step+1)*batch_size]
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    # bias_in: bbias,
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="modified GAT")
    # group = parser.add_mutually_exclusive_group()  # allow to specify options that conflict with each other
    parser.add_argument("-s", "--sparse", action="store_true", help="use sparse operation to reduce memory consumption", required=True)
    parser.add_argument("--epochs", type=int, default=100000, help="number of epochs", required=True)
    parser.add_argument("--lr", type=float, default=0.008, help="learning rate", required=True)
    parser.add_argument("--patience", type=int, default=50, help="for early stopping", required=True)
    parser.add_argument("--l2_coef", type=float, default=0.005, help="l2 regularization coefficient", required=True)
    parser.add_argument('--hid_units', nargs='+', help='numbers of hidden units per each attention head in each layer', required=True)
    parser.add_argument("--n_heads", nargs='+', help="number of attention head", required=True)
    parser.add_argument("--residual", action="store_true", help='use residual connections')
    parser.add_argument("--attention_drop", type=float, default=0.0, help="dropout probability for attention layer", required=True)
    parser.add_argument("--edge_attr_directory", type=str, help="directory storing all edge attribute (.npz file) which stores the sparse adjacency matrix", required=True)
    parser.add_argument("--node_features_path", type=str, help="csv file path for the node features", required=True)
    parser.add_argument("--label_path", type=str, help="csv file path for the ground truth label", required=True)
    parser.add_argument("--log_directory", type=str, help="directory for logging to tensorboard", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8, help="ratio of data used for training (the rest will be used for testing)")

    args = parser.parse_args()
    args.hid_units = [int(x) for x in args.hid_units]
    args.n_heads = [int(x) for x in args.n_heads]

    print()
    print("Hyperparameter:")
    print(args)
    print()
    train(args.sparse, args.epochs, args.lr, args.patience, 
        args.l2_coef, args.hid_units, args.n_heads, args.residual, 
        args.attention_drop, args.edge_attr_directory, 
        args.node_features_path, args.label_path, args.log_directory, args.train_ratio)

