import numpy as np
import tensorflow as tf
import sys
conv1d = tf.layers.conv1d

def attn_head_2(seq, out_sz, adjs, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False, name=''):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # for i, adj in enumerate(adjs[1:]):
            # if i == 0:
            #     continue
        #     logits += tf.get_variable(name+str(i)+'edge_attr_scale', shape=()) * adj
        tf.summary.histogram('hi', logits)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation


def attn_head(seq, out_sz, adjs, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False, name=''):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  # (batch, nodes, f')

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # (batch, nodes, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)  # (batch, nodes, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # (batch, nodes, 1) + (batch, 1, nodes) = (batch, nodes, nodes)
        for i, adj in enumerate(adjs):
            edge_attr_coef = tf.get_variable(name+str(i)+'edge_attr_scale', shape=(), initializer=tf.constant_initializer(1e9))
            tf.summary.histogram(name+str(i)+'edge_attr_scale', tf.math.log(edge_attr_coef))
            logits += edge_attr_coef * adj
        trainable_bias = tf.get_variable(name+str(i)+'biases', shape=(adj.shape), initializer=tf.constant_initializer(-1e9))
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + trainable_bias)
        tf.summary.histogram(name+str(i)+'biases', tf.math.log(-trainable_bias))


        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, edge_adjs, edge_attr_name, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False, name=''):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        for i, (adj, edge_name) in enumerate(zip(edge_adjs, edge_attr_name)):
            edge_attr_coef = tf.get_variable(name+str(i)+'attention', shape=(1), initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.5))
            tf.summary.scalar(name+str(i)+'_' + edge_name + '_attention', tf.math.reduce_sum(edge_attr_coef))
            # print_op = tf.print("tensors:", edge_attr_coef * adj, output_stream=sys.stdout)
            # print_op_2 = tf.print("logits:", logits, output_stream=sys.stdout)
            # with tf.control_dependencies([print_op, print_op_2]):
            # print_op_4 = tf.print(name+str(i)+'edge_attr_scale', edge_attr_coef, output_stream=sys.stdout)
            # with tf.control_dependencies([print_op_4]):
                # logits = tf.sparse_add(logits, edge_attr_coef * adj, threshold=0.05)
            logits = tf.sparse_add(logits, edge_attr_coef * adj)
        trainable_bias = tf.get_variable(name+str(i)+'biases', shape=(1), initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.5))
        logits = tf.sparse_add(logits, trainable_bias * adj_mat)
        tf.summary.scalar(name+str(i)+'biases', tf.math.reduce_sum(trainable_bias))

        # tensorflow doest not support sparse variable
        # trainable_bias = tf.get_variable(name+str(i)+'biases', shape=(adj_mat.get_shape()), initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.5))
        # trainable_bias = tf.Variable(adj_mat)
        # logits = tf.sparse_add(logits, adj_mat.__mul__(trainable_bias)) # element-wise sparse matrix multiplication


        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq
    return activation(ret)  # activation

