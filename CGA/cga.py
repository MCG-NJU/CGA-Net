import tensorflow as tf
import os
import sys

def conv1d_1x1(features,
               out_fdim,
               scope,
               is_training,
               with_bias=False,
               init='xavier',
               weight_decay=0,
               activation_fn='relu',
               bn=True,
               bn_momentum=0.98,
               bn_eps=1e-3):
    """A simple 1x1 1D convolution, ref: https://github.com/zeliu98/CloserLook3D

    Args:
        features: Input features, float32[n_points, in_fdim]
        out_fdim: Output features dim
        scope: name scope
        is_training: True indicates training phase
        with_bias: If True, adds a learnable bias to the output
        init: Weight initializer
        weight_decay: If > 0 , add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution


    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        in_fdim = int(features.shape[-1])
        w = _variable_with_weight_decay('weights',
                                        shape=[in_fdim, out_fdim],
                                        init=init,
                                        wd=weight_decay)
        if with_bias:
            biases = _variable_on_cpu('biases', [out_fdim], tf.constant_initializer(0.0))
            x = tf.matmul(features, w) + biases
        else:
            x = tf.matmul(features, w)
        if bn:
            x = batch_norm(x, is_training=is_training, scope='bn', bn_decay=bn_momentum, epsilon=bn_eps)

        if activation_fn == 'relu':
            x = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

def cga(xyz,
        neighbor_idx,
        features,
        fdim,
        is_training,
        init='xavier',
        weight_decay=0,
        activation_fn='relu',
        bn=True,
        bn_momentum=0.98,
        bn_eps=1e-3):
    """category guided aggregation module.
        Args:
            xyz: point position, Nx3, float32
            neighbor_idx: Nxn_neighbor, int32
            F: input features, NxC, float32
            f_dim: the feature dim, int32
            is_training: True indicates training phase, bool
            init: initialization manner
            weight_decay: If > 0, add L2Loss weight decay multiplied by this float
            activation_fn: Activation function
            bn: If True, add batch norm after convolution
        Returns:
            aggregated features [N, C], binary logits [N, n_neighbors, 2]
        """
    with tf.variable_scope('cga') as sc:
        N = xyz.get_shape()[0].value
        input_dim = features.get_shape()[-1].value
        n_neighbors = neighbor_idx.get_shape()[-1]

        shadow_features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)
        neighbor_features = tf.gather(shadow_features, neighbor_idx)
        center_features = tf.expand_dims(features, axis=1)
        center_features = tf.tile(center_features, [1, n_neighbors, 1])

        shadow_xyzs = tf.concat([xyz, tf.zeros_like(xyz[:1, :])], axis=0)
        neighbor_points = tf.gather(shadow_xyzs, neighbor_idx)
        center_points = tf.expand_dims(xyz, 1)
        xyz_neighbor_diff = center_points - neighbor_points

        concat_features_xyzdiff = tf.concat([center_features, neighbor_features, xyz_neighbor_diff], axis=-1)
        concat_features_xyzdiff = tf.reshape(concat_features_xyzdiff, [N*n_neighbors, input_dim*2+3])
        binary = conv1d_1x1(concat_features_xyzdiff, 2, 'binary_pred', is_training=is_training, with_bias=True,
                        init=init, weight_decay=weight_decay, activation_fn=None, bn=False) # (N*n_neighbors)x2

        binary_soft = tf.nn.softmax(binary, axis=-1)
        binary_soft = tf.reshape(binary_soft, [N, num_neighbors, 2])
        intra_neighbor = binary_soft[:, :, 1:]
        inter_neighbor = binary_soft[:, :, :1]

        '''
        SAM module
        '''
        with tf.variable_scope('sam') as sc:
            # intra_similarity = tf.div_no_nan( # Nxn_neighborsxC
            #     tf.reduce_sum(tf.multiply(features, neighbor_features), axis=-1, keep_dims=True),
            #     1e-10 + tf.multiply(tf.norm(features, axis=-1, keep_dims=True),
            #                         tf.norm(neighbor_features, axis=-1, keep_dims=True)))
            # intra_similarity = tf.div_no_nan(intra_similarity,
            #                                  1e-10 + tf.reduce_sum(intra_similarity, axis=-2, keep_dims=True))
            intra_features = tf.multiply(intra_neighbor, neighbor_features)
            intra_features = tf.div_no_nan(tf.reduce_sum(intra_features, axis=-2), tf.reduce_sum(intra_neighbor, axis=-2))

            intra_features = tf.concat([intra_features, features], axis=-1)
            intra_features = conv1d_1x1(intra_features, fdim, 'intra_fc', is_training=is_training, with_bias=False,
                                        init=init,
                                        weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                        bn_momentum=bn_momentum,
                                        bn_eps=bn_eps)

        '''
        DRM module
        '''
        with tf.variable_scope('drm') as sc:
            concat_features_diff = tf.concat([center_features - neighbor_features, diff_xyz], axis=-1)
            concat_features_diff = tf.reshape(concat_features_diff, [N*n_neighbors, input_dim+3])
            relation = conv1d_1x1(concat_features_diff, fdim, 'inter_rel', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

            relation = tf.reshape(relation, [N, num_neighbors, fdim])
            inter_features = tf.multiply(inter_neighbor, relation)
            inter_features = tf.div_no_nan(tf.reduce_sum(inter_features, axis=-2),
                                           tf.reduce_sum(inter_neighbor, axis=-2) + 1e-10)
            inter_features = tf.concat([inter_features, features], axis=-1)
            inter_features = conv1d_1x1(inter_features, fdim, 'inter_fc', is_training=is_training, with_bias=False,
                                        init=init,
                                        weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                        bn_momentum=bn_momentum,
                                        bn_eps=bn_eps)
        '''
        fusion
        '''
        with tf.variable_scope('fusion') as sc:
            fused_features = tf.concat([features, intra_features, inter_features], axis=-1)
            fused_features = conv1d_1x1(fused_features, fdim, 'fusion_feature', is_training=is_training,
                                            with_bias=False,
                                            init=init,
                                            weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                            bn_momentum=bn_momentum,
                                            bn_eps=bn_eps)

        # logits_cga = conv1d_1x1(fused_features, num_classes, 'segmentation_pred_cga',
        #                                is_training=is_training, with_bias=True,
        #                                init=init, weight_decay=weight_decay, activation_fn=None, bn=False)
    return fused_features, binary


def get_binary_loss(neighbors_idx, labels, pred_binary):
    '''
    :param neighbors_idx: Nxn_neighbors
    :param labels: N
    :return:
    binary loss, float32
    '''
    n_neighbors = tf.shape(neighbors_idx)[-1]
    center_labels = labels
    shadow_labels = tf.concat([center_labels, tf.zeros_like(center_labels[:1])], axis=0)
    neighbor_labels = tf.gather(shadow_labels, neighbors_idx, axis=0)
    center_labels = tf.expand_dims(center_labels, -1)
    center_labels = tf.tile(center_labels, [1, n_neighbors])
    binary_labels = tf.cast(tf.equal(center_labels, neighbor_labels), tf.int32)
    binary_labels = tf.reshape(binary_labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binary_labels,
                                                                   logits=pred_binary,
                                                                   name='cross_entropy_binary')
    loss_binary = tf.reduce_mean(cross_entropy, name='cross_entropy_binary_mean')

    return loss_binary


