import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(ROOT_DIR)

from ..local_aggregation_operators import *


def nearest_upsample_block(layer_ind, inputs, features, scope):
    """
    This Block performing an upsampling by nearest interpolation
    Args:
        layer_ind: Upsampled to which layer
        inputs: a dict contains all inputs
        features: x = [n1, d]
        scope: name scope

    Returns:
        x = [n2, d]
    """

    with tf.variable_scope(scope) as sc:
        upsampled_features = ind_closest_pool(features, inputs['upsamples'][layer_ind], 'nearest_upsample')
        return upsampled_features


def resnet_scene_segmentation_head(config,
                                   inputs,
                                   F,
                                   base_fdim,
                                   is_training,
                                   init='xavier',
                                   weight_decay=0,
                                   activation_fn='relu',
                                   bn=True,
                                   bn_momentum=0.98,
                                   bn_eps=1e-3):
    """A head for scene segmentation with resnet backbone.

    Args:
        config: config file
        inputs: a dict contains all inputs
        F: all stage features
        base_fdim: the base feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        prediction logits [num_points, num_classes]
    """
    with tf.variable_scope('resnet_scene_segmentation_head') as sc:
        fdim = base_fdim
        features = F[-1]

        features = nearest_upsample_block(4, inputs, features, 'nearest_upsample_0')
        features = tf.concat((features, F[3]), axis=1)
        features = conv1d_1x1(features, 8 * fdim, 'up_conv0', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(3, inputs, features, 'nearest_upsample_1')
        features = tf.concat((features, F[2]), axis=1)
        features = conv1d_1x1(features, 4 * fdim, 'up_conv1', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(2, inputs, features, 'nearest_upsample_2')
        features = tf.concat((features, F[1]), axis=1)
        features = conv1d_1x1(features, 2 * fdim, 'up_conv2', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(1, inputs, features, 'nearest_upsample_3')
        features = tf.concat((features, F[0]), axis=1)
        features = conv1d_1x1(features, fdim, 'up_conv3', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = conv1d_1x1(features, fdim, 'segmentation_head', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        logits_coarse = conv1d_1x1(features, config.num_classes, 'segmentation_pred_coarse', is_training=is_training, with_bias=True,
                            init=init, weight_decay=weight_decay, activation_fn=None, bn=False)


        features_cga, pred_binary = cga(xyz=inputs['points'][0],
                                        neighbor_idx=inputs['neighbors'][0][:, ::config.neighbor_step],
                                        features=features,
                                        fdim=32,
                                        is_training=is_training,
                                        init=init,
                                        weight_decay=weight_decay,
                                        activation_fn=activation_fn,
                                        bn=bn,
                                        bn_momentum=bn_momentum,
                                        bn_eps=bn_eps)

        logits_cga = conv1d_1x1(features_cga, config.num_classes, 'segmentation_pred_cga', is_training=is_training, with_bias=True,
                        init=init, weight_decay=weight_decay, activation_fn=None, bn=False)
    return logits_coarse, logits_cga, pred_binary

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
        N = tf.shape(xyz)[0]
        input_dim = features.get_shape()[-1].value
        n_neighbors = tf.shape(neighbor_idx)[-1]

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
        binary_soft = tf.reshape(binary_soft, [N, n_neighbors, 2])
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
            concat_features_diff = tf.concat([center_features - neighbor_features, xyz_neighbor_diff], axis=-1)
            concat_features_diff = tf.reshape(concat_features_diff, [N*n_neighbors, input_dim+3])
            relation = conv1d_1x1(concat_features_diff, fdim, 'inter_rel', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

            relation = tf.reshape(relation, [N, n_neighbors, fdim])
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
            fused_features = conv1d_1x1(fused_features, 2*fdim, 'fusion_feature', is_training=is_training,
                                            with_bias=False,
                                            init=init,
                                            weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                            bn_momentum=bn_momentum,
                                            bn_eps=bn_eps)

        # logits_cga = conv1d_1x1(fused_features, num_classes, 'segmentation_pred_cga',
        #                                is_training=is_training, with_bias=True,
        #                                init=init, weight_decay=weight_decay, activation_fn=None, bn=False)
    return fused_features, binary
