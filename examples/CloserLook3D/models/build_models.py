import tensorflow as tf

from .heads import resnet_scene_segmentation_head
from .backbone import resnet_backbone

class SceneSegModel(object):
    def __init__(self, flat_inputs, is_training, config):
        self.config = config
        self.is_training = is_training

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_inds'] = flat_inputs[ind]
            ind += 1
            self.inputs['cloud_inds'] = flat_inputs[ind]

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']

        with tf.variable_scope('SceneSegModel'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']

            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, weight_decay=config.weight_decay,
                                activation_fn=config.activation_fn, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)
            self.logits, self.logits_cga, self.binary = resnet_scene_segmentation_head(config, self.inputs, F, base_fdim=fdim,
                                                         is_training=is_training, init=config.init,
                                                         weight_decay=config.weight_decay,
                                                         activation_fn=config.activation_fn,
                                                         bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

    def get_coarse_seg_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputs['point_labels'],
                                                                       logits=self.logits,
                                                                       name='cross_entropy')
        cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        self.loss_coarse = cross_entropy
        tf.add_to_collection('losses', self.loss_coarse)
        tf.add_to_collection('segmentation_losses', self.loss_coarse)

    def get_cga_seg_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputs['point_labels'],
                                                                       logits=self.logits_cga,
                                                                       name='cross_entropy_cga')
        cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_cga_mean')

        self.loss_cga = cross_entropy
        tf.add_to_collection('losses_cga', self.loss_cga)
        tf.add_to_collection('segmentation_losses', self.loss_cga)

    def get_binary_loss(self):
        neighbors_idx = self.inputs['neighbors'][0][:, ::self.config.neighbor_step]
        n_neighbors = tf.shape(neighbors_idx)[-1]
        center_labels = self.inputs['point_labels']
        shadow_labels = tf.concat([center_labels, tf.zeros_like(center_labels[:1])], axis=0)
        neighbor_labels = tf.gather(shadow_labels, neighbors_idx, axis=0)
        center_labels = tf.expand_dims(center_labels, -1)
        center_labels = tf.tile(center_labels, [1, n_neighbors])
        binary_labels = tf.cast(tf.equal(center_labels, neighbor_labels), tf.int32)
        binary_labels = tf.reshape(binary_labels, [-1])

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binary_labels,
                                                                       logits=self.binary,
                                                                       name='cross_entropy_binary')

        self.loss_binary = tf.reduce_mean(cross_entropy, name='cross_entropy_binary_mean')
        tf.add_to_collection('losses_binary', self.loss_binary)


