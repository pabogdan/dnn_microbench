import keras
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np
from keras import backend as K


class RewiringCallback(Callback):

    def __init__(self, connectivity_proportion=None,
                 soft_limit=False,
                 fixed_conn=False,
                 noise_coeff=10 ** (-6)):
        super(RewiringCallback, self).__init__()
        self.connectivity_proportion = connectivity_proportion
        self.soft_limit = soft_limit
        self.fixed_conn = fixed_conn
        self._data = {}
        self._batch_rewires = {}
        self.noise_coeff = noise_coeff

    @staticmethod
    def get_kernels_and_masks(model):
        kernels = []
        masks = []
        layers = []
        for layer in model.layers:
            if hasattr(layer, "mask"):
                kernels.append(K.get_value(layer.kernel))
                masks.append(K.get_value(layer.mask))
                layers.append(layer)
        return np.asarray(kernels), np.asarray(masks), layers

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        _, _, layers = \
            RewiringCallback.get_kernels_and_masks(self.model)
        for l in layers:
            self._batch_rewires["rewirings_for_layer_{}".format(l.name)] = 0

    def on_batch_begin(self, batch, logs=None):
        # save the weights before updating to compare sign later
        self.pre_kernels, self.pre_masks, _ = \
            RewiringCallback.get_kernels_and_masks(self.model)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        # retrieve the new weights (after a batch)
        self.post_kernels, self.post_masks, self.layers = \
            RewiringCallback.get_kernels_and_masks(self.model)

        for k, m, l, i in zip(self.post_kernels, self.post_masks, self.layers,
                              np.arange(len(self.layers))):
            # If you invert the mask, are all those entries in kernel == 0?
            assert np.all(k[~m.astype(bool)] == 0)
            # if not self.soft_limit:

            # check that the connectivity is at the correct level
            assumed_prop = np.sum(m) / float(m.size)
            if self.connectivity_proportion:
                conn_prop = self.connectivity_proportion[i]
            else:
                conn_prop = l.connectivity_level
            if conn_prop:
                assert (np.isclose(assumed_prop, conn_prop, 0.0001)), \
                    "{} vs. {}".format(assumed_prop, conn_prop)

        for pre_m, post_m in zip(self.pre_masks, self.post_masks):
            # Check that the mask has not changed between batch begin and end
            assert np.all(pre_m == post_m)

        if self.fixed_conn:
            # ASSESSING THE PERFORMANCE OF THE NETWORK WHEN THE CONNECTIVITY
            # IS SPARSE, BUT REWIRING IS DISABLED
            return

        # Let's rewire!
        for pre_m, post_m, pre_k, post_k, l in zip(
                self.pre_masks, self.post_masks,
                self.pre_kernels, self.post_kernels,
                self.layers):
            pre_sign = np.sign(pre_k)
            post_sign = np.sign(post_k)

            # retrieve indices of synapses which require rewiring
            need_rewiring = np.where(pre_sign - post_sign)

            # update the mask by selecting other synapses to be active
            number_needing_rewiring = need_rewiring[0].size
            self._batch_rewires["rewirings_for_layer_{}".format(l.name)] += \
                number_needing_rewiring

            logs.update(self._batch_rewires)
            if number_needing_rewiring == 0:
                continue

            post_m[need_rewiring] = 0
            if not self.soft_limit:
                # HARD REWIRING
                rewiring_candidates = np.asarray(np.where(post_m == 0))
                choices = np.random.choice(
                    np.arange(rewiring_candidates[0].size),
                    number_needing_rewiring,
                    replace=False)
                chosen_partners = tuple(rewiring_candidates[:, choices])
            else:
                # SOFT REWIRING
                rewiring_candidates = np.where(post_m == 0)
                noise = np.random.normal(scale=self.noise_coeff,
                                         size=post_k.shape)
                post_post_k = post_k + noise
                post_post_sign = np.sign(post_post_k)
                sign_diff = post_post_sign - post_sign
                # Disregard active connections, only focus on dormant ones
                rew_candidates_mask = np.zeros(post_k.shape).astype(bool)
                rew_candidates_mask[rewiring_candidates] = True
                sign_diff[np.invert(rew_candidates_mask)] = 0
                chosen_partners = np.where(sign_diff)

            new_m = post_m
            new_m[chosen_partners] = 1
            # enable the new connections
            K.set_value(l.mask, new_m)
            # update original kernel
            post_k = post_k * new_m
            K.set_value(l.original_kernel, post_k)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print("\nEpoch {:3} results:".format(epoch))
        for k, m, l in zip(self.post_kernels, self.post_masks, self.layers):
            self._data['no_connections_{}'.format(l.name)] = np.sum(m)
            self._data['no_rewires_for_layer_{}'.format(l.name)] = \
                self._batch_rewires["rewirings_for_layer_{}".format(l.name)]
            self._data['proportion_connections_{}'.format(l.name)] = np.sum(m) / float(m.size)
            print("Layer {:10} has {:8} connections, corresponding to "
                  "{:>5.1%} of "
                  "the total connectivity".format(
                l.name,
                self._data['no_connections_{}'.format(l.name)],
                self._data['proportion_connections_{}'.format(l.name)]))
        logs.update(dict(logs.items() | self._data.items()))
        return logs

    def stats(self):
        return {
            "epoch_data": self._data,
            "batch_data": self._batch_rewires
        }
