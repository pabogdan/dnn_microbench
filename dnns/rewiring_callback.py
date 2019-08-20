import keras
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np
from keras import backend as K


class RewiringCallback(Callback):

    def __init__(self, connectivity_proportion=None,
                 soft_limit=False,
                 fixed_conn=False,
                 noise_coeff=10 ** -6,
                 asserts_on=False):
        super(RewiringCallback, self).__init__()
        self.connectivity_proportion = connectivity_proportion
        self.soft_limit = soft_limit
        self.fixed_conn = fixed_conn
        self._data = {}
        self._batch_rewires = {}
        self.noise_coeff = noise_coeff
        self.asserts_on = asserts_on

    @staticmethod
    def get_kernels_and_masks(model):
        kernels = []
        masks = []
        layers = []
        for layer in model.layers:
            if hasattr(layer, "mask"):
                kernels.append(layer.get_weights()[0])
                masks.append(K.get_value(layer.mask))
                layers.append(layer)
        return kernels, masks, layers

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

        if self.asserts_on:
            for k, m, l, i in zip(self.post_kernels, self.post_masks, self.layers,
                                  np.arange(len(self.layers))):
                # If you invert the mask, are all those entries in kernel == 0?
                # if l.connectivity_level:
                #     assert np.all(k[~m.astype(bool)] == 0)
                # x = K.get_value(l.original_kernel)
                # assert np.all(x[~m.astype(bool)] == 0) or x[~m.astype(bool)].size == 0
                # check that the connectivity is at the correct level
                assumed_prop = np.sum(m) / float(m.size)

                conn_prop = l.connectivity_level
                if conn_prop and not l.connectivity_decay:
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
            # save them in a variable and apply the mask (only rewire active conns)
            masked_sign_diff = (pre_sign * post_sign)*post_m
            need_rewiring = np.where(masked_sign_diff < 0)

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
                # Apply noise only to dormant connections
                post_post_k = pre_k + (noise * post_m)
                post_post_sign = np.sign(post_post_k)
                sign_diff = post_post_sign * pre_sign
                # Disregard active connections, only focus on dormant ones
                rew_candidates_mask = np.zeros(post_k.shape).astype(bool)
                rew_candidates_mask[rewiring_candidates] = True
                sign_diff[np.invert(rew_candidates_mask)] = 0
                chosen_partners = np.where(sign_diff < 0)

            new_m = post_m
            new_m[chosen_partners] = 1
            # enable the new connections
            K.set_value(l.mask, new_m)
            # update original kernel
            # post_k = post_k * new_m
            # K.set_value(l.original_kernel, post_k)
            if self.soft_limit:
                # masked pre inactive (dormant) connections
                masked_pres = post_post_k * (~rew_candidates_mask)
                # clip masked_pres so that they don't drift too far from 0
                masked_pres = np.clip(masked_pres, -1.5, 1.5)
                # add to masked post actives
                masked_posts = post_k * (rew_candidates_mask)
                K.set_value(l.original_kernel, masked_pres + masked_posts)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print("\nEpoch {:3} results:".format(epoch))
        total_num_of_conns = 0
        total_num_active_conns = 0
        for k, m, l in zip(self.post_kernels, self.post_masks, self.layers):
            # report
            total_num_of_conns += m.size
            curr_no_active_connections = np.count_nonzero(m)
            total_num_active_conns += curr_no_active_connections
            conn_level = l.connectivity_level or curr_no_active_connections / float(m.size)
            self._data['no_connections_{}'.format(l.name)] = curr_no_active_connections
            self._data['no_rewires_for_layer_{}'.format(l.name)] = \
                self._batch_rewires["rewirings_for_layer_{}".format(l.name)]
            self._data['proportion_connections_{}'.format(l.name)] = conn_level
            print("Layer {:10} has {:8} connections, corresponding to "
                  "{:>5.1%} of "
                  "the total connectivity".format(
                l.name,
                self._data['no_connections_{}'.format(l.name)],
                self._data['proportion_connections_{}'.format(l.name)]))
            # decay connectivity level
            if hasattr(l, "connectivity_decay") and l.connectivity_decay:
                l.connectivity_level -= l.connectivity_level * l.connectivity_decay
                new_number_of_active_conns = l.get_number_of_active_connections()
                if new_number_of_active_conns != curr_no_active_connections:
                    no_diff = curr_no_active_connections - new_number_of_active_conns
                    rewiring_candidates = np.asarray(np.where(m == 1))
                    choices = np.random.choice(
                        np.arange(rewiring_candidates[0].size),
                        no_diff,
                        replace=False)
                    chosen_partners = tuple(rewiring_candidates[:, choices])
                    m[chosen_partners] = 0
                    K.set_value(l.mask, m)
        global_conn_lvl = total_num_active_conns/float(total_num_of_conns)
        print("Total stats: {:8} active connections, corresponding to {:>5.1%} "
              "of total connectivity".format(
            total_num_active_conns,
            global_conn_lvl))
        self._data['global_connectivity_lvl'] = global_conn_lvl
        logs.update(dict(logs.items() | self._data.items()))
        return logs

    def stats(self):
        return {
            "epoch_data": self._data,
            "batch_data": self._batch_rewires
        }
