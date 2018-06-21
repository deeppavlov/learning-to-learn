import tensorflow as tf
from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import construct_dict_without_none_entries


class ArtDer(Meta):

    @staticmethod
    def check_kwargs(**kwargs):
        pass

    def _create_optimizer_states(self, num_exercises, var_scope, gpu_idx):
        return list()

    @staticmethod
    def _shuffle(t):
        with tf.device('/cpu:0'):
            return tf.random_shuffle(t)

    def _get_indices(self, num_indices, batch_size):
        with tf.name_scope('get_indices'):
            s = tf.slice(self._indices, [0], batch_size)
            relevant_indices = tf.concat([self._shuffle(s) for _ in range(10)], 0)
            return tf.reshape(tf.slice(relevant_indices, [0], [num_indices]), [1, -1])

    def _get_indices_line(self, num_indices, num_sets, batch_size):
        with tf.name_scope('get_indices_line'):
            num_repeats = tf.cast(
                tf.floordiv(
                    tf.to_float(num_indices) * tf.to_float(num_sets),
                    tf.to_float(tf.reshape(batch_size, []))
                ),
                tf.int32
            ) + 1
            # print("(ArtDer._get_index_sets)num_repeats:", num_repeats)
            # with tf.device('/cpu:0'):
            #     num_repeats = tf.Print(
            #         num_repeats,
            #         [num_repeats],
            #         message="\n(ArtDer._get_index_sets)num_repeats:",
            #         summarize=400,
            #     )
            s = tf.slice(self._indices, [0], batch_size)
            # with tf.device('/cpu:0'):
            #     s = tf.Print(
            #         s,
            #         [s],
            #         message="\n(ArtDer._get_index_sets)s:",
            #         summarize=400,
            #     )
            i0 = tf.constant(0)
            line = tf.zeros([0], dtype=tf.int32)
            # print("(ArtDer._get_index_sets)num_repeats:", num_repeats)
            # print("(ArtDer._get_index_sets)s:", s)
            c = lambda i, l: i < num_repeats
            b = lambda i, l: [
                i + 1,
                tf.concat(
                    [l, self._shuffle(s)],
                    0
                )
            ]
            num_iter, full_line = tf.while_loop(
                c, b, loop_vars=[i0, line],
                # shape_invariants=[i0.get_shape(), tf.TensorShape([None, tf.Dimension(num_indices)])],
                shape_invariants=[i0.get_shape(), tf.TensorShape([None])]
            )
            return full_line

    def _get_index_sets(self, num_indices, batch_size, num_sets):
        with tf.name_scope('get_indices'):
            indices_line = self._get_indices_line(num_indices, num_sets, batch_size)
            # with tf.device('/cpu:0'):
            #     indices_line = tf.Print(
            #         indices_line,
            #         [indices_line],
            #         message="\n(ArtDer._get_index_sets)indices_line:",
            #         summarize=400,
            #     )
            i0 = tf.constant(0)
            indices = tf.ones([0, num_indices], dtype=tf.int32)
            c = lambda i, m: i < num_sets
            b = lambda i, m: [
                i + 1,
                tf.concat(
                    [
                        indices,
                        tf.reshape(tf.slice(indices_line, [i*num_indices], [num_indices]), [1, -1])
                    ],
                    axis=0)
            ]
            num_iter, index_sets = tf.while_loop(
                c, b, loop_vars=[i0, indices],
                # shape_invariants=[i0.get_shape(), tf.TensorShape([None, tf.Dimension(num_indices)])],
                shape_invariants=[i0.get_shape(), tf.TensorShape([None, None])]
            )
            return index_sets

    def _get_batch_size(self, opt_ins):
        ovs = list(opt_ins.values())
        if 'o' in ovs[0]:
            iv = ovs[0]['o']
        elif 'sigma' in ovs[0]:
            iv = ovs[0]['sigma']
        else:
            iv = None
        if isinstance(iv, list):
            t = iv[0]
        else:
            t = iv
        ndim = self._get_opt_ins_ndim(opt_ins)
        # print("(ArtDer._get_batch_size)ndim:", ndim)
        if ndim == 3:
            bdim = 1
        elif ndim == 2:
            bdim = 0
        else:
            bdim = None
        with tf.name_scope('get_batch_size'):
            sh = tf.shape(t)
            return tf.slice(sh, [bdim], [1])

    @staticmethod
    def _perform_gathering(
            o_index_sets,
            sigma_index_sets,
            o,
            sigma,
            gather_dim,
    ):
        go = tf.gather(
            o,
            o_index_sets,
            axis=gather_dim
        )
        gs = tf.gather(
            sigma,
            sigma_index_sets,
            axis=gather_dim
        )
        return go, gs

    def _prepare_for_concatenation(self, unprep, gather_dim):
        if self._selection_application == 'shuffle':
            if isinstance(unprep, list):
                prep = list()
                for el in unprep:
                    prep.append(
                        tf.reduce_mean(el, gather_dim)
                    )
            else:
                prep = tf.reduce_mean(unprep, gather_dim)
        elif self._selection_application == 'mean':
            if isinstance(unprep, list):
                prep = list()
                for el in unprep:
                    prep.append(
                        tf.reduce_mean(
                            el,
                            axis=gather_dim + 1
                        )
                    )
            else:
                prep = tf.reduce_mean(
                    unprep,
                    axis=1+gather_dim
                )
        else:
            prep = None
        return prep

    def _perform_appending(self, opt_ins, o_index_sets, sigma_index_sets, batch_size):
        ndim = self._get_opt_ins_ndim(opt_ins)
        if ndim == 3:
            gather_dim = 1
        elif ndim == 2:
            gather_dim = 0
        else:
            gather_dim = None
        batch_size = tf.to_float(tf.reshape(batch_size, []))
        factor = tf.to_float(batch_size) / (tf.to_float(self._num_sel) + tf.to_float(batch_size))
        with tf.name_scope('perform_appending'):
            for ok, ov in opt_ins.items():
                with tf.name_scope(ok):
                    o = ov['o']
                    sigma = ov['sigma']
                    if isinstance(o, list):
                        gathered_o = list()
                        gathered_sigma = list()
                        for ot, st in zip(o, sigma):
                            go, gs = self._perform_gathering(
                                o_index_sets,
                                sigma_index_sets,
                                ot,
                                st,
                                gather_dim,
                            )
                            gathered_o.append(go)
                            gathered_sigma.append(gs)


                    else:
                        gathered_o, gathered_sigma = self._perform_gathering(
                            o_index_sets,
                            sigma_index_sets,
                            o,
                            sigma,
                            gather_dim,
                        )
                    # print("(ArtDer._perform_appending)gathered_o:", gathered_o)
                    # print("(ArtDer._perform_appending)gathered_sigma:", gathered_sigma)
                    prep_o = self._prepare_for_concatenation(gathered_o, gather_dim)
                    prep_sigma = self._prepare_for_concatenation(gathered_sigma, gather_dim)
                    # print("(Artder._perform_appending)prep_o:", prep_o)
                    # print("(Artder._perform_appending)prep_sigma:", prep_sigma)
                    # print("(Artder._perform_appending)o:", o)
                    # print("(Artder._perform_appending)sigma:", sigma)
                    if isinstance(o, list):
                        ov['o'] = [factor * tf.concat([ot, self._sel_contribution * sel_ot], gather_dim)
                                   for ot, sel_ot in zip(o, prep_o)]
                        ov['sigma'] = [tf.concat([st, sel_st], gather_dim)
                                       for st, sel_st in zip(sigma, prep_sigma)]
                    else:
                        ov['o'] = factor * tf.concat([o, self._sel_contribution * prep_o], gather_dim)
                        ov['sigma'] = tf.concat([sigma, prep_sigma], gather_dim)
        return opt_ins

    def _append_shuffle(self, optimizer_ins):
        batch_size = self._get_batch_size(optimizer_ins)
        with tf.name_scope('collect_indices'):
            o_index_sets = self._get_index_sets(self._num_sel, batch_size, 1)
            tr = tf.transpose(o_index_sets)
            tr = self._shuffle(tr)
            sigma_index_sets = tf.transpose(tr)
        optimizer_ins = self._perform_appending(optimizer_ins, o_index_sets, sigma_index_sets, batch_size)
        return optimizer_ins

    def _append_mean(self, optimizer_ins):
        batch_size = self._get_batch_size(optimizer_ins)
        with tf.name_scope('collect_indices'):
            o_index_sets = self._get_index_sets(self._selection_size, batch_size, self._num_sel)
        optimizer_ins = self._perform_appending(optimizer_ins, o_index_sets, o_index_sets, batch_size)
        return optimizer_ins

    def _append_selections(
            self,
            optimizer_ins
    ):
        with tf.name_scope('append_selection'):
            if self._selection_application == 'shuffle':
                optimizer_ins = self._append_shuffle(optimizer_ins)
            elif self._selection_application == 'mean':
                optimizer_ins = self._append_mean(optimizer_ins)
        return optimizer_ins

    def _optimizer_core(self, optimizer_ins, states, gpu_idx, permute=False):
        # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
        # optimizer_ins = self._forward_permute(optimizer_ins)

        self._append_selections(optimizer_ins)

        self._multiply_by_factor(
            optimizer_ins,
            dict(
                sigma=self._learning_rate
            )
        )
        return self._empty_core(optimizer_ins)

    def __init__(
            self,
            pupil,
            regime='train',
            additional_metrics=None,
            flags=None,
            # selection_size=32,  # ignored if selection_application is shuffle
            # num_sel=1,
            base_optimizer_type='sgd',
            selection_application='mean',
            matrix_mod='phi_and_psi',
            no_end=False,
    ):
        """
        :param regime:
        :param additional_metrics:
        :param flags: a list containing some of the following
            'summarize_opt_ins': if present summary operations for optimizer inputs ('o', 's', 'sigma') are created
            'opt_ins_substitution': if present optimizer ins will be replaced with constant tensors. To specify them
                got to line "if 'opt_ins_substitution' in self._flags:" and choose your option
        """
        if additional_metrics is None:
            additional_metrics = list()
        if flags is None:
            flags = list()

        self._pupil = pupil
        self._regime = regime
        self._additional_metrics = additional_metrics
        self._flags = flags
        self._normalizing = None
        self._inp_gradient_clipping = None

        self._base_optimizer_type = base_optimizer_type
        self._selection_application = selection_application

        self._learning_rate = tf.placeholder(tf.float32, name='learning_rate', shape=[])
        self._sel_contribution = tf.placeholder(tf.float32, name='sel_contribution', shape=[])
        self._selection_size = tf.placeholder(tf.int32, name='selection_size', shape=[])
        self._num_sel = tf.placeholder(tf.int32, name='num_sel', shape=[])

        self._get_theta = False
        self._get_omega_and_beta = False
        self._matrix_mod = matrix_mod
        self._hooks = dict(
            train_with_meta_optimizer_op=None,
            reset_optimizer_inference_pupil_storage=None,
            loss=None,
            pupil_trainable_initializers=None,
            train_optimizer_summary=None,
            learning_rate=None,
            sel_contribution=None,
            selection_size=None,
            num_sel=None,
        )

        for add_metric in self._additional_metrics:
            self._hooks[add_metric] = None
        self._hooks['learning_rate'] = self._learning_rate
        self._hooks['sel_contribution'] = self._sel_contribution
        self._hooks['selection_size'] = self._selection_size
        self._hooks['num_sel'] = self._num_sel
        self._debug_tensors = list()

        self._indices = tf.constant(
            [i for i in range(100000)],
            dtype=tf.int32
        )

        if regime == 'inference':
            pass
        else:
            print('Only inference regime is supported')
            raise NotImplementedError

        self._inference_graph()

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)
