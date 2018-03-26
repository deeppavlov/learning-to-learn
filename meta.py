import tensorflow as tf
from useful_functions import (construct, get_keys_from_nested, get_obj_elem_by_path, device_name_scope,
                              write_elem_in_obj_by_path, stop_gradient_in_nested, compose_save_list, average_gradients,
                              retrieve_from_inner_dicts, distribute_into_inner_dicts, print_optimizer_ins)


LEARNING_RATE_FOR_EMPTY_CORE = 1.


class Meta(object):

    @staticmethod
    def _stack_different_exercises_variables(variables):
        """stack variables from different checkpoints or permutations.
         Stacking is performed along dimension which is last to tensor inner dimensions
         Args:
             variables - dictionary with _pupil variables.
             Each dictionary value is either a list of lists of variables or a list of variables.
             In the first case inner list is a list of variables across different checkpoints
             and outer listing is made across varibles of similar function, e. g. weights from different layers.
             In the second case list is a list a list of variables across different checkpoints."""
        stacked = dict()
        for k, v in variables.items():
            if isinstance(v[0], list()):
                stacked[k] = list()
                for one_var in v:
                    stacked[k].append(tf.stack(one_var))
            else:
                stacked[k] = tf.stack(v)
        return stacked

    @staticmethod
    def _gpu_idx_borders(gpu_map):
        borders = list()
        start = 0
        for idx, gpu_idx in enumerate(gpu_map):
            if gpu_map[start] != gpu_idx:
                borders.append([start, idx])
                start = idx
        borders.append([start, len(gpu_map)])
        return borders

    @staticmethod
    def _stack_placeholders(gpu_borders, placeholders):
        if isinstance(placeholders[0], list):
            stacked_by_gpu = [list() for _ in range(len(gpu_borders))]
            for gpu_idx, borders in enumerate(gpu_borders):
                with tf.device('/gpu:%s' % gpu_idx):
                    ex_placeholders = placeholders[borders[0]:borders[1]]
                    for unr_pl in zip(*ex_placeholders):
                        stacked_by_gpu[gpu_idx].append(tf.stack(unr_pl))
        else:
            stacked_by_gpu = list()
            for borders in gpu_borders:
                stacked_by_gpu.append(tf.stack(placeholders[borders[0]:borders[1]]))
        return stacked_by_gpu

    @staticmethod
    def _stack_trainable_variables(gpu_borders, trainable):
        stacked_tmpl = construct(trainable[0])
        stacked_by_gpu = [construct(trainable[0]) for _ in gpu_borders]
        for ok, ov in stacked_tmpl.items():
            for ik in ov.keys():
                for gpu_idx, borders in enumerate(gpu_borders):
                    with tf.device('/gpu:%s' % gpu_idx):
                        stacked_by_gpu[gpu_idx][ok][ik] = tf.stack(
                            [tr[ok][ik] for tr in trainable[borders[0]:borders[1]]])
        return stacked_by_gpu

    @staticmethod
    def _retrieve_and_unstack_trainable_variables(num_exercises, trainable_by_gpu):
        """trainable_by_gpu is a list of dicts each containing layer information, e. g.:
            embedding_layer: {'matrix': t1, 'o': t2, 's': t3, 'sigma': t4}
        t1, t2 and so on are stacked tensors for exercises on gpu.
        This method retrieves only tensors with trainable variables values, unstacks them and distributes among
        num_exercises dicts of structure similar to trainable_by_gpu[i]"""
        allowed_keys = ['matrix', 'bias']
        tmpl = construct(trainable_by_gpu[0])
        unstacked = [dict() for _ in range(num_exercises)]
        for ok, ov in tmpl.items():
            for d in unstacked:
                d[ok] = dict()
            for ik in ov.keys():
                allowed = False
                for k in allowed_keys:
                    allowed = allowed or (k == ik)
                if allowed:
                    to_distribute = list()
                    for tr in trainable_by_gpu:
                        to_distribute.extend(tf.unstack(tr[ok][ik]))
                    for ex_idx, d in enumerate(unstacked):
                        d[ok][ik] = to_distribute[ex_idx]
        return unstacked

    @staticmethod
    def _stack_storages(gpu_borders, storages):
        stacked_tmpl = construct(storages[0])
        stacked_by_gpu = [construct(storages[0]) for _ in gpu_borders]
        paths = get_keys_from_nested(stacked_tmpl)
        for path in paths:
            for gpu_idx, borders in enumerate(gpu_borders):
                with tf.device('/gpu:%s' % gpu_idx):
                    write_elem_in_obj_by_path(
                        stacked_by_gpu[gpu_idx], path,
                        tf.stack(
                            [get_obj_elem_by_path(stor, path) for stor in storages])
                    )
        return stacked_by_gpu

    @staticmethod
    def _unstack_storages(num_exercises, storages):
        tmpl = construct(storages[0])
        unstacked = [construct(tmpl) for _ in range(num_exercises)]
        paths = get_keys_from_nested(tmpl)
        for path in paths:
            to_distribute = list()
            for st in storages:
                to_distribute.extend(tf.unstack(get_obj_elem_by_path(st, path)))
            for unst, distr in zip(unstacked, to_distribute):
                write_elem_in_obj_by_path(unst, path, distr)
        return unstacked

    @classmethod
    def _stack_exercises(
            cls,
            gpu_borders,
            pupil_grad_eval_inputs,
            pupil_grad_eval_labels,
            optimizer_grad_inputs,
            optimizer_grad_labels,
            pupil_trainable_variables,
            pupil_grad_eval_pupil_storage,
            optimizer_grad_pupil_storage
    ):
        pupil_grad_eval_inputs = cls._stack_placeholders(gpu_borders, pupil_grad_eval_inputs)
        pupil_grad_eval_labels = cls._stack_placeholders(gpu_borders, pupil_grad_eval_labels)
        optimizer_grad_inputs = cls._stack_placeholders(gpu_borders, optimizer_grad_inputs)
        optimizer_grad_labels = cls._stack_placeholders(gpu_borders, optimizer_grad_labels)
        pupil_trainable_variables = cls._stack_trainable_variables(gpu_borders, pupil_trainable_variables)
        pupil_grad_eval_pupil_storage = cls._stack_storages(gpu_borders, pupil_grad_eval_pupil_storage)
        optimizer_grad_pupil_storage = cls._stack_storages(gpu_borders, optimizer_grad_pupil_storage)
        return pupil_grad_eval_inputs, pupil_grad_eval_labels, optimizer_grad_inputs, optimizer_grad_labels, \
            pupil_trainable_variables, pupil_grad_eval_pupil_storage, optimizer_grad_pupil_storage

    @staticmethod
    def _stack_duplicate_o_s(optimizer_ins):
        """Stacking if one matrix is used several times"""
        stacked = dict()
        stack_keys = ['o', 's']
        for k, v in optimizer_ins.items():
            stacked[k] = construct(v)
            for stack_key in stack_keys:
                one_set = v[stack_key]
                if isinstance(one_set, list):
                    united = tf.stack(one_set)
                    united_ndims = len(united.get_shape().as_list())
                    perm = [1, 0] + [i for i in range(2, united_ndims)]
                    stacked[k][stack_key] = tf.transpose(united, perm=perm)
        return stacked

    @staticmethod
    def _make_inputs_and_labels_placeholders(pupil, num_unrollings, num_exercises, gpu_map):
        """If both num_unrollings is not None outputs are lists of lists where
        inner list is for unrollings and outer is for exercises. If num_unrollings is None outputs are lists of
        placeholders."""
        pupil_grad_eval_inputs = list()
        pupil_grad_eval_labels = list()

        optimizer_grad_inputs = list()
        optimizer_grad_labels = list()

        for ex_idx in range(num_exercises):
            if num_unrollings is not None:
                pupil_grad_eval_inputs.append(list())
                pupil_grad_eval_labels.append(list())
                optimizer_grad_inputs.append(list())
                optimizer_grad_labels.append(list())
            with tf.name_scope('exercise_%s' % ex_idx):
                with tf.name_scope('pupil_grad_eval_placeholders'):
                    if num_unrollings is not None:
                        for i in range(num_unrollings):
                            placeholders = pupil.make_inputs_and_labels_placeholders(
                                '/gpu:%s' % gpu_map[ex_idx], 'unrolling_%s' % i)
                            pupil_grad_eval_inputs[ex_idx].append(placeholders['inputs'])
                            pupil_grad_eval_labels[ex_idx].append(placeholders['labels'])
                    else:
                        placeholders = pupil.make_inputs_and_labels_placeholders(
                            '/gpu:%s' % gpu_map[ex_idx], None)
                        pupil_grad_eval_inputs.append(placeholders['inputs'])
                        pupil_grad_eval_labels.append(placeholders['labels'])
                with tf.name_scope('optimizer_grad_placeholders'):
                    if num_unrollings is not None:
                        for i in range(num_unrollings):
                            placeholders = pupil.make_inputs_and_labels_placeholders(
                                '/gpu:%s' % gpu_map[ex_idx], 'unrolling_%s' % i)
                            optimizer_grad_inputs[ex_idx].append(placeholders['inputs'])
                            optimizer_grad_labels[ex_idx].append(placeholders['labels'])
                    else:
                        placeholders = pupil.make_inputs_and_labels_placeholders(
                            '/gpu:%s' % gpu_map[ex_idx], None)
                        optimizer_grad_inputs.append(placeholders['inputs'])
                        optimizer_grad_inputs.append(placeholders['labels'])
        return pupil_grad_eval_inputs, pupil_grad_eval_labels, optimizer_grad_inputs, optimizer_grad_labels

    @staticmethod
    def _create_pupil_variables_and_savers(pupil, num_exercises, gpu_map):
        trainable = list()
        pupil_grad_eval_pupil_storage = list()
        optimizer_grad_pupil_storage = list()
        savers = list()
        for ex_idx in range(num_exercises):
            tr = pupil.create_trainable_variables_dictionary_for_optimizer(
                gpu_map[ex_idx], 'trainable_vars_ex_%s' % ex_idx)
            savers.append(pupil.create_saver(tr))
            trainable.append(tr)
            pupil_grad_eval_pupil_storage.append(pupil.create_storage(
                gpu_map[ex_idx], 'pupil_grad_eval_states_ex_%s' % ex_idx))
            optimizer_grad_pupil_storage.append(
                pupil.create_storage(gpu_map[ex_idx], 'optimizer_grad_states_ex_%s' % ex_idx))
        return trainable, pupil_grad_eval_pupil_storage, optimizer_grad_pupil_storage, savers

    def _add_standard_train_hooks(self):
        self._hooks['pupil_grad_eval_inputs'] = self._pupil_grad_eval_inputs
        self._hooks['pupil_grad_eval_labels'] = self._pupil_grad_eval_labels
        self._hooks['optimizer_grad_inputs'] = self._optimizer_grad_inputs
        self._hooks['optimizer_grad_labels'] = self._optimizer_grad_labels
        self._hooks['pupil_savers'] = self._pupil_savers

    @staticmethod
    def _stop_gradients_in_o_and_s(optimizer_ins):
        for v in optimizer_ins.values():
            v['o'] = tf.stop_gradient(v['o'])
            v['s'] = tf.stop_gradient(v['s'])
        return optimizer_ins

    def _eval_pupil_gradients_for_optimizer_training(
            self, pupil_grad_eval_inputs, pupil_grad_eval_labels,
            pupil_trainable_variables, pupil_grad_eval_pupil_storage):
        loss, optimizer_ins, new_storage = self._pupil.loss_and_opt_ins(
            pupil_grad_eval_inputs, pupil_grad_eval_labels,
            pupil_grad_eval_pupil_storage, opt_ins=pupil_trainable_variables)
        s_vectors, map_ = retrieve_from_inner_dicts(optimizer_ins, 's')
        sigma_vectors = tf.gradients(loss, s_vectors)
        sigma_vectors = [tf.stop_gradient(sigma) for sigma in sigma_vectors]
        optimizer_ins = self._stop_gradients_in_o_and_s(optimizer_ins)
        optimizer_ins = distribute_into_inner_dicts(optimizer_ins, 'sigma', sigma_vectors, map_)
        return optimizer_ins, stop_gradient_in_nested(new_storage), loss

    def _eval_pupil_gradients_for_optimizer_inference(self):
        loss, optimizer_ins, storage_save_ops = self._pupil.loss_and_opt_ins_for_inference()
        s_vectors, map_ = retrieve_from_inner_dicts(optimizer_ins, 's')
        # print('(Meta._eval_pupil_gradients_for_optimizer_inference)map_:', map_)
        sigma_vectors = tf.gradients(loss, s_vectors)
        optimizer_ins = distribute_into_inner_dicts(optimizer_ins, 'sigma', sigma_vectors, map_)
        return optimizer_ins, storage_save_ops, loss

    @staticmethod
    def _empty_core(optimizer_ins):
        # print('(Meta._empty_core)optimizer_ins:')
        # print_optimizer_ins(optimizer_ins)
        with tf.name_scope('phi_and_psi'):
            for k, v in optimizer_ins.items():
                with tf.name_scope(k):
                    with tf.name_scope('phi'):
                        if isinstance(v['o'], list):
                            v['phi'] = tf.concat(v['o'], axis=-2, name='phi')
                        else:
                            v['phi'] = v['o']
                        # if isinstance(v['o'], list):
                        #     v['phi'] = tf.add_n(v['o']) / len(v['o']) * learning_rate**.5
                        # else:
                        #     v['phi'] = v['o'] * learning_rate**.5
                    with tf.name_scope('psi'):
                        # v['psi'] = v['sigma']
                        if isinstance(v['s'], list):
                            v['psi'] = tf.concat(v['sigma'], axis=-2, name='phi')
                        else:
                            v['psi'] = v['sigma']
                        # if isinstance(v['sigma'], list):
                        #     v['psi'] = tf.add_n(v['sigma']) / len(v['sigma']) * learning_rate**.5
                        # else:
                        #     v['psi'] = v['sigma'] * learning_rate**.5
        return optimizer_ins, []

    @staticmethod
    def _compose_mods(optimizer_outs, learning_rate=None):
        with tf.name_scope('pupil_mods'):
            for k, v in optimizer_outs.items():
                with tf.name_scope(k):
                    if 'matrix' in v:
                        ndims = len(v['phi'].get_shape().as_list())
                        # batch_size = v['phi'].get_shape().as_list()[-2]
                        # print("\n(Meta._compose_mods)v['phi'].shape:", v['phi'].get_shape().as_list())
                        # print("\n(Meta._compose_mods)v['psi'].shape:", v['psi'].get_shape().as_list())
                        with tf.name_scope('matrix'):
                            if ndims == 3:
                                eq = 'ijk,ijl->ikl'
                            elif ndims == 2:
                                eq = 'jk,jl->kl'
                            v['matrix_mods'] = tf.einsum(eq, v['phi'], v['psi'])  # / batch_size
                            with tf.device('/cpu:0'):
                                v['matrix_mods'] = tf.Print(
                                    v['matrix_mods'],
                                    [v['matrix_mods']],
                                    message='\n' + k + '\nmatrix:\n')

                    if 'bias' in v:
                        with tf.name_scope('bias'):
                            v['bias_mods'] = tf.reduce_sum(v['psi'], axis=-2)
                            with tf.device('/cpu:0'):
                                v['bias_mods'] = tf.Print(
                                    v['bias_mods'], [v['bias_mods']], message='\n' + k + '\nbias:\n')
            return optimizer_outs

    @staticmethod
    def _add_mods(mods):
        with tf.name_scope('add_modifications'):
            for v in mods.values():
                if 'matrix' in v:
                    v['matrix'] = v['matrix'] + v['matrix_mods']
                if 'bias' in v:
                    v['bias'] = v['bias'] + v['bias_mods']
            return mods

    def _compute_optimizer_gradients(self, loss):
        loss += self._additional_loss
        optimizer_trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='optimizer_trainable_variables')
        return self._optimizer_for_optimizer_training.compute_gradients(loss, var_list=optimizer_trainable_variables)

    @staticmethod
    def _tune_gradients(grads_and_vars):
        grads, v = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, 1.)
        return grads, v

    def _train_graph(self):
        with tf.name_scope('optimizer_train_graph'):
            pupil_grad_eval_inputs, pupil_grad_eval_labels, optimizer_grad_inputs, optimizer_grad_labels, \
                pupil_trainable_variables, pupil_grad_eval_pupil_storage, optimizer_grad_pupil_storage = \
                    self._stack_exercises(
                        self._gpu_borders,
                        self._pupil_grad_eval_inputs,
                        self._pupil_grad_eval_labels,
                        self._optimizer_grad_inputs,
                        self._optimizer_grad_labels,
                        self._pupil_trainable_variables,
                        self._pupil_grad_eval_pupil_storage,
                        self._optimizer_grad_pupil_storage
                    )

            start_losses_by_gpu = list()
            end_losses_by_gpu = list()
            tower_grads = list()

            for gpu_idx in range(self._num_gpus):
                device_name = '/gpu:%s' % gpu_idx
                with tf.device(device_name):
                    with tf.name_scope(device_name_scope(device_name)):
                        optimizer_states = self._create_optimizer_states(
                            self._num_ex_on_gpus[gpu_idx],
                            'train_optimizer_states',
                            gpu_idx
                        )
                        self._create_permutation_matrices(self._num_ex_on_gpus[gpu_idx], gpu_idx)
                        tmp_states = optimizer_states
                        one_gpu_end_loss = 0
                        one_gpu_start_loss = 0
                        for unr_idx in range(self._num_optimizer_unrollings):
                            with tf.name_scope('optimizer_unrolling_%s' % unr_idx):
                                optimizer_ins, pupil_grad_eval_pupil_storage[gpu_idx], start_loss =\
                                    self._eval_pupil_gradients_for_optimizer_training(
                                        pupil_grad_eval_inputs[gpu_idx], pupil_grad_eval_labels[gpu_idx],
                                        pupil_trainable_variables[gpu_idx], pupil_grad_eval_pupil_storage[gpu_idx]
                                    )
                                optimizer_outs, tmp_states = self._optimizer_core(
                                    optimizer_ins, self._num_ex_per_gpu[gpu_idx], tmp_states, gpu_idx)
                                mods = self._compose_mods(optimizer_outs)
                                new_pupil_trainable = self._add_mods(mods)
                                end_loss, _, optimizer_grad_pupil_storage[gpu_idx] = self._pupil.loss_and_opt_ins(
                                    optimizer_grad_inputs[gpu_idx], optimizer_grad_labels[gpu_idx],
                                    optimizer_grad_pupil_storage[gpu_idx], opt_ins=new_pupil_trainable)
                                one_gpu_start_loss += start_loss
                                one_gpu_end_loss += end_loss
                        new_pupil_trainable = self._retrieve_and_unstack_trainable_variables(
                            self._num_exercises, new_pupil_trainable)
                        pupil_grad_eval_pupil_storage = self._unstack_storages(
                            self._num_exercises, pupil_grad_eval_pupil_storage)
                        optimizer_grad_pupil_storage = self._unstack_storages(
                            self._num_exercises, optimizer_grad_pupil_storage)
                        save_ops = compose_save_list(
                            (optimizer_states, tmp_states),
                            name_scope='save_opt_states'
                        ) + compose_save_list(
                            (self._pupil_trainable_variables, new_pupil_trainable), name_scope='save_pupil_train_vars'
                        ) + compose_save_list(
                            (self._pupil_grad_eval_pupil_storage, pupil_grad_eval_pupil_storage),
                            name_scope='save_pupil_grad_eval_pupil_storage'
                        ) + compose_save_list(
                            (self._optimizer_grad_pupil_storage, optimizer_grad_pupil_storage),
                            name_scope='save_optimizer_grad_pupil_storage'
                        )

                        with tf.control_dependencies(save_ops):
                            one_gpu_end_loss = tf.identity(one_gpu_end_loss)
                            start_losses_by_gpu.append(one_gpu_start_loss)
                            end_losses_by_gpu.append(one_gpu_end_loss)
                            grads_and_vars = self._compute_optimizer_gradients(one_gpu_end_loss)
                            tower_grads.append(grads_and_vars)
            with tf.device(self._base_device):
                with tf.name_scope('unite_exercise_gradients'):
                    grads_and_vars = average_gradients(tower_grads)
                    grads, v = self._tune_gradients(grads_and_vars)
                    train_op = self._optimizer_for_optimizer_training.apply_gradients(zip(grads, v))
                    self._hooks['optimizer_train_op'] = train_op

    def _inference_graph(self):
        with tf.name_scope('optimizer_inference_graph'):
            with tf.device('/gpu:0'):
                optimizer_states = self._create_optimizer_states(1, 'inference_optimizer_states', 0)
                optimizer_ins, pupil_save_ops, start_loss = self._eval_pupil_gradients_for_optimizer_inference()
                opt = tf.train.GradientDescentOptimizer(1.)
                grads, vars = zip(*opt.compute_gradients(start_loss))
                optimizer_outs, new_optimizer_states = self._optimizer_core(
                    optimizer_ins, None, optimizer_states, 0)
                for var, gr in zip(vars, grads):
                    with tf.device('/cpu:0'):
                        optimizer_outs['lstm_layer_0']['psi'] = tf.Print(
                            optimizer_outs['lstm_layer_0']['psi'], [gr], message='\n' + var.name + ':\n')
                # print('\n(Meta._inference_graph)optimizer_states:', optimizer_states)
                # print('\n(Meta._inference_graph)new_optimizer_states:', new_optimizer_states)
                mods = self._compose_mods(optimizer_outs, learning_rate=LEARNING_RATE_FOR_EMPTY_CORE)
                # print('\n(Meta._inference_graph)')
                # print_optimizer_ins(mods)
                mods = self._add_mods(mods)
                optimizer_save_states_ops = compose_save_list(
                    (optimizer_states, new_optimizer_states), name_scope='save_optimizer_states')
                # print('\n(Meta._inference_graph)pupil_save_ops:', pupil_save_ops)
                # print('\n(Meta._inference_graph)new_optimizer_states:', new_optimizer_states)
                with tf.control_dependencies(pupil_save_ops+optimizer_save_states_ops):
                    train_op = tf.group(*self._pupil.apply_mods(mods), name='train_with_meta_optimizer_op')
                self._hooks['train_with_meta_optimizer_op'] = train_op
                self._hooks['loss'] = start_loss


