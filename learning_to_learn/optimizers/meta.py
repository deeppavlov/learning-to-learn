from collections import OrderedDict
import tensorflow as tf
from learning_to_learn.useful_functions import construct, get_keys_from_nested, get_obj_elem_by_path, \
    device_name_scope, write_elem_in_obj_by_path, stop_gradient_in_nested, compose_save_list, average_gradients, \
    retrieve_from_inner_dicts, distribute_into_inner_dicts, custom_matmul, values_from_nested, sort_lists_map, \
    global_l2_loss, filter_none_gradients, go_through_nested_with_name_scopes_to_perform_func_and_distribute_results, \
    global_norm, func_on_list_in_nested, append_to_nested, get_substitution_tensor, variable_summaries, \
    apply_to_nested, flatten

from learning_to_learn.tensors import compute_metrics

LEARNING_RATE_FOR_EMPTY_CORE = 1.
CLIP_NORM = 1e+5


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
    def _filter_opt_flow_dict(flow, remaining_inner_keys):
        new = dict()
        for ok, ov in flow.items():
            new[ok] = dict()
            d = new[ok]
            for ik in remaining_inner_keys:
                if ik in flow[ok]:
                    d[ik] = flow[ok][ik]
        return new

    @staticmethod
    def _retrieve_and_unstack_trainable_variables(num_exercises, trainable_on_gpu,
                                                  name_scope='unstack_trainable_variables'):
        """trainable_by_gpu is a list of dicts each containing layer information, e. g.:
            embedding_layer: {'matrix': t1, 'o': t2, 's': t3, 'sigma': t4}
        t1, t2 and so on are stacked tensors for exercises on gpu.
        This method retrieves only tensors with trainable variables values, unstacks them and distributes among
        num_exercises dicts of structure similar to trainable_by_gpu[i]"""
        with tf.name_scope(name_scope):
            allowed_keys = ['matrix', 'bias']
            tmpl = construct(trainable_on_gpu)
            unstacked = [dict() for _ in range(num_exercises)]
            for ok, ov in tmpl.items():
                with tf.name_scope(ok):
                    for d in unstacked:
                        d[ok] = dict()
                    for ik in ov.keys():
                        allowed = False
                        for k in allowed_keys:
                            allowed = allowed or (k == ik)
                        if allowed:
                            for t, d in zip(tf.unstack(trainable_on_gpu[ok][ik], name=ik), unstacked):
                                d[ok][ik] = t
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

    # @staticmethod
    # def _unstack_storage(num_exercises, storage, name_scope='unstack_storage'):
    #     with tf.name_scope(name_scope):
    #         tmpl = construct(storage)
    #         unstacked = [construct(tmpl) for _ in range(num_exercises)]
    #         paths = get_keys_from_nested(tmpl)
    #         key_values, _ = sort_lists_map(paths)
    #         paths = sort_lists_of_ints_and_str(paths)
    #         for path in paths:
    #             to_distribute = list()
    #             to_distribute.extend(tf.unstack(get_obj_elem_by_path(storage, path)))
    #             for unst, distr in zip(unstacked, to_distribute):
    #                 write_elem_in_obj_by_path(unst, path, distr)
    #     return unstacked

    @staticmethod
    def _unstack_storage(num_exercises, storage, name_scope='unstack_storages'):
        with tf.name_scope(name_scope):
            tmpl = construct(storage)
            unstacked = [construct(tmpl) for _ in range(num_exercises)]
            paths = get_keys_from_nested(tmpl)
            key_values, _ = sort_lists_map(paths)
            go_through_nested_with_name_scopes_to_perform_func_and_distribute_results(
                storage, key_values, [], tf.unstack, unstacked)
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

    def _make_inputs_and_labels_placeholders(self, pupil, num_unrollings, num_exercises, gpu_map):
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
        pupil_trainable_initializers = list()
        for ex_idx in range(num_exercises):
            tr, tr_pupil_format = pupil.create_trainable_variables_dictionary_for_optimizer(
                gpu_map[ex_idx], 'trainable_vars_ex_%s' % ex_idx)
            savers.append(pupil.create_saver(tr_pupil_format))
            pupil_trainable_initializers.append(
                tf.variables_initializer(
                    values_from_nested(tr), name='trainable_variables_initializer_for_ex_%s' % ex_idx
                )
            )
            trainable.append(tr)
            pupil_grad_eval_pupil_storage.append(pupil.create_storage(
                gpu_map[ex_idx], 'pupil_grad_eval_states_ex_%s' % ex_idx))
            optimizer_grad_pupil_storage.append(
                pupil.create_storage(gpu_map[ex_idx], 'optimizer_grad_states_ex_%s' % ex_idx))
        return trainable, pupil_grad_eval_pupil_storage, optimizer_grad_pupil_storage, savers,\
            pupil_trainable_initializers

    def _add_standard_train_hooks(self):
        self._hooks['pupil_grad_eval_inputs'] = self._pupil_grad_eval_inputs
        self._hooks['pupil_grad_eval_labels'] = self._pupil_grad_eval_labels
        self._hooks['optimizer_grad_inputs'] = self._optimizer_grad_inputs
        self._hooks['optimizer_grad_labels'] = self._optimizer_grad_labels
        self._hooks['pupil_savers'] = self._pupil_savers

    @staticmethod
    def _stop_gradients_in_opt_ins(optimizer_ins, inner_keys):
        with tf.name_scope('stop_gradients'):
            for k, v in optimizer_ins.items():
                with tf.name_scope(k):
                    for ik in inner_keys:
                        with tf.name_scope(ik):
                            # print("\n(Meta._stop_gradients_in_o_and_s)v['o']:", v['o'])
                            if isinstance(v[ik], (list, tuple)):
                                for idx, t in enumerate(v[ik]):
                                    v[ik][idx] = tf.stop_gradient(t)
                            else:
                                v[ik] = tf.stop_gradient(v[ik])
        return optimizer_ins

    def _eval_pupil_gradients_for_optimizer_training(
            self, pupil_grad_eval_inputs, pupil_grad_eval_labels,
            pupil_trainable_variables, pupil_grad_eval_pupil_storage):
        """optimizer_ins is nested dictionary. its first level keys are layer names and second level keys are names
        of tensors related to specific layer. E. g. 'o', 's', 'sigma', 'matrix', 'bias'. optimizer_outs have similar
        altough they also contain 'psi' and 'phi' tensors. In train mode first dim of tensors is exercise dim"""
        with tf.name_scope('eval_pupil_gradients_for_optimizer_training'):
            # print("(Meta._eval_pupil_gradients_for_optimizer_training)pupil_grad_eval_inputs:", pupil_grad_eval_inputs)
            loss, optimizer_ins, new_storage, predictions, labels = self._pupil.loss_and_opt_ins(
                pupil_grad_eval_inputs, pupil_grad_eval_labels,
                pupil_grad_eval_pupil_storage, opt_ins=pupil_trainable_variables)
            # print('(Meta._eval_pupil_gradients_for_optimizer_training)AFTER LOSS_AND_OPT_INS')
            # print_optimizer_ins(optimizer_ins)
            s_vectors, map_ = retrieve_from_inner_dicts(optimizer_ins, 's')
            # print('(Meta._eval_pupil_gradients_for_optimizer_training)s_vectors:', s_vectors)
            # print('(Meta._eval_pupil_gradients_for_optimizer_training)map_:', map_)
            sigma_vectors = tf.gradients(loss, s_vectors)
            optimizer_ins = distribute_into_inner_dicts(optimizer_ins, 'sigma', sigma_vectors, map_)

            optimizer_ins = self._stop_gradients_in_opt_ins(optimizer_ins, ['o', 's', 'sigma'])
            if self._normalizing is not None:
                optimizer_ins = self._normalize(optimizer_ins, self._normalizing)
            # print('(Meta._eval_pupil_gradients_for_optimizer_training)AFTER GRADIENT STOPPING')
            # print_optimizer_ins(optimizer_ins)

            # print('(Meta._eval_pupil_gradients_for_optimizer_training)AFTER SIGMA DISTRIBUTION')
            # print_optimizer_ins(optimizer_ins)
            with tf.name_scope('stop_gradients_in_pupil_storage_subsequent_pupil_launches'):
                new_storage = stop_gradient_in_nested(new_storage)
        return optimizer_ins, new_storage, loss, predictions, labels

    @staticmethod
    def _summarize_opt_flow(opt_flow, keys, summary_types, name_scope):
        res = list()
        with tf.name_scope(name_scope):
            for ok, ov in opt_flow.items():
                with tf.name_scope(ok):
                    for ik, iv in ov.items():
                        if ik in keys:
                            with tf.name_scope(ik):
                                sum_ops = variable_summaries(iv, summary_types, None)
                                with tf.device('/cpu:0'):
                                    res.append(tf.summary.merge(sum_ops))
        with tf.device('/cpu:0'):
            return tf.summary.merge(res)

    def _eval_pupil_gradients_for_optimizer_inference(self):
        loss, optimizer_ins, storage_save_ops, predictions, labels = self._pupil.loss_and_opt_ins_for_inference()
        # print('(Meta._eval_pupil_gradients_for_optimizer_inference)optimizer_ins:', optimizer_ins)
        s_vectors, map_ = retrieve_from_inner_dicts(optimizer_ins, 's')
        # print('(Meta._eval_pupil_gradients_for_optimizer_inference)map_:', map_)
        sigma_vectors = tf.gradients(loss, s_vectors)
        optimizer_ins = distribute_into_inner_dicts(optimizer_ins, 'sigma', sigma_vectors, map_)
        if self._normalizing is not None:
            optimizer_ins = self._normalize(optimizer_ins, self._normalizing)
        # print('(Meta._eval_pupil_gradients_for_optimizer_inference)optimizer_ins:', optimizer_ins)
        return optimizer_ins, storage_save_ops, loss, predictions, labels

    @staticmethod
    def _multiply_by_factor(opt_flow, factors):
        for ok, ov in opt_flow.items():
            with tf.name_scope(ok):
                for ik, iv in ov.items():
                    if ik in factors:
                        with tf.name_scope(ik):
                            ov[ik] = apply_to_nested(iv, lambda x: x*factors[ik])
        return opt_flow

    def _normalize(self, opt_flow, normalizing):
        with tf.name_scope('normalizing'):
            if normalizing['type'] == 'factor':
                opt_flow = self._multiply_by_factor(opt_flow, normalizing['factors'])
        return opt_flow

    def _denormalize(self, opt_flow, normalizing):
        with tf.name_scope('denormalizing'):
            if normalizing['type'] == 'factor':
                backward_factor_1 = 1.
                if 'sigma' in normalizing['factors']:
                    backward_factor_1 *= 1 / normalizing['factors']['sigma']
                if 'o' in normalizing['factors']:
                    backward_factor_2 = backward_factor_1 * 1 / normalizing['factors']['o']
                else:
                    backward_factor_2 = backward_factor_1
                opt_flow = self._multiply_by_factor(
                    opt_flow,
                    {'matrix_mods': backward_factor_2,
                     'bias_mods': backward_factor_1}
                )
        return opt_flow

    @staticmethod
    def _concat_opt_ins(opt_ins, inner_keys):
        num_concatenated = -1
        with tf.name_scope('concat_opt_ins'):
            for ov in opt_ins.values():
                for ik in inner_keys:
                    if num_concatenated < 0:
                        num_concatenated = len(ov[ik])
                    ov[ik + '_c'] = tf.concat(ov[ik], -2, name=ik + '_c')
        return opt_ins, num_concatenated

    @staticmethod
    def _split_opt_ins(opt_ins, inner_keys, num_splits):
        with tf.name_scope('split_opt_ins'):
            for ov in opt_ins.values():
                for ik in inner_keys:
                    ov[ik + '_spl'] = tf.split(ov[ik], num_splits, axis=-2)
        return opt_ins

    @staticmethod
    def _all_ins_2_1_vec(opt_ins, keys_2_unite):
        map_ = OrderedDict()
        vecs = list()
        with tf.name_scope('all_ins_2_1_vec'):
            for ok, ov in opt_ins.items():
                ok_map = OrderedDict()
                for ik in keys_2_unite:
                    if ik in ov:
                        if isinstance(ov[ik], list):
                            vecs.extend(ov[ik])
                            ok_map[ik] = [ov[ik][0].get_shape().as_list()[-1]] * len(ov[ik])
                        else:
                            vecs.append(ov[ik])
                            ok_map[ik] = ov[ik].get_shape().as_list()[-1]
                map_[ok] = ok_map
            return tf.concat(vecs, -1, name='all_ins_in_1_vec'), map_

    @staticmethod
    def _unpack_all_ins_from_1_vec(vec, map_):
        with tf.name_scope('unpack_all_ins_from_1_vec'):
            map_ = construct(map_)
            res = dict()
            flattened = flatten(map_)
            splitted = tf.split(vec, flattened, axis=-1)
            pointer = 0
            for ok, ok_map in map_.items():
                res[ok] = dict()
                for ik, split_dims in ok_map.items():
                    if isinstance(split_dims, list):
                        num_splits = len(split_dims)
                        res[ok][ik] = splitted[pointer:pointer+num_splits]
                        pointer += num_splits
                    else:
                        res[ok][ik] = splitted[pointer]
                        pointer += 1

            return res

    @staticmethod
    def _substitute_opt_ins(opt_ins, substitution_way):
        with tf.name_scope('opt_ins_substitution'):
            for ok, ov in opt_ins.items():
                with tf.name_scope(ok):
                    for ik, iv in ov.items():
                        if ik in ['s', 'o', 'sigma']:
                            with tf.name_scope(ik):
                                if isinstance(iv, list):
                                    ov[ik] = [
                                        get_substitution_tensor(
                                            t, substitution_way, minval=-.0001, maxval=.0001, value=.0001)
                                        for t in iv
                                    ]
                                else:
                                    ov[ik] = get_substitution_tensor(
                                        iv, substitution_way, minval=-.0001, maxval=.0001, value=.0001)
        return opt_ins

    @staticmethod
    def _mv_tensors(opt_ins, source_keys, dest_keys):
        for ov in opt_ins.values():
            for s_key, d_key in zip(source_keys, dest_keys):
                ov[d_key] = ov[s_key]
        return opt_ins

    @staticmethod
    def _compose_phi_and_psi(optimizer_outs):
        with tf.name_scope('phi_and_psi'):
            for k, v in optimizer_outs.items():
                with tf.name_scope(k):
                    with tf.name_scope('phi'):
                        if isinstance(v['o_pr'], list):
                            v['phi'] = tf.concat(v['o_pr'], axis=-2, name='phi')
                        else:
                            v['phi'] = v['o_pr']
                        # print('(Meta._compose_phi_and_psi)v["o_pr"]:', v['o_pr'])
                        # if isinstance(v['o'], list):
                        #     v['phi'] = tf.add_n(v['o']) / len(v['o']) * learning_rate**.5
                        # else:
                        #     v['phi'] = v['o'] * learning_rate**.5
                    with tf.name_scope('psi'):
                        # v['psi'] = v['sigma']
                        if isinstance(v['sigma_pr'], list):
                            v['psi'] = tf.concat(v['sigma_pr'], axis=-2, name='psi')
                        else:
                            v['psi'] = v['sigma_pr']
                        # print('(Meta._compose_phi_and_psi)v["sigma_pr"]:', v['sigma_pr'])
                        # if isinstance(v['sigma'], list):
                        #     v['psi'] = tf.add_n(v['sigma']) / len(v['sigma']) * learning_rate**.5
                        # else:
                        #     v['psi'] = v['sigma'] * learning_rate**.5
        return optimizer_outs

    @staticmethod
    def _get_optimizer_ins_ndims(opt_ins):
        ov = list(opt_ins.values())[0]
        s = ov['s']
        if isinstance(s, list):
            return len(s[0].get_shape().as_list())
        else:
            return len(s.get_shape().as_list())

    @staticmethod
    def _expand_exercise_dim(opt_ins, inner_keys):
        for ov in opt_ins.values():
            for ik in inner_keys:
                iv = ov[ik]
                if isinstance(iv, list):
                    for idx, tensor in enumerate(iv):
                        iv[idx] = tf.expand_dims(tensor, axis=0)
                else:
                    ov[ik] = tf.expand_dims(iv, axis=0)
        return opt_ins

    @staticmethod
    def _collapse_exercise_dim(opt_ins, inner_keys):
        for ov in opt_ins.values():
            for ik in inner_keys:
                iv = ov[ik]
                if isinstance(iv, list):
                    for idx, tensor in enumerate(iv):
                        print("(Meta._collapse_exercise_dim)tensor:", tensor)
                        tensor_shape = tensor.get_shape().as_list()
                        iv[idx] = tf.reshape(tensor, shape=tensor_shape[1:])
                else:
                    tensor_shape = iv.get_shape().as_list()
                    ov[ik] = tf.reshape(iv, shape=tensor_shape[1:])
        return opt_ins

    @staticmethod
    def _forward_permute(optimizer_ins, in_perm_keys, out_perm_keys, collapse_1st_dim=False):
        with tf.name_scope('forward_permute'):
            for k, v in optimizer_ins.items():
                with tf.name_scope(k):
                    if 'in_perm' in v:
                        for in_perm_key in in_perm_keys:
                            with tf.name_scope(in_perm_key):
                                if isinstance(v[in_perm_key], list):
                                    v[in_perm_key] = [custom_matmul(t, v['in_perm']) for t in v[in_perm_key]]
                                    if collapse_1st_dim:
                                        v['o'] = [tf.reshape(vec, vec.get_shape().as_list()[1:]) for vec in v['o']]
                                else:
                                    v[in_perm_key] = custom_matmul(v[in_perm_key], v['in_perm'])
                                    if collapse_1st_dim:
                                        v['o'] = tf.reshape(v['o'], v['o'].get_shape().as_list()[1:])
                    if 'out_perm' in v:
                        for out_perm_key in out_perm_keys:
                            with tf.name_scope(out_perm_key):
                                if isinstance(v[out_perm_key], list):
                                    v[out_perm_key] = [custom_matmul(t, v['out_perm']) for t in v[out_perm_key]]
                                    if collapse_1st_dim:
                                        v['sigma'] = [
                                            tf.reshape(vec, vec.get_shape().as_list()[1:]) for vec in v['sigma']]
                                else:
                                    v[out_perm_key] = custom_matmul(v[out_perm_key], v['out_perm'])
                                    if collapse_1st_dim:
                                        v['sigma'] = tf.reshape(v['sigma'], v['sigma'].get_shape().as_list()[1:])
        return optimizer_ins

    @staticmethod
    def _backward_permute(optimizer_outs, in_perm_keys, out_perm_keys, collapse_1st_dim=False):
        with tf.name_scope('backward_permute'):
            for k, v in optimizer_outs.items():
                with tf.name_scope(k):
                    if 'in_perm' in v:
                        in_tr = tf.matrix_transpose(v['in_perm'])
                        for in_perm_key in in_perm_keys:
                            with tf.name_scope(in_perm_key):
                                if isinstance(v[in_perm_key], list):
                                    v[in_perm_key] = [custom_matmul(t, in_tr) for t in v[in_perm_key]]
                                    if collapse_1st_dim:
                                        v[in_perm_key] = [
                                            tf.reshape(
                                                vec, vec.get_shape().as_list()[1:]) for vec in v[in_perm_key]]
                                else:
                                    v[in_perm_key] = custom_matmul(v[in_perm_key], in_tr)
                                    if collapse_1st_dim:
                                        v[in_perm_key] = tf.reshape(
                                            v[in_perm_key], v[in_perm_key].get_shape().as_list()[1:])
                    if 'out_perm' in v:
                        out_tr = tf.matrix_transpose(v['out_perm'])
                        for out_perm_key in out_perm_keys:
                            with tf.name_scope(out_perm_key):
                                if isinstance(v[out_perm_key], list):
                                    v[out_perm_key] = [custom_matmul(t, out_tr) for t in v[out_perm_key]]
                                    if collapse_1st_dim:
                                        v[out_perm_key] = [
                                            tf.reshape(vec, vec.get_shape().as_list()[1:]) for vec in v[out_perm_key]]
                                else:
                                    v[out_perm_key] = custom_matmul(v[out_perm_key], out_tr)
                                    if collapse_1st_dim:
                                        v[out_perm_key] = tf.reshape(
                                            v[out_perm_key], v[out_perm_key].get_shape().as_list()[1:])
        return optimizer_outs

    @staticmethod
    def _empty_core(optimizer_ins):
        # print('(Meta._empty_core)optimizer_ins:')
        # print_optimizer_ins(optimizer_ins)
        with tf.name_scope('core'):
            for k, v in optimizer_ins.items():
                v['o_pr'] = v['o']
                v['sigma_pr'] = v['sigma']
        return optimizer_ins, []

    def _compose_mods(self, optimizer_outs, learning_rate=None):
        mods = list()
        with tf.name_scope('pupil_mods'):
            for k, v in optimizer_outs.items():
                with tf.name_scope(k):
                    if 'matrix' in v:
                        ndims = len(v['phi'].get_shape().as_list())
                        # print("\n(Meta._compose_mods)v['phi'].shape:", v['phi'].get_shape().as_list())
                        # print("\n(Meta._compose_mods)v['psi'].shape:", v['psi'].get_shape().as_list())
                        with tf.name_scope('matrix'):
                            with tf.name_scope('matrix_mods_computation'):
                                if ndims == 3:
                                    eq = 'ijk,ijl->ikl'
                                elif ndims == 2:
                                    eq = 'jk,jl->kl'
                                v['matrix_mods'] = tf.einsum(eq, v['phi'], v['psi'])  # / batch_size
                            # with tf.device('/cpu:0'):
                            #     v['matrix_mods'] = tf.Print(
                            #         v['matrix_mods'],
                            #         [v['matrix_mods']],
                            #         message='\n' + k + '\nmatrix:\n')
                            # with tf.device('/cpu:0'):
                            #     v['matrix_mods'] = tf.Print(
                            #         v['matrix_mods'], [v['matrix_mods']], message="(%s)v['matrix_mods']: " % k,
                            #         summarize=10
                            #     )

                            with tf.name_scope('basic_matrix_mods_computation'):
                                o = tf.concat(v['o'], -2, name='o')
                                sigma = tf.concat(v['sigma'], -2, name='sigma')
                                o_shape = o.get_shape().as_list()
                                sigma_shape = sigma.get_shape().as_list()
                                o_ndims = len(o_shape)

                                if ndims > o_ndims:
                                    o = tf.reshape(o, [1] + o_shape)
                                    sigma = tf.reshape(sigma, [1] + sigma_shape)
                                elif ndims < o_ndims:
                                    o = tf.reshape(o, [-1] + o_shape[2:])
                                    sigma = tf.reshape(sigma, [-1] + sigma_shape[2:])
                                basic_mods = tf.einsum(eq, o, sigma)
                                
                            with tf.name_scope('relative_difference_computation'):
                                rel_diff = (v['matrix_mods'] - basic_mods) / \
                                           (tf.abs(basic_mods) + 1e-7) * \
                                           (tf.to_float(tf.greater(basic_mods, 0.)) - .5) * 2.

                            # v['matrix_mods'] = basic_mods

                            # v['matrix_mods'] = tf.zeros(basic_mods.get_shape().as_list())

                            # with tf.device('/cpu:0'):
                            #     if k == 'lstm_layer_0':
                            #         v['matrix_mods'] = tf.Print(
                            #             v['matrix_mods'], [rel_diff],
                            #             "(relative difference)%s = " % k, summarize=20)
                            #         v['matrix_mods'] = tf.Print(
                            #             v['matrix_mods'], [basic_mods],
                            #             "(basic_mods)%s = " % k, summarize=20)
                            #     v['matrix_mods'] = tf.Print(
                            #         v['matrix_mods'], [v['matrix_mods']],
                            #         "(v['matrix_mods'])%s = " % k, summarize=20)
                            mods.append(v['matrix_mods'])

                    if 'bias' in v:
                        with tf.name_scope('bias'):
                            # with tf.name_scope('basic_bias_mods_computation'):
                            #     sigma = tf.concat(v['sigma'], -2, name='sigma')
                            #     if ndims == 2:
                            #         sigma = tf.reshape(sigma, sigma.get_shape().as_list()[1:])
                            #     v['bias_mods'] = tf.reduce_sum(sigma, axis=-2, name='basic_bias_mods')

                            v['bias_mods'] = tf.reduce_sum(v['psi'], axis=-2)

                            # v['bias_mods'] = tf.zeros(v['bias_mods'].get_shape().as_list())
                            mods.append(v['bias_mods'])
                            # with tf.device('/cpu:0'):
                            #     v['bias_mods'] = tf.Print(
                            #         v['bias_mods'], [v['bias_mods']], message='\n' + k + '\nbias:\n')
                if self._normalizing is not None:
                    optimizer_outs = self._denormalize(
                        optimizer_outs,
                        self._normalizing
                    )
            if self._inp_gradient_clipping is not None:
                with tf.name_scope('clip_gradients'):
                    glob_norm = global_norm(mods)
                    factor = tf.divide(
                        self._clip_norm,
                        tf.maximum(glob_norm, self._clip_norm),
                        name='factor')
                    # norm = tf.stop_gradient(tf.maximum(global_norm(mods), 1.))

                    # with tf.device('/cpu:0'):
                    #     factor = tf.Print(factor, [factor], message='(Meta._compose_mods)factor: ')

                    for k, v in optimizer_outs.items():
                        with tf.name_scope(k):
                            if 'matrix_mods' in v:
                                with tf.name_scope('matrix'):
                                    v['matrix_mods'] = tf.multiply(v['matrix_mods'], factor, name='matrix_mods')
                            if 'bias_mods' in v:
                                with tf.name_scope('bias'):
                                    v['bias_mods'] = tf.multiply(v['bias_mods'], factor, name='bias_mods')

                    glob_loss = global_l2_loss(mods)

                    # with tf.device('/cpu:0'):
                    #     glob_loss = tf.Print(glob_loss, [glob_norm], message="(Meta._compose_mods)glob_norm: ")

                    self._additional_loss += glob_loss / self._clip_norm
        return optimizer_outs

    @staticmethod
    def _sub_mods(mods):
        with tf.name_scope('subtract_modifications'):
            for k, v in mods.items():
                with tf.name_scope(k):
                    if 'matrix' in v:
                        v['matrix'] = tf.subtract(v['matrix'], v['matrix_mods'], name='matrix')
                    if 'bias' in v:
                        v['bias'] = tf.subtract(v['bias'], v['bias_mods'], name='bias')
            return mods

    def _compute_optimizer_gradients(self, loss, name_scope='compute_optimizer_gradients'):
        with tf.name_scope(name_scope):
            loss = tf.add(loss, self._additional_loss, name='loss_with_additional_loss')
            optimizer_trainable_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='optimizer_trainable_variables')
            return self._optimizer_for_optimizer_training.compute_gradients(
                loss, var_list=optimizer_trainable_variables)

    @staticmethod
    def _l2_loss(reg_rate):
        with tf.name_scope('l2_loss'):
            weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
            regularizer = tf.contrib.layers.l2_regularizer(reg_rate)
            loss = 0
            for w in weights:
                loss += regularizer(w)
        return loss

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
            start_additional_metrics_by_gpu = dict()
            end_additional_metrics_by_gpu = dict()
            for add_metric in self._additional_metrics:
                start_additional_metrics_by_gpu[add_metric] = list()
                end_additional_metrics_by_gpu[add_metric] = list()
            summaries = list()

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
                        one_gpu_end_losses = list()
                        one_gpu_start_losses = list()
                        one_gpu_start_additional_metrics = dict()
                        one_gpu_end_additional_metrics = dict()
                        for add_metric in self._additional_metrics:
                            one_gpu_start_additional_metrics[add_metric] = list()
                            one_gpu_end_additional_metrics[add_metric] = list()
                        # print("(Meta._train_graph)pupil_grad_eval_inputs:", pupil_grad_eval_inputs)
                        for unr_idx in range(self._num_optimizer_unrollings):
                            with tf.name_scope('optimizer_unrolling_%s' % unr_idx):
                                optimizer_ins, pupil_grad_eval_pupil_storage[gpu_idx], start_loss, start_predictions,\
                                    start_labels = \
                                        self._eval_pupil_gradients_for_optimizer_training(
                                            pupil_grad_eval_inputs[gpu_idx][unr_idx],
                                            pupil_grad_eval_labels[gpu_idx][unr_idx],
                                            pupil_trainable_variables[gpu_idx], pupil_grad_eval_pupil_storage[gpu_idx]
                                        )
                                start_additional_metrics = compute_metrics(
                                    self._additional_metrics, start_predictions,
                                    start_labels, start_loss, keep_first_dim=True)

                                # print("(Meta._train_graph)pupil_trainable_variables[%s]:" % gpu_idx,
                                #       pupil_trainable_variables[gpu_idx])

                                # print("(Meta._train_graph)BEFORE OPTIMIZER CORE:")
                                # print_optimizer_ins(optimizer_ins)
                                if 'summarize_opt_ins' in self._flags:
                                    summaries.append(
                                        self._summarize_opt_flow(
                                            optimizer_ins,
                                            ['o', 's', 'sigma'],
                                            ['mean', 'stddev', 'min', 'max', 'histogram'],
                                            'opt_ins'
                                        )
                                    )
                                if 'opt_ins_substitution' in self._flags:
                                    optimizer_ins = self._substitute_opt_ins(optimizer_ins, 'constant')
                                optimizer_outs, tmp_states = self._optimizer_core(
                                    optimizer_ins, tmp_states, gpu_idx, permute=self._permute)
                                optimizer_outs = self._compose_phi_and_psi(optimizer_outs)
                                optimizer_outs_with_mods = self._compose_mods(optimizer_outs)
                                optimizer_outs_mods_are_applied = self._sub_mods(optimizer_outs_with_mods)
                                new_pupil_trainable = self._filter_opt_flow_dict(
                                    optimizer_outs_mods_are_applied, ['matrix', 'bias'])

                                end_loss, _, optimizer_grad_pupil_storage[gpu_idx], end_predictions, end_labels = \
                                    self._pupil.loss_and_opt_ins(
                                        optimizer_grad_inputs[gpu_idx][unr_idx],
                                        optimizer_grad_labels[gpu_idx][unr_idx],
                                        optimizer_grad_pupil_storage[gpu_idx], opt_ins=new_pupil_trainable,
                                        name_scope='pupil_loss_after_pupil_modification'
                                    )
                                end_additional_metrics = compute_metrics(
                                    self._additional_metrics, end_predictions,
                                    end_labels, end_loss, keep_first_dim=True)
                                # print("(Meta._train_graph)new_pupil_trainable:",
                                #       new_pupil_trainable)

                                pupil_trainable_variables[gpu_idx] = new_pupil_trainable
                                one_gpu_start_losses.append(start_loss)
                                one_gpu_end_losses.append(end_loss)
                                one_gpu_start_additional_metrics = append_to_nested(
                                    one_gpu_start_additional_metrics,
                                    start_additional_metrics
                                )
                                one_gpu_end_additional_metrics = append_to_nested(
                                    one_gpu_end_additional_metrics,
                                    end_additional_metrics
                                )
                        one_gpu_start_losses = tf.stack(one_gpu_start_losses)
                        one_gpu_end_losses = tf.stack(one_gpu_end_losses)

                        # with tf.device('/cpu:0'):
                        #     one_gpu_start_losses = tf.Print(
                        #         one_gpu_start_losses,
                        #         [tf.reduce_mean(one_gpu_start_losses, axis=-1)],
                        #         message="(Meta._train_graph)start losses by unrollings on gpu %s: " % gpu_idx,
                        #         summarize=20)
                        #     one_gpu_end_losses = tf.Print(
                        #         one_gpu_end_losses,
                        #         [tf.reduce_mean(one_gpu_end_losses, axis=-1)],
                        #         message="(Meta._train_graph)end losses by unrollings on gpu %s: " % gpu_idx,
                        #         summarize=20)

                        one_gpu_end_loss = tf.reduce_mean(one_gpu_end_losses)
                        train_optimizer_loss = one_gpu_end_loss + self._l2_loss(self._regularization_rate)
                        one_gpu_start_loss = tf.reduce_mean(one_gpu_start_losses)

                        one_gpu_start_additional_metrics = func_on_list_in_nested(
                            one_gpu_start_additional_metrics, tf.reduce_mean)
                        one_gpu_end_additional_metrics = func_on_list_in_nested(
                            one_gpu_end_additional_metrics, tf.reduce_mean)

                        new_pupil_trainable = self._retrieve_and_unstack_trainable_variables(
                            self._num_exercises, new_pupil_trainable)
                        # print("(Meta._train_graph)pupil_grad_eval_pupil_storage (before unstacking):",
                        #       pupil_grad_eval_pupil_storage)
                        pupil_grad_eval_pupil_storage[gpu_idx] = self._unstack_storage(
                            self._num_exercises, pupil_grad_eval_pupil_storage[gpu_idx],
                            name_scope='unstack_pupil_grad_eval_pupil_storage')
                        # print("(Meta._train_graph)pupil_grad_eval_pupil_storage (after unstacking):",
                        #       pupil_grad_eval_pupil_storage)
                        optimizer_grad_pupil_storage[gpu_idx] = self._unstack_storage(
                            self._num_exercises, optimizer_grad_pupil_storage[gpu_idx],
                            name_scope='unstack_optimizer_grad_pupil_storage'
                        )
                        # print("(Meta._train_graph)self._gpu_borders:", self._gpu_borders)
                        # print("(Meta._train_graph)len(self._pupil_trainable_variables):",
                        #       len(self._pupil_trainable_variables))
                        # print("(Meta._train_graph)len(self._pupil_grad_eval_pupil_storage):",
                        #       len(self._pupil_grad_eval_pupil_storage))
                        # print("(Meta._train_graph)len(self._optimizer_grad_pupil_storage):",
                        #       len(self._optimizer_grad_pupil_storage))

                        # print("(Meta._train_graph)new_pupil_trainable:", new_pupil_trainable)
                        # print("(Meta._train_graph)pupil_grad_eval_pupil_storage:", pupil_grad_eval_pupil_storage)
                        # print("(Meta._train_graph)self._optimizer_grad_pupil_storage"
                        #       "[self._gpu_borders[gpu_idx][0]:self._gpu_borders[gpu_idx][1]]:",
                        #       self._optimizer_grad_pupil_storage
                        #           [self._gpu_borders[gpu_idx][0]:self._gpu_borders[gpu_idx][1]])
                        # print("(Meta._train_graph)optimizer_grad_pupil_storage[gpu_idx]:",
                        #       optimizer_grad_pupil_storage[gpu_idx])
                        save_ops = compose_save_list(
                            (optimizer_states, tmp_states),
                            name_scope='save_opt_states'
                        ) + compose_save_list(
                            (self._pupil_trainable_variables[
                                 self._gpu_borders[gpu_idx][0]:self._gpu_borders[gpu_idx][1]],
                             new_pupil_trainable),
                            name_scope='save_pupil_train_vars'
                        ) + compose_save_list(
                            (self._pupil_grad_eval_pupil_storage[
                                 self._gpu_borders[gpu_idx][0]:self._gpu_borders[gpu_idx][1]],
                             pupil_grad_eval_pupil_storage[gpu_idx]),
                            name_scope='save_pupil_grad_eval_pupil_storage'
                        ) + compose_save_list(
                            (self._optimizer_grad_pupil_storage[
                                 self._gpu_borders[gpu_idx][0]:self._gpu_borders[gpu_idx][1]],
                             optimizer_grad_pupil_storage[gpu_idx]),
                            name_scope='save_optimizer_grad_pupil_storage'
                        )

                        with tf.control_dependencies(save_ops):
                            train_optimizer_loss = tf.identity(
                                train_optimizer_loss, name='train_optimizer_loss_on_gpu_%s' % gpu_idx)
                            start_losses_by_gpu.append(one_gpu_start_loss)
                            end_losses_by_gpu.append(one_gpu_end_loss)
                            grads_and_vars = self._compute_optimizer_gradients(
                                train_optimizer_loss, name_scope='compute_opt_grads_on_gpu_%s' % gpu_idx)

                            start_additional_metrics_by_gpu = append_to_nested(
                                start_additional_metrics_by_gpu,
                                one_gpu_start_additional_metrics
                            )
                            end_additional_metrics_by_gpu = append_to_nested(
                                end_additional_metrics_by_gpu,
                                one_gpu_end_additional_metrics
                            )

                            # new_grads_and_vars = list()
                            # with tf.device('/cpu:0'):
                            #     for gv in grads_and_vars:
                            #         new_grads_and_vars.append(
                            #             (tf.Print(
                            #                 gv[0], [l2_loss_per_elem(gv[0])],
                            #                 message="gpu %s, gradients by %s = " % (gpu_idx, gv[1].name),
                            #                 summarize=10
                            #             ),
                            #              gv[1])
                            #         )
                            # debug_grads = tf.gradients(one_gpu_end_loss, self._debug_tensors)
                            # # print("(Meta._train_graph)self._debug_tensors:", self._debug_tensors)
                            # # print("(Meta._train_graph)debug_grads:", debug_grads)
                            # with tf.device('/cpu:0'):
                            #     for idx, mres in enumerate(debug_grads):
                            #         new_grads_and_vars[0] = (
                            #             tf.Print(
                            #                 new_grads_and_vars[0][0], [l2_loss_per_elem(mres)],
                            #                 message="%s res layer in core matmul res grad: " % idx,
                            #                 summarize=10
                            #             ),
                            #             new_grads_and_vars[0][1]
                            #         )
                            # grads_and_vars = new_grads_and_vars

                            grads_and_vars = filter_none_gradients(grads_and_vars)
                            tower_grads.append(grads_and_vars)

            with tf.device(self._base_device):
                with tf.name_scope('unite_exercise_gradients'):
                    grads_and_vars = average_gradients(tower_grads)
                    grads, v = self._tune_gradients(grads_and_vars)

                    # opt_vars_l2_norm = apply_to_nested(self._opt_trainable, l2_loss_per_elem)
                    # grads[0] = tf_print_nested(opt_vars_l2_norm, 'opt_vars_l2_norm', grads[0], 10)

                    train_op = self._optimizer_for_optimizer_training.apply_gradients(zip(grads, v))

                    self._hooks['optimizer_train_op'] = train_op

                    all_start_losses = tf.concat(start_losses_by_gpu, 0)
                    all_end_losses = tf.concat(end_losses_by_gpu, 0)

                    # with tf.device('/cpu:0'):
                    #     all_start_losses = tf.Print(
                    #         all_start_losses,
                    #         [all_start_losses],
                    #         message='all start losses by gpu: ',
                    #         summarize=20
                    #     )
                    #     all_end_losses = tf.Print(
                    #         all_end_losses,
                    #         [all_end_losses],
                    #         message='all end losses by gpu: ',
                    #         summarize=20
                    #     )

                    # with tf.device('/cpu:0'):
                    #     all_start_losses = tf.Print(
                    #         all_start_losses, [all_start_losses], message="all_start_losses: ")
                    self._hooks['start_loss'] = tf.reduce_mean(all_start_losses)
                    self._hooks['end_loss'] = tf.reduce_mean(all_end_losses)
                    start_additional_metrics = func_on_list_in_nested(start_additional_metrics_by_gpu, tf.reduce_mean)
                    end_additional_metrics = func_on_list_in_nested(end_additional_metrics_by_gpu, tf.reduce_mean)
                    for add_metric in self._additional_metrics:
                        self._hooks['start_' + add_metric] = start_additional_metrics[add_metric]
                        self._hooks['end_' + add_metric] = end_additional_metrics[add_metric]
                    if self._hooks['train_optimizer_summary'] is None:
                        self._hooks['train_optimizer_summary'] = summaries
                    elif len(summaries) > 0:
                        self._hooks['train_optimizer_summary'].extend(summaries)

    def _inference_graph(self):
        with tf.name_scope('optimizer_inference_graph'):
            with tf.device('/gpu:0'):
                optimizer_states = self._create_optimizer_states(1, 'inference_optimizer_states', 0)
                optimizer_ins, pupil_save_ops, start_loss, predictions, labels = \
                    self._eval_pupil_gradients_for_optimizer_inference()

                additional_metrics = compute_metrics(
                    self._additional_metrics, predictions,
                    labels, start_loss, keep_first_dim=False)

                # optimizer_ins = self._extend_with_permutations(optimizer_ins, 0)
                # print('\n(Meta._inference_graph)optimizer_ins before permutations:')
                # print_optimizer_ins(optimizer_ins)
                # optimizer_ins = self._forward_permute(optimizer_ins, ['o'], ['sigma'], collapse_1st_dim=True)
                # print('\n(Meta._inference_graph)optimizer_ins after permutations:')
                # print_optimizer_ins(optimizer_ins)
                # opt = tf.train.GradientDescentOptimizer(1.)
                # grads, vars = zip(*opt.compute_gradients(start_loss))
                optimizer_ins = self._expand_exercise_dim(optimizer_ins, ['o', 'sigma'])
                optimizer_outs, new_optimizer_states = self._optimizer_core(
                    optimizer_ins, optimizer_states, 0, permute=False)
                optimizer_outs = self._collapse_exercise_dim(optimizer_outs, ['o_pr', 'sigma_pr'])
                # print('\n(Meta._inference_graph)optimizer_outs:')
                # print_optimizer_ins(optimizer_outs)
                # optimizer_outs = self._backward_permute(optimizer_outs, ['o_pr'], ['sigma_pr'], collapse_1st_dim=True)
                optimizer_outs = self._compose_phi_and_psi(optimizer_outs)
                # for var, gr in zip(vars, grads):
                #     with tf.device('/cpu:0'):
                #         optimizer_outs['lstm_layer_0']['psi'] = tf.Print(
                #             optimizer_outs['lstm_layer_0']['psi'], [gr], message='\n' + var.name + ':\n')
                # print('\n(Meta._inference_graph)optimizer_states:', optimizer_states)
                # print('\n(Meta._inference_graph)new_optimizer_states:', new_optimizer_states)
                mods = self._compose_mods(optimizer_outs, learning_rate=LEARNING_RATE_FOR_EMPTY_CORE)
                # print('\n(Meta._inference_graph)')
                # print_optimizer_ins(mods)
                mods = self._sub_mods(mods)
                optimizer_save_states_ops = compose_save_list(
                    (optimizer_states, new_optimizer_states), name_scope='save_optimizer_states')
                # print('\n(Meta._inference_graph)pupil_save_ops:', pupil_save_ops)
                # print('\n(Meta._inference_graph)new_optimizer_states:', new_optimizer_states)
                with tf.control_dependencies(pupil_save_ops+optimizer_save_states_ops):
                    train_op = tf.group(*self._pupil.apply_mods(mods), name='train_with_meta_optimizer_op')
                self._hooks['train_with_meta_optimizer_op'] = train_op
                self._hooks['loss'] = start_loss

                for add_metric in self._additional_metrics:
                    self._hooks[add_metric] = additional_metrics[add_metric]
