import tensorflow as tf
from some_useful_functions import construct


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
    def _make_inputs_and_labels_placeholders(_pupil, num_unrollings, num_exercises, gpu_map):
        """If both num_unrollings and num_exercises are not None outputs are lists of lists where
        inner list is for unrollings and outer is for exercises. If only one of variables num_unrollings and
        num_exercises is not None outputs are lists of placeholders. And finally if both num_unrollings and
        num_exercises are None  outputs are placeholders."""
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
                            placeholders = _pupil.make_inputs_and_labels_placeholders(
                                '/gpu:%s' % gpu_map[ex_idx], 'unrolling_%s' % i)
                            pupil_grad_eval_inputs[ex_idx].append(placeholders['inputs'])
                            pupil_grad_eval_labels[ex_idx].append(placeholders['labels'])
                    else:
                        placeholders = _pupil.make_inputs_and_labels_placeholders(
                            '/gpu:%s' % gpu_map[ex_idx], None)
                        pupil_grad_eval_inputs.append(placeholders['inputs'])
                        pupil_grad_eval_labels.append(placeholders['labels'])
                with tf.name_scope('optimizer_grad_placeholders'):
                    if num_unrollings is not None:
                        for i in range(num_unrollings):
                            placeholders = _pupil.make_inputs_and_labels_placeholders(
                                '/gpu:%s' % gpu_map[ex_idx], 'unrolling_%s' % i)
                            optimizer_grad_inputs[ex_idx].append(placeholders['inputs'])
                            optimizer_grad_labels[ex_idx].append(placeholders['labels'])
                    else:
                        placeholders = _pupil.make_inputs_and_labels_placeholders(
                            '/gpu:%s' % gpu_map[ex_idx], None)
                        optimizer_grad_inputs.append(placeholders['inputs'])
                        optimizer_grad_inputs.append(placeholders['labels'])
        return pupil_grad_eval_inputs, pupil_grad_eval_labels, optimizer_grad_inputs, optimizer_grad_labels

    @staticmethod
    def _create__pupil_variables_and_savers(_pupil, num_exercises, gpu_map):
        trainable = list()
        pupil_grad_eval_storage = list()
        optimizer_grad_pupil_storage = list()
        savers = list()
        for ex_idx in range(num_exercises):
            tr = _pupil.create_trainable_variables_dictionary(
                gpu_map[ex_idx], 'trainable_vars_ex_%s' % ex_idx)
            savers.append(_pupil.create_saver(tr))
            trainable.append(tr)
            pupil_grad_eval_storage.append(_pupil.create_storage(
                gpu_map[ex_idx], 'pupil_grad_eval_states_ex_%s' % ex_idx))
            optimizer_grad_pupil_storage.append(
                _pupil.create_storage(gpu_map[ex_idx], 'optimizer_grad_states_ex_%s' % ex_idx))
        return trainable, pupil_grad_eval_storage, optimizer_grad_pupil_storage, savers