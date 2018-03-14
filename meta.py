import tensorflow as tf
from some_useful_functions import construct


class Meta(object):

    @staticmethod
    def _stack_different_exercises_variables(variables):
        """stack variables from different checkpoints or permutations.
         Stacking is performed along dimension which is last to tensor inner dimensions
         Args:
             variables - dictionary with optimizee variables.
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
    def _make_inputs_and_labels_placeholders(optimizee, num_unrollings, num_exercises, gpu_map, regime):
        """If both num_unrollings and num_exercises are not None outputs are lists of lists where
        inner list is for unrollings and outer is for exercises. If only one of variables num_unrollings and
        num_exercises is not None outputs are lists of placeholders. And finally if both num_unrollings and
        num_exercises are None  outputs are placeholders."""
        if regime == 'train':
            grad_eval_inputs = list()
            grad_eval_labels = list()

            optimizer_learn_inputs = list()
            optimizer_learn_labels = list()

            for ex_idx in range(num_exercises):
                if num_unrollings is not None:
                    grad_eval_inputs.append(list())
                    grad_eval_labels.append(list())
                    optimizer_learn_inputs.append(list())
                    optimizer_learn_labels.append(list())
                with tf.name_scope('exercise_%s' % ex_idx):
                    with tf.name_scope('grad_eval_placeholders'):
                        if num_unrollings is not None:
                            for i in range(num_unrollings):
                                placeholders = optimizee.make_inputs_and_labels_placeholders(
                                    '/gpu:%s' % gpu_map[ex_idx], 'unrolling_%s' % i)
                                grad_eval_inputs[ex_idx].append(placeholders['inputs'])
                                grad_eval_labels[ex_idx].append(placeholders['labels'])
                        else:
                            placeholders = optimizee.make_inputs_and_labels_placeholders(
                                '/gpu:%s' % gpu_map[ex_idx], None)
                            grad_eval_inputs.append(placeholders['inputs'])
                            grad_eval_labels.append(placeholders['labels'])
                    with tf.name_scope('optimizer_learn_placeholders'):
                        if num_unrollings is not None:
                            for i in range(num_unrollings):
                                placeholders = optimizee.make_inputs_and_labels_placeholders(
                                    '/gpu:%s' % gpu_map[ex_idx], 'unrolling_%s' % i)
                                optimizer_learn_inputs[ex_idx].append(placeholders['inputs'])
                                optimizer_learn_labels[ex_idx].append(placeholders['labels'])
                        else:
                            placeholders = optimizee.make_inputs_and_labels_placeholders(
                                '/gpu:%s' % gpu_map[ex_idx], None)
                            optimizer_learn_inputs.append(placeholders['inputs'])
                            optimizer_learn_inputs.append(placeholders['labels'])
            return grad_eval_inputs, grad_eval_labels, optimizer_learn_inputs, optimizer_learn_labels
        elif regime == 'inference':
            with tf.name_scope('optimizer_inference_placeholders'):
                if num_unrollings is not None:
                    grad_eval_inputs = list()
                    grad_eval_labels = list()
                    for i in range(num_unrollings):
                        placeholders = optimizee.make_inputs_and_labels_placeholders(
                            '/gpu:0', 'unrolling_%s' % i)
                        grad_eval_inputs.append(placeholders['inputs'])
                        grad_eval_labels.append(placeholders['labels'])
                else:
                    placeholders = optimizee.make_inputs_and_labels_placeholders(
                        '/gpu:0', None)
                    grad_eval_inputs = placeholders['inputs']
                    grad_eval_labels = placeholders['labels']
            return grad_eval_inputs, grad_eval_labels

    @staticmethod
    def _create_optimizee_variables_and_savers(optimizee, num_exercises, gpu_map):
        trainable = list()
        grad_eval_storage = list()
        optimizer_learn_storage = list()
        savers = list()
        for ex_idx in range(num_exercises):
            tr = optimizee.create_trainable_variables_dictionary(
                gpu_map[ex_idx], 'trainable_vars_ex_%s' % ex_idx)
            savers.append(optimizee.create_saver(tr))
            trainable.append(tr)
            grad_eval_storage.append(optimizee.create_storage(gpu_map[ex_idx], 'grad_eval_states_ex_%s' % ex_idx))
            optimizer_learn_storage.append(
                optimizee.create_storage(gpu_map[ex_idx], 'optimizer_learn_states_ex_%s' % ex_idx))
        return trainable, grad_eval_storage, optimizer_learn_storage, savers