import tensorflow as tf
from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import construct_dict_without_none_entries


class Empty(Meta):
    @staticmethod
    def form_kwargs(kwargs_for_building, insertions):
        for insertion in insertions:
            if insertion['list_index'] is None:
                kwargs_for_building[insertion['hp_name']] = insertion['paste']
            else:
                kwargs_for_building[insertion['hp_name']][insertion['list_index']] = insertion['paste']
        return kwargs_for_building

    @staticmethod
    def check_kwargs(**kwargs):
        pass

    def _create_optimizer_states(self, num_exercises, var_scope, gpu_idx):
        return list()

    def _optimizer_core(self, optimizer_ins, states, gpu_idx, permute=False):
        # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
        # optimizer_ins = self._forward_permute(optimizer_ins)
        for ok, ov in optimizer_ins.items():
            with tf.name_scope(ok):
                for ik, iv in ov.items():
                    print(
                        "(Empty._optimizer_core)optimizer_ins[%s][%s].shape:" % (ok, ik),
                        iv[0].get_shape().as_list() if isinstance(iv, list) else iv.get_shape().as_list()
                    )
                    if ik == 'o':
                        with tf.name_scope(ik):
                            if isinstance(iv, list):
                                ov[ik] = [v * self._learning_rate for v in iv]
                            else:
                                ov[ik] = iv * self._learning_rate
                    print(
                        "(Empty._optimizer_core)after multiplying optimizer_ins[%s][%s]:" % (ok, ik),
                        ov[ik][0].get_shape().as_list() if isinstance(ov[ik], list) else ov[ik].get_shape().as_list()
                    )
        return self._empty_core(optimizer_ins)

    def __init__(
            self,
            pupil,
            regime='train',
            additional_metrics=None,
            flags=None,
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

        self._learning_rate = tf.placeholder(tf.float32, name='learning_rate', shape=[])

        self._hooks = dict(
            train_with_meta_optimizer_op=None,
            reset_optimizer_inference_pupil_storage=None,
            loss=None,
            pupil_trainable_initializers=None,
            train_optimizer_summary=None,
            learning_rate=None,
        )

        for add_metric in self._additional_metrics:
            self._hooks[add_metric] = None
        self._hooks['learning_rate'] = self._learning_rate
        self._debug_tensors = list()

        if regime == 'inference':
            pass
        else:
            print('Only inference regime is supported')
            raise NotImplementedError

        self._inference_graph()

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)
