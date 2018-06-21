import tensorflow as tf
from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import construct_dict_without_none_entries


class Empty(Meta):

    @staticmethod
    def check_kwargs(**kwargs):
        pass

    def _create_optimizer_states(self, num_exercises, var_scope, gpu_idx):
        return list()

    def _optimizer_core(self, optimizer_ins, states, gpu_idx, permute=False):
        mul_dict = dict(
            sigma=self._learning_rate
        )
        if self._get_omega_and_beta:
            mul_dict['omega'] = self._learning_rate
            mul_dict['beta'] = self._learning_rate

        self._multiply_by_factor(
            optimizer_ins,
            mul_dict
        )
        return self._empty_core(optimizer_ins)

    def __init__(
            self,
            pupil,
            regime='train',
            additional_metrics=None,
            flags=None,
            get_theta=False,
            get_omega_and_beta=False,
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
        self._get_theta = get_theta
        self._get_omega_and_beta = get_omega_and_beta
        self._matrix_mod = matrix_mod
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
