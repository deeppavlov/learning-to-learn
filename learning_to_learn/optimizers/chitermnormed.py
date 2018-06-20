import tensorflow as tf
from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import construct_dict_without_none_entries, global_norm, normalize


class ChiTermNormed(Meta):
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

    def _add_chi_term_sum(
            self,
            optimizer_ins
    ):
        for ok, ov in optimizer_ins.items():
            if isinstance(ov['o'], list):
                ov['o'] = [
                    o + self._chi_contribution * normalize(theta, o)
                    for o, theta in zip(ov['o'],ov['theta'])
                ]
            else:
                ov['o'] = ov['o'] + self._chi_contribution * normalize(ov['theta'], ov['o'])
        return optimizer_ins

    def _add_chi_term_mul(
            self,
            optimizer_ins
    ):
        for ok, ov in optimizer_ins.items():
            if isinstance(ov['o'], list):
                ov['o'] = [
                    o + self._chi_contribution * normalize(theta * tf.square(o), o)
                    for o, theta in zip(ov['o'], ov['theta'])
                ]
            else:
                ov['o'] = ov['o'] + self._chi_contribution * normalize(
                    ov['theta'] * tf.square(ov['o']),
                    ov['o']
                )
        return optimizer_ins

    def _add_chi_term_exp(self, optimizer_ins):
        for ok, ov in optimizer_ins.items():
            if isinstance(ov['o'], list):
                ov['o'] = [
                    o*tf.exp(
                        self._chi_contribution * normalize(o*theta, o)
                    )
                    for o, theta in zip(ov['o'],ov['theta'])
                ]
            else:
                ov['o'] = ov['o'] * tf.exp(
                    self._chi_contribution * normalize(ov['theta']*ov['o'], ov['o'])
                )
        return optimizer_ins

    def _add_chi_term(
            self,
            optimizer_ins
    ):
        if self._chi_application == 'sum':
            optimizer_ins = self._add_chi_term_sum(optimizer_ins)
        elif self._chi_application == 'mul':
            optimizer_ins = self._add_chi_term_mul(optimizer_ins)
        elif self._chi_application == 'exp':
            optimizer_ins = self._add_chi_term_exp(optimizer_ins)
        return optimizer_ins

    def _optimizer_core(self, optimizer_ins, states, gpu_idx, permute=False):
        # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
        # optimizer_ins = self._forward_permute(optimizer_ins)
        self._multiply_by_factor(
            optimizer_ins,
            dict(
                theta=self._chi_contribution
            )
        )

        self._add_chi_term(optimizer_ins)

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
            get_theta=True,
            base_optimizer_type='sgd',
            chi_application='sum',
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
        self._get_omega = False
        self._normalizing = None
        self._inp_gradient_clipping = None

        self._base_optimizer_type = base_optimizer_type
        self._chi_application = chi_application

        self._learning_rate = tf.placeholder(tf.float32, name='learning_rate', shape=[])
        self._chi_contribution = tf.placeholder(tf.float32, name='chi_contribution', shape=[])
        self._hooks = dict(
            train_with_meta_optimizer_op=None,
            reset_optimizer_inference_pupil_storage=None,
            loss=None,
            pupil_trainable_initializers=None,
            train_optimizer_summary=None,
            learning_rate=None,
            chi_contribution=None
        )

        for add_metric in self._additional_metrics:
            self._hooks[add_metric] = None
        self._hooks['learning_rate'] = self._learning_rate
        self._hooks['chi_contribution'] = self._chi_contribution
        self._debug_tensors = list()

        if regime == 'inference':
            pass
        else:
            print('Only inference regime is supported')
            raise NotImplementedError

        self._inference_graph()

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)