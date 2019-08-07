import copy

from learning_to_learn.useful_functions import construct, get_elem_from_nested


class Controller(object):
    """Controller is a class which instances are used for computing changing learning parameters. For example
    learning rate. It is also responsible for training stopping
    Usage:
        1. Construct controller by passing him 'storage' (a dictionary with monitored parameters, usually if used in
        _train method of Environment class 'storage' is self._storage) and specifications. Specifications is
        a dictionary with necessary parameters for computing the controlled value
        2. Use 'get' method to get current value
        3. If you wish to use Controller instance with new specifications you should:
            - add new private method to Controller which will be responsible for processing new specifications. It
              should take no arguments and return monitored value.
            - add elif entry in __init__ method for assigning 'get' with newly created method
            - if new approach requires parameters not provided in self._storage than add them. Also don't forget
              to pass this parameters to _update_storage method in the bottom of _train"""

    @staticmethod
    def create_change_tracking_specifications(specifications):
        if isinstance(specifications, list):
            old_specs = construct(specifications)
        if isinstance(specifications, dict):
            old_specs = [dict(specifications)]
        new_specs = dict()
        new_specs['old_specs'] = old_specs
        new_specs['type'] = 'changes_detector'
        return new_specs

    @staticmethod
    def spec_value_map():
        return dict(
            exponential_decay='init',
            fixed='value',
            linear='start',
            adaptive_change='init',
        )

    def __init__(self, storage, specifications):
        # print("(Controller.__init__)specifications:", specifications)
        self._storage = storage
        self._specifications = copy.deepcopy(specifications)
        if self._specifications['type'] == 'limit_steps':
            self.get = self._limit_steps
        elif self._specifications['type'] == 'exponential_decay':
            self.get = self._exponential_decay
        elif self._specifications['type'] == 'fixed':
            self.get = self._fixed
        elif self._specifications['type'] == 'periodic_truth':
            self.get = self._periodic_truth
        elif self._specifications['type'] == 'true_on_steps':
            self.get = self._true_on_steps
        elif self._specifications['type'] == 'always_false':
            self.get = self._always_false
        elif self._specifications['type'] == 'changes_detector':
            self._value_controllers = list()
            self._last_values = list()
            for value_specs in self._specifications['old_specs']:
                self._value_controllers.append(Controller(self._storage, value_specs))
                self._last_values.append(self._value_controllers[-1].get())
            self.get = self._changes_detector

        elif self._specifications['type'] == 'linear':
            self.get = self._linear
        elif self._specifications['type'] == 'adaptive_change':
            self.get = self._adaptive_change
            self._specifications = dict(self._specifications)
            self._specifications['impatience'] = 0
            self._specifications['value'] = self._specifications['init']
            self._specifications['line_len_after_prev_get_call'] = -1
            self._init_ops_for_adaptive_controller()
        elif self._specifications['type'] == 'fire_at_best':
            self.get = self._fire_at_best
            self._init_ops_for_adaptive_controller()
        elif self._specifications['type'] == 'while_progress':
            self.get = self._while_progress
            self._specifications = dict(self._specifications)
            self._specifications['impatience'] = 0
            self._specifications['cur_made_prog'] = False
            self._specifications['prev_made_prog'] = True
            self._init_ops_for_adaptive_controller()
            self._specifications['current_value'] = self._specifications['changing_parameter_controller'].get()
        elif self._specifications['type'] == 'while_progress_no_changing_parameter':
            self.get = self._while_progress_no_changing_parameter
            self._specifications = dict(self._specifications)
            self._specifications['impatience'] = 0
            self._init_ops_for_adaptive_controller()
        elif self._specifications['type'] == 'logarithmic_truth':
            self.get = self._logarithmic_truth
        else:
            raise ValueError(
                "Not supported controller type {}".format(repr(self._specifications['type']))
            )

    def _init_ops_for_adaptive_controller(self):
        if 'direction' not in self._specifications:
            self._specifications['direction'] = 'down'
        # if self._specifications['direction'] == 'down':
        #     self._specifications['comp_func'] = self._comp_func_gen(min)
        # else:
        #     self._specifications['comp_func'] = self._comp_func_gen(max)
        # print(self._storage)
        if 'best' not in self._specifications:
            if self._specifications['direction'] == 'down':
                self._specifications['best'] = float('+inf')
            else:
                self._specifications['best'] = float('-inf')
        self._specifications['line'] = get_elem_from_nested(
            self._storage,
            self._specifications['path_to_target_metric_storage']
        )

        # Length of list of target values when get() was called last time.
        # It is used to make controller do not change the value if no measurements were made.
        # For instance 'should_continue' and 'learning_rate' are called frequently whereas
        # validations are rare. Changes in length of list with target values are used
        # for detecting of new measurement and making a decision to increase impatience.
        self._specifications['line_len_after_prev_get_call'] = -1

    def _is_improved(self):
        if self._specifications['direction'] is 'down':
            return min(self._specifications['line'], default=float('+inf')) < self._specifications['best']
        return max(self._specifications['line'], default=float('-inf')) > self._specifications['best']

    def _update_best(self):
        if self._specifications['direction'] is 'down':
            self._specifications['best'] = min(self._specifications['line'], default=float('+inf'))
        else:
            self._specifications['best'] = max(self._specifications['line'], default=float('-inf'))

    def get_best_target_metric_value(self):
        return self._specifications.get('best', None)

    def get_target_metric_storage_path(self):
        return self._specifications.get('path_to_target_metric_storage', None)

    @staticmethod
    def _comp_func_gen(comp):
        def f(line):
            if len(line) == 0:
                return True
            elif len(line) == 1:
                return comp(line) == line[-1]
            else:
                return comp(line) == line[-1] and comp(line[:-1]) != line[-1]

        return f

    def _changes_detector(self):
        something_changed = False
        for idx, (last_value, controller) in enumerate(zip(self._last_values, self._value_controllers)):
            if last_value != controller.get():
                something_changed = something_changed or True
                self._last_values[idx] = controller.get()
        return something_changed

    def _exponential_decay(self):
        num_stairs = self._storage['step'] // self._specifications['period']
        returned_value = self._specifications['init']
        return returned_value * self._specifications['decay'] ** num_stairs

    def _linear(self):
        start = self._specifications['start']
        end = self._specifications['end']
        step_interval = self._specifications['interval']
        step = self._storage['step']
        if step < step_interval:
            return start + (end - start) * step / step_interval
        else:
            return end

    def _limit_steps(self):
        if self._storage['step'] >= self._specifications['limit']:
            return False
        else:
            return True

    def _fixed(self):
        return self._specifications['value']

    def _periodic_truth(self):
        if self._storage['step'] % self._specifications['period'] == 0:
            return True
        else:
            return False

    @staticmethod
    def _next_in_log_truth(start, factor):
        new_value = int(round(start * factor))
        if new_value <= start:
            return start + 1
        else:
            return new_value

    def _logarithmic_truth(self):
        while self._storage['step'] > self._specifications['start']:
            self._specifications['start'] = self._next_in_log_truth(
                self._specifications['start'],
                self._specifications['factor']
            )
        if self._storage['step'] == self._specifications['start'] \
                and self._specifications['start'] < self._specifications['end']:
            return True
        else:
            return False

    def _true_on_steps(self):
        if self._storage['step'] in self._specifications['steps']:
            return True
        else:
            return False

    def _adaptive_change(self):
        """Controller value does not change until specs['max_no_progress_points'] + 1
        no progress points are collected."""
        specs = self._specifications
        # if specs['comp_func'](specs['line']):
        if self._is_improved():
            specs['impatience'] = 0
            self._update_best()
            return specs['value']
        else:
            if specs['line_len_after_prev_get_call'] < len(specs['line']):
                specs['impatience'] += 1
                specs['line_len_after_prev_get_call'] = len(specs['line'])
                if specs['impatience'] > specs['max_no_progress_points']:
                    specs['value'] *= specs['decay']
                    specs['impatience'] = 0
            return specs['value']

    def _fire_at_best(self):
        specs = self._specifications
        # if specs['comp_func'](specs['line']) and specs['line_len_after_prev_get_call'] < len(specs['line']):
        if self._is_improved():
            # specs['line_len_after_prev_get_call'] = len(specs['line'])
            self._update_best()
            return True
        else:
            return False

    def _while_progress(self):  # for example if learning does not bring improvement return False
        """Returns False if two values of target parameter (learning_rate) did not improve the results
        or when previous parameter value did not bring improvement and specs['max_no_progress_points']
        points on target metric no progress has been made."""
        specs = self._specifications
        value = specs['changing_parameter_controller'].get()
        if specs['current_value'] == value:
            # if specs['comp_func'](specs['line']):
            if self._is_improved():
                specs['impatience'] = 0
                specs['cur_made_prog'] = True
                self._update_best()
                ret = True
            else:
                if not specs['prev_made_prog'] \
                        and specs['impatience'] > specs['max_no_progress_points']:
                    ret = False
                else:
                    if specs['line_len_after_prev_get_call'] < len(specs['line']):
                        specs['impatience'] += 1
                    ret = True
        else:
            if not specs['cur_made_prog'] and not specs['prev_made_prog']:
                return False
            else:
                specs['prev_made_prog'] = specs['cur_made_prog']
                specs['cur_made_prog'] = False
                specs['impatience'] = 0
                specs['current_value'] = value
                ret = True
        specs['line_len_after_prev_get_call'] = len(specs['line'])
        return ret

    def _while_progress_no_changing_parameter(self):
        specs = self._specifications
        # print("(Controller._while_progress_no_changing_parameter)specs['impatience']:",
        #       specs['impatience'])
        # if specs['comp_func'](specs['line']):
        if self._is_improved():
            self._update_best()
            specs['impatience'] = 0
            ret = True
        else:
            if specs['impatience'] > specs['max_no_progress_points']:
                ret = False
            else:
                if specs['line_len_after_prev_get_call'] < len(specs['line']):
                    specs['impatience'] += 1
                ret = True
        specs['line_len_after_prev_get_call'] = len(specs['line'])
        return ret

    @staticmethod
    def _always_false():
        return False

    @property
    def name(self):
        return self._specifications['name']

    @classmethod
    def get_logarithmic_truth_steps(cls, spec):
        steps = []
        step = spec['start']
        while True:
            step = cls._next_in_log_truth(
                step,
                spec['factor']
            )
            if step > spec['end']:
                steps.append(spec['end'])
                break
            steps.append(step)
        return steps
