from collections import OrderedDict
from useful_functions import InvalidArgumentError, search_in_nested_dictionary,\
                                  construct, paste_into_nested_structure, unite_dicts, remove_keys_from_dictionary


# general args parsing. Used for test and train methods
def set_controller_name_in_specs(controller_specs, name):
    if isinstance(controller_specs, dict):
        if 'name' not in controller_specs:
            controller_specs['name'] = name


def process_abbreviation_in_1_entry(key, value, method_name):
    new_value = construct(value)
    if key == 'stop':
        if isinstance(value, int):
            new_value = {'type': 'limit_steps', 'limit': value}
        set_controller_name_in_specs(new_value, 'stop')
    if key == 'batch_size':
        if isinstance(value, int):
            new_value = {'type': 'fixed', 'value': value}
        set_controller_name_in_specs(new_value, 'batch_size')
    if key == 'num_unrollings':
        if isinstance(value, int):
            new_value = {'type': 'fixed', 'value': value}
        set_controller_name_in_specs(new_value, 'num_unrollings')
    if key == 'checkpoint_steps':
        if isinstance(value, list):
            new_value = {'type': 'true_on_steps', 'steps': value}
        elif isinstance(value, int):
            new_value = {'type': 'periodic_truth', 'period': value}
        else:
            new_value = {'type': 'always_false'}
        set_controller_name_in_specs(new_value, 'checkpoint_steps')
    if key == 'learning_rate':
        set_controller_name_in_specs(new_value, 'learning_rate')

    if key == 'debug':
        if isinstance(value, int):
            new_value = {'type': 'true_on_steps', 'steps': [value]}
        else:
            new_value = None
        set_controller_name_in_specs(new_value, 'debug')
    if key == 'opt_inf_stop':
        if isinstance(value, int):
            new_value = {
                'type': 'limit_steps',
                'limit': value
            }
        set_controller_name_in_specs(new_value, 'optimizer_inference_num_steps')
    if key == 'reset_period':
        if isinstance(value, int):
            new_value = {
                'type': 'periodic_truth',
                'limit': value
            }
        set_controller_name_in_specs(new_value, 'optimizer_inference_num_steps')

    if method_name == 'train' or 'train_optimizer':
        if key == 'additions_to_feed_dict' or key == 'opt_inf_additions_to_feed_dict':
            # print('inside additions_to_feed_dict shortcuts processing')
            if new_value is not None:
                for addition in new_value:
                    # print('addition:', addition)
                    if not isinstance(addition['value'], dict):
                        # print('Removing of shortcut is happening now')
                        addition['value'] = {'type': 'fixed', 'value': addition['value']}
                        # print("addition['value']:", addition['value'])
                    set_controller_name_in_specs(addition['value'], addition['placeholder'])
    return new_value


def process_abbreviations(env_instance, set_of_kwargs, method_name):
    for key, value in set_of_kwargs.items():
        value = process_abbreviation_in_1_entry(key, value, method_name)
        set_of_kwargs[key] = value
    if search_in_nested_dictionary(set_of_kwargs, 'summary') is None:
        add_graph_to_summary = search_in_nested_dictionary(set_of_kwargs, 'add_graph_to_summary')
        train_summary_tensors = search_in_nested_dictionary(set_of_kwargs, 'train_summary_tensors')
        if train_summary_tensors is not None:
            if len(train_summary_tensors) > 0:
                summary_tensors_provided = True
            else:
                summary_tensors_provided = False
        else:
            summary_tensors_provided = False
        if add_graph_to_summary or summary_tensors_provided:
            set_of_kwargs['summary'] = True
        else:
            set_of_kwargs['summary'] = False
    process_datasets_shortcuts(env_instance, set_of_kwargs)
    process_batch_kwargs_shortcuts(set_of_kwargs, method_name)


def process_batch_kwargs_shortcuts(set_of_kwargs, method_name):
    if method_name == 'train':
        if 'train_batch_kwargs' not in set_of_kwargs:
            set_of_kwargs['train_batch_kwargs'] = dict()
        if 'num_unrollings' in set_of_kwargs:
            if 'num_unrollings' not in set_of_kwargs['train_batch_kwargs']:
                set_of_kwargs['train_batch_kwargs']['num_unrollings'] = set_of_kwargs['num_unrollings']
            del set_of_kwargs['num_unrollings']
        if 'vocabulary' in set_of_kwargs:
            if 'vocabulary' not in set_of_kwargs['train_batch_kwargs']:
                set_of_kwargs['train_batch_kwargs']['vocabulary'] = set_of_kwargs['vocabulary']
            del set_of_kwargs['vocabulary']
        if 'valid_batch_kwargs' not in set_of_kwargs:
            set_of_kwargs['valid_batch_kwargs'] = dict()
        if 'num_unrollings' in set_of_kwargs['train_batch_kwargs']:
            if 'num_unrollings' not in set_of_kwargs['valid_batch_kwargs']:
                set_of_kwargs['valid_batch_kwargs'] = {'num_unrollings': 1}
        if 'vocabulary' in set_of_kwargs['train_batch_kwargs']:
            if 'vocabulary' not in set_of_kwargs['valid_batch_kwargs']:
                set_of_kwargs['valid_batch_kwargs']['vocabulary'] = list(
                    set_of_kwargs['train_batch_kwargs']['vocabulary'])
    if method_name == 'test':
        if 'valid_batch_kwargs' not in set_of_kwargs:
            set_of_kwargs['valid_batch_kwargs'] = dict()
            if 'num_unrollings' in set_of_kwargs:
                set_of_kwargs['valid_batch_kwargs']['num_unrollings'] = {'num_unrollings': 1}
                del set_of_kwargs['num_unrollings']
            if 'vocabulary' in set_of_kwargs:
                set_of_kwargs['valid_batch_kwargs']['vocabulary'] = list(set_of_kwargs['vocabulary'])
                del set_of_kwargs['vocabulary']
    if method_name == 'train_optimizer':
        if 'train_batch_kwargs' not in set_of_kwargs:
            set_of_kwargs['train_batch_kwargs'] = dict()
        if 'vocabulary' in set_of_kwargs:
            if 'vocabulary' not in set_of_kwargs['train_batch_kwargs']:
                set_of_kwargs['train_batch_kwargs']['vocabulary'] = set_of_kwargs['vocabulary']
            del set_of_kwargs['vocabulary']
        if 'valid_batch_kwargs' not in set_of_kwargs:
            set_of_kwargs['valid_batch_kwargs'] = dict()
        if 'vocabulary' in set_of_kwargs['train_batch_kwargs']:
            if 'vocabulary' not in set_of_kwargs['valid_batch_kwargs']:
                set_of_kwargs['valid_batch_kwargs']['vocabulary'] = list(
                    set_of_kwargs['train_batch_kwargs']['vocabulary'])
        if 'num_unrollings' in set_of_kwargs:
            if 'num_unrollings' not in set_of_kwargs['train_batch_kwargs']:
                set_of_kwargs['train_batch_kwargs']['num_unrollings'] = set_of_kwargs['num_unrollings']
            if 'num_unrollings' not in set_of_kwargs['valid_batch_kwargs']:
                set_of_kwargs['valid_batch_kwargs'] = {'num_unrollings': 1}
            del set_of_kwargs['num_unrollings']


def process_datasets_shortcuts(env_instance,
                               set_of_kwargs):
    taken_names = list(env_instance.datasets.keys())
    train_dataset = process_single_dataset_shortcut(env_instance, set_of_kwargs, taken_names, 'train_dataset')
    if train_dataset is not None:
        set_of_kwargs['train_dataset'] = train_dataset
    train_datasets = process_single_dataset_shortcut(env_instance, set_of_kwargs, taken_names, 'train_datasets')
    if train_datasets is not None:
        set_of_kwargs['train_datasets'] = train_datasets
    opt_inf_train_datasets = process_single_dataset_shortcut(
        env_instance, set_of_kwargs, taken_names, 'opt_inf_train_datasets')
    if opt_inf_train_datasets is not None:
        set_of_kwargs['opt_inf_train_datasets'] = opt_inf_train_datasets
    opt_inf_validation_datasets = process_single_dataset_shortcut(
        env_instance, set_of_kwargs, taken_names, 'opt_inf_validation_datasets')
    if opt_inf_validation_datasets is not None:
        set_of_kwargs['opt_inf_validation_datasets'] = opt_inf_validation_datasets
    validation_datasets = process_validation_datasets_shortcuts(env_instance, set_of_kwargs, taken_names)
    set_of_kwargs['validation_datasets'] = validation_datasets
    if 'opt_inf_train_datasets' in set_of_kwargs and 'opt_inf_validation_datasets' in set_of_kwargs:
        if len(set_of_kwargs['opt_inf_validation_datasets']) == 1:
            set_of_kwargs['opt_inf_validation_datasets'] *= len(set_of_kwargs['opt_inf_train_datasets'])


def process_validation_datasets_shortcuts(env_instance,
                                          set_of_kwargs,
                                          taken_names):
    validation_datasets = list()
    if 'validation_datasets' in set_of_kwargs:
        taken_names.extend(set_of_kwargs['validation_datasets'].keys())
        validation_datasets.extend(set_of_kwargs['validation_datasets'])
    if 'validation_dataset_names' in set_of_kwargs:
        for name in set_of_kwargs['validation_dataset_names']:
            if name not in env_instance.datasets.keys():
                raise InvalidArgumentError("Wrong value '%s' of variable '%s'\nAllowed values: '%s'" %
                                           (name,
                                            "set_of_kwargs['validation_dataset_names']",
                                            list(env_instance.datasets.keys())))
            validation_datasets.append([env_instance.datasets[name], name])
    if 'validation_dataset_texts' in set_of_kwargs:
        for text in set_of_kwargs['validation_dataset_texts']:
            key, value = process_input_text_dataset(text, taken_names)
            taken_names.append(key)
            validation_datasets.append([value, key])
    if 'validation_dataset_filenames' in set_of_kwargs:
        for filename in set_of_kwargs['validation_dataset_filenames']:
            key, value = env_instance.process_dataset_filename(filename)
            taken_names.append(key)
            validation_datasets.append([value, key])
    _ = remove_keys_from_dictionary(
        set_of_kwargs,
        ['validation_datasets', 'validation_dataset_names', 'validation_dataset_texts', 'validation_dataset_filenames'])
    return validation_datasets


def process_shortcut_with_dataset_names(
        env_instance,
        set_of_kwargs,
        key
):
    all_names_are_present = True
    for name in set_of_kwargs[key]:
        all_names_are_present = all_names_are_present and (name in list(env_instance.datasets.keys()))
    if not all_names_are_present:
        raise InvalidArgumentError("Wrong value '%s' of variable '%s'\nAllowed values: '%s'" %
                                   (set_of_kwargs[key], "set_of_kwargs[%s]" % key,
                                    list(env_instance.datasets.keys())),
                                   set_of_kwargs[key],
                                   "set_of_kwargs[%s]" % key,
                                   list(env_instance.datasets.keys()))
    return [[env_instance.datasets[name], name] for name in list(env_instance.datasets.keys())]


def process_shortcut_with_dataset_texts(
        set_of_kwargs,
        taken_names,
        key
):
    ret = [list(process_input_text_dataset(text, taken_names))[::-1]
           for text in set_of_kwargs[key]]
    taken_names.extend([d[1] for d in ret])
    return ret


def process_shortcut_with_dataset_filename(
        set_of_kwargs,
        taken_names,
        key
):
    ret = [list(process_dataset_filename(filename, taken_names))[::-1]
           for filename in set_of_kwargs[key]]
    taken_names.extend([d[1] for d in ret])
    return ret


def process_standard_combination_of_list_of_dataset_abbreviations(
        env_instance,
        set_of_kwargs,
        taken_names,
        checked_keys
):
    # print('(process_standard_combination_of_list_of_dataset_abbreviations)set_of_kwargs:', set_of_kwargs)
    # print('(process_standard_combination_of_list_of_dataset_abbreviations)taken_names:', taken_names)
    # print('(process_standard_combination_of_list_of_dataset_abbreviations)checked_keys:', checked_keys)
    if checked_keys[0] in set_of_kwargs:
        taken_names.extend([ds[1] for ds in set_of_kwargs[checked_keys[0]]])
        ret = set_of_kwargs[checked_keys[0]]
    elif checked_keys[1] in set_of_kwargs:
        ret = process_shortcut_with_dataset_names(env_instance, set_of_kwargs, checked_keys[1])
    elif checked_keys[2] in set_of_kwargs:
        ret = process_shortcut_with_dataset_texts(set_of_kwargs, taken_names, checked_keys[2])
    elif checked_keys[3] in set_of_kwargs:
        ret = process_shortcut_with_dataset_filename(set_of_kwargs, taken_names, checked_keys[3])
    else:
        ret = None
    _ = remove_keys_from_dictionary(
        set_of_kwargs, checked_keys)
    return ret


def process_single_dataset_shortcut(
        env_instance,
        set_of_kwargs,
        taken_names,
        key
):
    if key == 'train_dataset':
        if 'train_dataset' in set_of_kwargs:
            taken_names.extend(list(set_of_kwargs['train_dataset'].keys()))
            ret = set_of_kwargs['train_dataset']
        elif 'train_dataset_name' in set_of_kwargs:
            if set_of_kwargs['train_dataset_name'] not in env_instance.datasets.keys():
                raise InvalidArgumentError("Wrong value '%s' of variable '%s'\nAllowed values: '%s'" %
                                           (set_of_kwargs['train_dataset_name'], "set_of_kwargs['train_dataset_name']",
                                            list(env_instance.datasets.keys())),
                                           set_of_kwargs['train_dataset_name'],
                                           "set_of_kwargs['train_dataset_name']",
                                           list(env_instance.datasets.keys()))
            ret = [env_instance.datasets[set_of_kwargs['train_dataset_name']], set_of_kwargs['train_dataset_name']]
        elif 'train_dataset_text' in set_of_kwargs:
            key, value = process_input_text_dataset(set_of_kwargs['train_dataset_text'], taken_names)
            taken_names.append(key)
            ret = [value, key]
        elif 'train_dataset_filename' in set_of_kwargs:
            key, value = process_dataset_filename(env_instance, set_of_kwargs['train_dataset_filename'])
            taken_names.append(key)
            ret = [value, key]
        else:
            ret = None
        _ = remove_keys_from_dictionary(
            set_of_kwargs, ['train_dataset', 'train_dataset_name', 'train_dataset_text', 'train_dataset_filename'])
        return ret

    if key == 'train_datasets':
        return process_standard_combination_of_list_of_dataset_abbreviations(
            env_instance, set_of_kwargs, taken_names,
            ['train_datasets', 'train_dataset_names', 'train_dataset_texts', 'train_dataset_filenames'])

    if key == 'opt_inf_train_datasets':
        return process_standard_combination_of_list_of_dataset_abbreviations(
            env_instance, set_of_kwargs, taken_names,
            ['opt_inf_train_datasets', 'opt_inf_train_dataset_names',
             'opt_inf_train_dataset_texts', 'opt_inf_train_dataset_filenames']
        )

    if key == 'opt_inf_validation_datasets':
        return process_standard_combination_of_list_of_dataset_abbreviations(
            env_instance, set_of_kwargs, taken_names,
            ['opt_inf_validation_datasets', 'opt_inf_validation_dataset_names',
             'opt_inf_validation_dataset_texts', 'opt_inf_validation_dataset_filenames']
        )


def process_input_text_dataset(input, taken_names):
    idx = 0
    base = 'default_'
    new_key = base + str(idx)
    while new_key in taken_names:
        idx += 1
        new_key = base + str(idx)
    return new_key, input


def process_dataset_filename(env_instance, input):
    splitted = input.split('/')
    env_instance.datasets[splitted[-1]] = input
    return splitted[-1], input


def parse_1_set_of_kwargs(env_instance,
                          kwargs_to_parse,
                          method_name,
                          repeated_key,
                          only_repeated,
                          old_arguments=None):
    # print('\n\n_parse_1_set_of_kwargs method:\nkwargs_to_parse:\n', kwargs_to_parse, '\nmethod_name:\n',
    #       method_name, '\nrepeated_key:\n', repeated_key, '\nonly_repeated:\n', only_repeated, '\nold_arguments:\n',
    #       old_arguments)
    # print("(parse_1_set_of_kwargs)kwargs_to_parse:", kwargs_to_parse)
    kwargs_to_parse = construct(kwargs_to_parse)
    process_abbreviations(env_instance, kwargs_to_parse, method_name)
    # print("(parse_1_set_of_kwargs)kwargs_to_parse:", kwargs_to_parse)
    if old_arguments is None:
        current_arguments = env_instance.get_default_method_parameters(method_name)
        if only_repeated:
            current_arguments = current_arguments[repeated_key]
    else:
        current_arguments = construct(old_arguments)

    # print("(parse_1_set_of_kwargs)current_arguments:", current_arguments)

    for key, value in kwargs_to_parse.items():
        paste_into_nested_structure(current_arguments, key, value)

    #print('current_arguments:\n', current_arguments)
    return current_arguments


def parse_list_of_sets_of_kwargs(env_instance,
                                 list_of_sets,
                                 method_name,
                                 repeated_key):
    # print('\n\n_parse_list_of_sets_of_kwargs method\nlist_of_sets:\n', list_of_sets, '\nmethod_name:\n', method_name,
    #       '\nrepeated_key:\n', repeated_key)
    # print("(parse_list_of_sets_of_kwargs)list_of_sets:", list_of_sets)
    parsed = parse_1_set_of_kwargs(env_instance,
                                   list_of_sets[0],
                                   method_name,
                                   repeated_key,
                                   False)

    parsed[repeated_key] = [parsed[repeated_key]]

    repeated_parsed = parsed[repeated_key][0]
    for kwargs_set in list_of_sets[1:]:
        repeated_parsed = parse_1_set_of_kwargs(env_instance,
                                                kwargs_set,
                                                method_name,
                                                repeated_key,
                                                True,
                                                old_arguments=repeated_parsed)
        parsed[repeated_key].append(repeated_parsed)
    #print('parsed:\n', parsed)
    return parsed


def parse_train_method_arguments(env_instance,
                                 train_args,
                                 train_kwargs,
                                 set_passed_parameters_as_default=False):
    """Performs parsing of 'train' and 'train_assistant' method arguments. Optionally updates
    self._pupil_default_training or self._assistant_default_training.
    Args:
        train_args: args passed to train method
        set_passed_parameters_as_default: defines if reset of default parameters is needed
        train_kwargs: kwargs passed to train method
    Returns:
        a dictionary of start parameters (same format as self._default_start)
        a list of dictionaries with all parameters required for launch (each dictionary has the same format as
            self._pupil_default_training or self._assistant_default_training)"""

    #print('\n\n_parse_train_method_arguments method\ntrain_args:', train_args, '\ntrain_kwargs:', train_kwargs)
    if len(train_args) == 0:
        #print('train_kwargs:', train_kwargs)
        parsed_arguments = parse_list_of_sets_of_kwargs(env_instance,
                                                        [train_kwargs],
                                                        'train',
                                                        'run')
    else:
        parsed_arguments = parse_list_of_sets_of_kwargs(env_instance,
                                                        train_args,
                                                        'train',
                                                        'run')

    return parsed_arguments


def parse_train_optimizer_method_arguments(
    env_instance,
    train_args,
    train_kwargs,
    set_passed_parameters_as_default=False
):
    if len(train_args) == 0:
        #print('train_kwargs:', train_kwargs)
        parsed_arguments = parse_list_of_sets_of_kwargs(env_instance,
                                                        [train_kwargs],
                                                        'train_optimizer',
                                                        'run')
    else:
        parsed_arguments = parse_list_of_sets_of_kwargs(env_instance,
                                                        train_args,
                                                        'train_optimizer',
                                                        'run')

    return parsed_arguments


# Following functions are used for composing kwargs for hyperparameters tuning
def parse_hp_name(hp_name):
    if '[' in hp_name:
        [name, list_indices] = hp_name.split('[')
        list_indices = list_indices[:-1].split(',')
        list_indices = [int(index) for index in list_indices]
        return name, list_indices
    return hp_name, None


def process_build_hp_text_abbreviation(hp_name, hp_values):
    name, list_indices = parse_hp_name(hp_name)
    return {name: {'hp_type': 'build_hp',
                   'type': None,
                   'fixed': None,
                   'varying': hp_values,
                   'list_indices': list_indices,
                   'share': None,
                   'controller': False}}


def spot_direction(hp_name):
    if hp_name == 'num_unrollings':
        return 'batch_kwarg'
    return 'batch_kwarg'


def process_build_hp_abbreviations(build_hps):
    build_hps = construct(build_hps)
    new_build_hps = dict()
    for hp_name, hp_values_and_specs in build_hps.items():
        if isinstance(hp_values_and_specs, list):
            new_build_hps.update(process_build_hp_text_abbreviation(hp_name, hp_values_and_specs))
        else:
            if 'hp_type' not in hp_values_and_specs:
                hp_values_and_specs['hp_type'] = 'build_hp'
            if 'list_indices' not in hp_values_and_specs:
                hp_values_and_specs['list_indices'] = None
            elif isinstance(hp_values_and_specs['list_indices'], int):
                hp_values_and_specs['list_indices'] = [hp_values_and_specs['list_indices']]
            if 'share' not in hp_values_and_specs:
                hp_values_and_specs['share'] = None
            hp_values_and_specs['controller'] = False
            hp_values_and_specs['type'] = None
            hp_values_and_specs['fixed'] = None
            if hp_values_and_specs['share'] is not None:
                if 'direction' not in hp_values_and_specs['share']:
                    default = spot_direction(hp_name)
                    if spot_direction(hp_name) is None:
                        hp_values_and_specs['share']['direction'] = 'batch_kwarg'
                    else:
                        hp_values_and_specs['share']['direction'] = default
                if 'controller' not in hp_values_and_specs['share']:
                    hp_values_and_specs['share']['controller'] = True
            new_build_hps[hp_name] = hp_values_and_specs
    new_build_hps = reshape_1_index_hps(new_build_hps)
    return new_build_hps


def process_other_hp_dictionary_absence(hp_name, hp_values):
    name, list_indices = parse_hp_name(hp_name)
    return {name: {'hp_type': 'additional_placeholder',
                   'type': 'fixed',
                   'fixed': dict(),
                   'varying': hp_values,
                   'list_indices': list_indices,
                   'share': None,
                   'controller': True}}


def process_other_hp_specs_absences(other_hps):
    other_hps = construct(other_hps)
    new_other_hps = dict()
    for hp_name, hp_values_and_specs in other_hps.items():
        if isinstance(hp_values_and_specs, list):
            new_other_hps.update(process_other_hp_dictionary_absence(hp_name, hp_values_and_specs))
        else:
            if 'list_indices' not in hp_values_and_specs:
                hp_values_and_specs['list_indices'] = None
            elif isinstance(hp_values_and_specs['list_indices'], int):
                hp_values_and_specs['list_indices'] = [hp_values_and_specs['list_indices']]
            if 'hp_type' not in hp_values_and_specs:
                hp_values_and_specs['hp_type'] = 'additional_placeholder'
            if 'controller' not in hp_values_and_specs:
                hp_values_and_specs['controller'] = True
            if 'type' not in hp_values_and_specs:
                if hp_values_and_specs['controller']:
                    hp_values_and_specs['type'] = 'fixed'
                else:
                    hp_values_and_specs['type'] = None
            if 'fixed' not in hp_values_and_specs:
                if hp_values_and_specs['controller']:
                    hp_values_and_specs['fixed'] = dict()
                else:
                    hp_values_and_specs['fixed'] = None
            hp_values_and_specs['share'] = None
            new_other_hps[hp_name] = hp_values_and_specs
    return new_other_hps


def expand_varying_entry(hps):
    hps = construct(hps)
    new_hps = dict()
    for hp_name, hp_values_and_specs in hps.items():
        if hp_values_and_specs['controller'] and isinstance(hp_values_and_specs['varying'], list):
            hp_values_and_specs['varying'] = {'value': hp_values_and_specs['varying']}
        new_hps[hp_name] = hp_values_and_specs
    return new_hps


def reshape_1_index_hps(hps):
    for hp_values_and_specs in hps.values():
        if hp_values_and_specs['list_indices'] is not None:
            if len(hp_values_and_specs['list_indices']) == 1:
                varying = hp_values_and_specs['varying']
                if isinstance(varying, list):
                    hp_values_and_specs['varying'] = [varying]
                else:
                    for spec_name, values in varying.items():
                        varying[spec_name] = [values]
                    hp_values_and_specs['varying'] = varying
    return hps


def process_other_hp_abbreviations(other_hps):
    other_hps = process_other_hp_specs_absences(other_hps)
    other_hps = expand_varying_entry(other_hps)
    other_hps = reshape_1_index_hps(other_hps)
    return other_hps


def create_controller_template(hp_name, hp_specs_and_values):
    template = dict()
    template['name'] = hp_name
    template['type'] = hp_specs_and_values['type']
    for key, value in hp_specs_and_values['fixed'].items():
        template[key] = value
    for key in hp_specs_and_values['varying'].keys():
        template[key] = 'not_specified'
    return template


def create_insert_instructions(hps):
    insert_instructions = dict()
    for hp_name, hp_specs_and_values in hps.items():
        hp_insert_instructions = dict()
        if hp_specs_and_values['controller']:
            template = create_controller_template(hp_name, hp_specs_and_values)
        else:
            template = None

        list_indices = hp_specs_and_values['list_indices']
        if list_indices is None:
            list_indices = [None]
        hp_insert_instructions['controller_template'] = template
        hp_insert_instructions['share'] = hp_specs_and_values['share']
        hp_type = hp_specs_and_values['hp_type']
        for list_index in list_indices:
            insert_instructions[(hp_type, hp_name, list_index)] = hp_insert_instructions
    return insert_instructions


def create_hp_type_groups(insert_instructions):
    groups = {'built-in': list(),
              'build_hp': list(),
              'additional_placeholder': list(),
              'batch_kwarg': list()}
    for hp_name, instruction in insert_instructions.items():
        groups[instruction['hp_type']].append(hp_name)
    return groups


def extract_and_sort_values(formalized_name, hp_specs_and_values):
    # print('\nextract_and_sort_values')
    # print('formalized_name:', formalized_name)
    # print('hp_specs_and_values:', hp_specs_and_values)
    varying = hp_specs_and_values['varying']
    if formalized_name[2] is None:
        if formalized_name[3] is None:
            return sorted(varying)
        else:
            return sorted(varying[formalized_name[3]])
    else:
        values_set_index = hp_specs_and_values['list_indices'].index(formalized_name[2])
        if formalized_name[3] is None:
            return sorted(varying[values_set_index])
        else:
            return sorted(varying[formalized_name[3]][values_set_index])


def separate_hps(hp_name, hp_specs_and_values):
    # print('\nseparate_hps')
    # print('hp_name:', hp_name)
    # print('hp_specs_and_values:', hp_specs_and_values)
    hp_type = hp_specs_and_values['hp_type']
    list_indices = hp_specs_and_values['list_indices']
    if list_indices is None:
        list_indices = [None]
    if isinstance(hp_specs_and_values['varying'], list):
        varying_keys = [None]
    else:
        varying_keys = list(hp_specs_and_values['varying'].keys())
    formalized_names = list()
    for list_index in list_indices:
        for varying_key in varying_keys:
            formalized_names.append((hp_type, hp_name, list_index, varying_key))

    formalized_hp = dict()
    for name in formalized_names:
        formalized_hp[name] = extract_and_sort_values(name, hp_specs_and_values)
    # print('end of separate hps\n')
    return formalized_hp


def separate_all_hps(not_formalized_hps):
    formalized_hps = dict()
    for hp_name, hp_specs_and_values in not_formalized_hps.items():
        formalized_hps.update(separate_hps(hp_name, hp_specs_and_values))
    return formalized_hps


def split_to_groups_by_hp_type(hps_formalized):
    built_in = dict()
    build_hp = dict()
    additional_placeholder = dict()
    batch_kwarg = dict()
    for hp_formalized_name, values in hps_formalized.items():
        if hp_formalized_name[0] == 'built-in':
            built_in[hp_formalized_name] = values
        elif hp_formalized_name[0] == 'build_hp':
            build_hp[hp_formalized_name] = values
        elif hp_formalized_name[0] == 'additional_placeholder':
            additional_placeholder[hp_formalized_name] = values
        elif hp_formalized_name[0] == 'batch_kwarg':
            batch_kwarg[hp_formalized_name] = values
    return [build_hp, built_in, batch_kwarg, additional_placeholder]


def split_to_groups_on_index(hps, index):
    # print('\nsplit_to_groups_on_index')
    # print('index:', index)
    # print('hps:', hps)
    groups = list()
    group = OrderedDict()
    old_key_entry = 'it_is_not_parameter_name_list_index_or_controller_specification'
    for key, value in hps.items():
        if key[index] != old_key_entry:
            groups.append(group)
            group = OrderedDict()
            old_key_entry = key[index]
        group[key] = value
    groups.append(group)
    groups = groups[1:]
    # print('groups:', groups)
    return groups


def sort(hps, index, key_length):
    # print('\nsort')
    # print('hps:', hps)
    hps = OrderedDict(sorted(hps.items(), key=lambda item: item[0][index]))
    if index < key_length - 1:
        groups = split_to_groups_on_index(hps, index)
        # print('\nsort')
        # print('groups:', groups)
        sorted_groups = list()
        for group in groups:
            # print('group:', group)
            sorted_groups.append(sort(group, index+1, key_length))
        # print('\nsort')
        # print('sorted_groups:', sorted_groups)
        print('index=%s, hps:' % index, unite_dicts(sorted_groups))
        return unite_dicts(sorted_groups)
    print('index=%s, hps:' % index, hps)
    return hps


def sort_hps(hps):
    """alphabetically, than by index, and finally by controller specification key"""
    # print('\nsort_hps')
    if len(hps) > 0:
        key_length = len(list(hps.keys())[0])
        # print('key_length:', key_length)
        hps = OrderedDict(hps.items())
        # print('hps:', hps)
        hps = sort(hps, 1, key_length)
        # print('(sort_hps)hps:', hps)
        # print('\n')
        return hps
    return OrderedDict()


def expand(list_of_combinations, hp_name, values):
    new_combinations = list()
    for value in values:
        if len(list_of_combinations) > 0:
            for old_combination in list_of_combinations:
                start = OrderedDict([(hp_name, value)])
                start.update(old_combination)
                new_combinations.append(start)
        else:
            new_combinations.append(OrderedDict([(hp_name, value)]))
    return new_combinations


def mix_hps(hps):
    list_of_combinations = list()
    for hp_name, values in reversed(list(hps.items())):
        list_of_combinations = expand(list_of_combinations, hp_name, values)
    return list_of_combinations


def create_empty_insertions_template(insert_instructions):
    # print('\ncreate_empty_insertions_template')
    template = dict()
    for key, value in insert_instructions.items():
        # print('key:', key)
        # print('value:', value)
        if value['controller_template'] is not None:
            paste_value = value['controller_template']
        else:
            paste_value = 'not_specified'
        if value['share'] is None:
            share = None
        else:
            if value['share']['controller']:
                if value['share']['direction'] == 'additional_placeholder':
                    share_paste = {'placeholder': key[1],
                                   'value': {'type': 'fixed',
                                             'value': 'not_specified',
                                             'name': key[1]}}
                else:
                    share_paste = {'type': 'fixed',
                                   'value': 'not_specified',
                                   'name': key[1]}
            else:
                share_paste = 'not_specified'
            share = {'hp_type': value['share']['direction'],
                     'hp_name': key[1],
                     'list_index': key[2],
                     'paste': share_paste}
        if key[0] == 'additional_placeholder':
            template[key] = {'hp_type': key[0],
                             'hp_name': key[1],
                             'list_index': key[2],
                             'paste': {'placeholder': key[1], 'value': paste_value},
                             'share': share}
        else:
            template[key] = {'hp_type': key[0],
                             'hp_name': key[1],
                             'list_index': key[2],
                             'paste': paste_value,
                             'share': share}
    return template


def create_one_combination_insertions(insert_template, hp_combination):
    # print('\ncreate_one_combination_insertions')
    combination_insertions = construct(insert_template)
    for hp_name, value in hp_combination.items():
        # print('hp_name:', hp_name)
        # print('value:', value)
        insert_key = (hp_name[0], hp_name[1], hp_name[2])
        single_tmpl = combination_insertions[insert_key]
        # print('single_tmpl:', single_tmpl)
        # print('hp_name[0]:', hp_name[0])
        # print('hp_name[3]:', hp_name[3])
        if hp_name[0] == 'additional_placeholder':
            if hp_name[3] is not None:
                single_tmpl['paste']['value'][hp_name[3]] = value
            else:
                single_tmpl['paste']['value'] = value
        else:
            if hp_name[3] is not None:
                single_tmpl['paste'][hp_name[3]] = value
            else:
                single_tmpl['paste'] = value
        if single_tmpl['share'] is not None:
            share = single_tmpl['share']
            if share['hp_type'] == 'additional_placeholder':
                share['paste']['value']['value'] = value
            else:
                share['paste']['value'] = value
    # print('\n')
    return combination_insertions


def create_insertions(insert_instructions, hp_combinations):
    # print('\ncreate_insertions')
    insertions = list()
    insert_template = create_empty_insertions_template(insert_instructions)
    for hp_combination in hp_combinations:
        # print('insert_template:', insert_template)
        insertions.append(list(create_one_combination_insertions(insert_template, hp_combination).values()))
    # print('\n')
    return insertions


def formalize_and_create_insertions(hps):
    # print('\nformalize_and_create_insertions')
    insert_instructions = create_insert_instructions(hps)
    # print('insert_instructions:', insert_instructions)
    hps = separate_all_hps(hps)
    # print('hps:', hps)
    hps_by_groups = split_to_groups_by_hp_type(hps)
    # print('hps_by_groups:', hps_by_groups)
    hps = OrderedDict()
    for hp_group in hps_by_groups:
        hp_group = sort_hps(hp_group)
        # print('hp_group:', hp_group)
        hps.update(hp_group)
    # print('(formalize_and_create_insertions) before mixing hps:', hps)
    hp_combinations = mix_hps(hps)
    # print('hp_combinations:', hp_combinations)
    combination_insertions = create_insertions(insert_instructions, hp_combinations)
    # print('combination_insertions:', combination_insertions)
    # print('\n')
    return hp_combinations, combination_insertions


def formalize_and_create_insertions_for_other_hps(hps):
    hps = process_other_hp_abbreviations(hps)
    hp_combinations, combination_insertions = formalize_and_create_insertions(hps)
    return hp_combinations, combination_insertions


def post_process_build_insertions(combination_insertions):
    post_processed = list()
    for one_combination_insertions in combination_insertions:
        one_combination_processed = list()
        for one_insertion in one_combination_insertions:
            del one_insertion['hp_type']
            share = one_insertion['share']
            del one_insertion['share']
            one_combination_processed.append((one_insertion, share))
        post_processed.append(one_combination_processed)
    return post_processed


def formalize_and_create_insertions_for_build_hps(hps):
    hps = process_build_hp_abbreviations(hps)
    # print('\nformalize_and_create_insertions_for_build_hps')
    # print('hps:', hps)
    hp_combinations, combination_insertions = formalize_and_create_insertions(hps)
    # print('hp_combinations:', hp_combinations)
    # print('combination_insertions:', combination_insertions)
    post_processed_combination_insertions = post_process_build_insertions(combination_insertions)
    # print('\n')
    return hp_combinations, post_processed_combination_insertions


def insert_not_build_hp(kwargs, one_hp_insertion):
    if one_hp_insertion['hp_type'] == 'built-in':
        if one_hp_insertion['list_index'] is None:
            kwargs[one_hp_insertion['hp_name']] = one_hp_insertion['paste']
        else:
            kwargs[one_hp_insertion['hp_name']][one_hp_insertion['list_index']] = one_hp_insertion['paste']
    if one_hp_insertion['hp_type'] == 'batch_kwarg':
        if 'train_batch_kwargs' not in kwargs:
            kwargs['train_batch_kwargs'] = dict()
        if one_hp_insertion['list_index'] is None:
            kwargs['train_batch_kwargs'][one_hp_insertion['hp_name']] = one_hp_insertion['paste']
        else:
            kwargs['train_batch_kwargs'][one_hp_insertion['hp_name']][one_hp_insertion['list_index']] = \
                one_hp_insertion['paste']
    if one_hp_insertion['hp_type'] == 'additional_placeholder':
        if 'additions_to_feed_dict' not in kwargs:
            kwargs['additions_to_feed_dict'] = list()
        kwargs['additions_to_feed_dict'].append(one_hp_insertion['paste'])
    return kwargs


def create_1_set_of_args_for_launches(kwargs, one_set_of_args_insertions):
    kwargs = construct(kwargs)
    for one_hp_insertion in one_set_of_args_insertions:
        kwargs = insert_not_build_hp(kwargs, one_hp_insertion)
    return kwargs


def create_all_args_for_launches(kwargs, all_insertions):
    args_for_launches = list()
    if len(all_insertions) > 0:
        for one_set_of_kwargs_insertions in all_insertions:
            args_for_launches.append(create_1_set_of_args_for_launches(kwargs, one_set_of_kwargs_insertions))
    else:
        args_for_launches.append(kwargs)
    return args_for_launches


def apply_share(kwargs, share):
    if share is not None:
        kwargs = insert_not_build_hp(kwargs, share)
    return kwargs


def apply_shares(kwargs, shares):
    for share in shares:
        kwargs = apply_share(kwargs, share)
    return kwargs


def configure_args_for_launches(env_instance, args_for_launches, shares):
    args_for_launches_to_be_used = construct(args_for_launches)
    parsed = list()
    for to_be_used in args_for_launches_to_be_used:
        with_shares = apply_shares(to_be_used, shares)
        one_parsed = parse_train_method_arguments(env_instance,
                                                  [],
                                                  with_shares,
                                                  set_passed_parameters_as_default=False)
        start_specs = one_parsed['start_specs']
        run_specs_set = one_parsed['run']
        del one_parsed['session_specs']
        parsed.append((start_specs, run_specs_set))
    return parsed
