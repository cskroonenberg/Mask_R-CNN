from jproperties import Properties
import numpy as np


def load_properties(file_path):

    raw_props = Properties()
    with open(file_path, 'rb') as props_reader:
        raw_props.load(props_reader)

    print("hey")
    props = {
        'model.img_size': parse_int_tuple(raw_props['model.img_size'][0]),
        'model.roi_size': parse_int_tuple(raw_props['model.roi_size'][0]),
        'model.n_labels': parse_int(raw_props['model.n_labels'][0]),
        'model.pos_thresh': parse_float(raw_props['model.pos_thresh'][0]),
        'model.neg_thresh': parse_float(raw_props['model.neg_thresh'][0]),
        'model.nms_thresh': parse_float(raw_props['model.nms_thresh'][0]),
        'model.hidden_dim': parse_int(raw_props['model.hidden_dim'][0]),
        'model.dropout': parse_float(raw_props['model.dropout'][0]),
        'model.backbone': raw_props['model.backbone'][0],
        'model.anc_scales': parse_float_list(raw_props['model.anc_scales'][0]),
        'model.anc_ratios': parse_float_list(raw_props['model.anc_ratios'][0]),
        'optim.lr': parse_float(raw_props['optim.lr'][0]),
        'optim.momentum': parse_float(raw_props['optim.momentum'][0])
    }

    return props


def parse_float(raw_float_str):
    if raw_float_str == 'None':
        return None
    return float(raw_float_str)


def parse_float_list(raw_list_str):
    if raw_list_str == 'None':
        return None
    np_arr = np.array(raw_list_str.strip('][').split(' '))
    np_arr = np_arr[np_arr != ''].tolist()
    clean_list = [float(elem) for elem in np_arr]
    return clean_list


def parse_int(raw_int_str):
    if raw_int_str == 'None':
        return None
    return int(raw_int_str)


def parse_int_tuple(raw_tuple_str):
    if raw_tuple_str == 'None':
        return None
    np_arr = np.array(raw_tuple_str.strip(')(').split(','))
    np_arr = np_arr[np_arr != ''].tolist()
    clean_tuple = tuple([int(elem) for elem in np_arr])
    return clean_tuple
