import itertools
import json

import numpy as np
import pandas as pd

from tf_crf_layer.crf_static_constraint_helper import allowed_transitions
from tokenizer_tools.tagset.NER.BILUO import BILUOEncoderDecoder

from typing import Sequence, Tuple, List, AnyStr, Dict


def generate_constraint_table(constraint_mapping: List[List[str]] = None, tag_dict: Dict[int, str] = None, tag_scheme='BIOUL'):
    constraint_matrix_list = []
    for constraint in constraint_mapping:
        all_tag_set = set()

        for entity in constraint:
            encoder = BILUOEncoderDecoder(entity)
            tag_set = encoder.all_tag_set()

            all_tag_set.update(tag_set)

        all_tag_dict = dict(i for i in tag_dict.items() if i[1] in all_tag_set)

        _, constraint_matrix = allowed_transitions(tag_scheme, all_tag_dict, tag_dict)

        constraint_matrix_list.append(constraint_matrix)

    constraint_matrix = np.stack([df.values for df in constraint_matrix_list], axis=0)
    return constraint_matrix


def filter_constraint(constraint: Dict[str, List[str]], tag_set: List[str] = None, intent_set: List[str] = None):
    """
    filter out entity not in tag_list

    :param constraint:
    :param tag_set:
    :param intent_set:
    :return:
    """

    valid_constraint = dict()
    for k, v in constraint.items():
        if intent_set and k not in intent_set:
            # discard
            continue

        # filter out tag that not in tag set
        if not tag_set:
            valid_v = v
        else:
            valid_v = list(filter(lambda x: x in tag_set, v))

        valid_constraint[k] = valid_v

    return valid_constraint


def sort_constraint(constraints: Dict[str, List[str]], intent_lookup_table: Dict[str, int]):
    sorted_constraint = sorted(constraints.items(), key=lambda x: intent_lookup_table[x[0]])
    sorted_constraint_list = [i[1] for i in sorted_constraint]

    return sorted_constraint_list
        

if __name__ == "__main__":
    with open('/home/howl/PycharmProjects/seq2annotation/data/constraint.json') as fd:
        constraint = json.load(fd)
        
    with open('/home/howl/PycharmProjects/seq2annotation/data/entity.txt') as fd:
        tag_list = [i.strip() for i in fd]
        tag_set_list = [BILUOEncoderDecoder(i).all_tag_set() for i in tag_list]

        tag_set = set()
        for tag in tag_set_list:
            tag_set.update(set(tag))

    # filter out entity not in tag_list
    valid_constraint = filter_constraint(constraint, tag_list)

    constraint_mapping = list(valid_constraint.values())
    tag_dict = dict(enumerate(tag_set))
        
    generate_constraint_table(constraint_mapping, tag_dict)

