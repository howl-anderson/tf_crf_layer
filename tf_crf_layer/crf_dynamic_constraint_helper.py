import itertools
import json

import pandas as pd

from tf_crf_layer.crf_static_constraint_helper import allowed_transitions
from tokenizer_tools.tagset.NER.BILUO import BILUOEncoderDecoder

from typing import Sequence, Tuple, List, AnyStr, Dict


def generate_constraint_table(constraint_mapping: List[List[str]] = None, tag_dict: Dict[int, str] = None, tag_scheme='BIOUL'):
    _, allowed_matrix = allowed_transitions(tag_scheme, tag_dict)
    constraint_matrix_list = []
    for constraint in constraint_mapping:
        mask_matrix = allowed_matrix.replace(True, False)

        from_list, to_list = None, None

        if not constraint:
            from_list, to_list = ['O'], ['O']

        for entity in constraint:
            encoder = BILUOEncoderDecoder(entity)
            tag_set = encoder.all_tag_set()

            product_of_tag = itertools.product(tag_set, tag_set)
            from_list, to_list = zip(*product_of_tag)

        mask_matrix.loc[from_list, to_list] = True

        constraint_matrix = allowed_matrix & mask_matrix
        constraint_serials = pd.Series(constraint_matrix.values.flatten())

        constraint_matrix_list.append(constraint_serials)

    constraint_matrix = pd.concat(constraint_matrix_list, axis=1).T
    return constraint_matrix.values


def filter_constraint(constraint: Dict[str, List[str]], tag_set: List[str] = None, intent_set: List[str] = None):
    # filter out entity not in tag_list
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
        
    generate_constraint_table(list(valid_constraint.values()), dict(enumerate(tag_set)))

