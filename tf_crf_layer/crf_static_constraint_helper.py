"""
Conditional random field

Some code are copied from AllenNLP
"""
from collections import namedtuple
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from tf_crf_layer.exceptions import ConfigurationError

ConstraintType = namedtuple('ConstraintType', ["BIO", "IOB1", "BIOUL", "BMES"])
constraint_type = ConstraintType("BIO", "IOB1", "BIOUL", "BMES")


def extract_tag_entity(label: str) -> (str, str):
    # extract tag and entity from label
    if label in ("START", "END"):
        tag = label
        entity = ""
    else:
        tag = label[0]
        entity = label[1:]

    return tag, entity


# def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> (List[Tuple[int, int]], pd.DataFrame):
#     """
#     Given labels and a constraint type, returns the allowed transitions. It will
#     additionally include transitions for the start and end states, which are used
#     by the conditional random field.
#
#     Parameters
#     ----------
#     constraint_type : ``str``, required
#         Indicates which constraint to apply. Current choices are
#         "BIO", "IOB1", "BIOUL", and "BMES".
#     labels : ``Dict[int, str]``, required
#         A mapping {label_id -> label}. Most commonly this would be the value from
#         Vocabulary.get_index_to_token_vocabulary()
#
#     Returns
#     -------
#     ``List[Tuple[int, int]]``
#         The allowed transitions (from_label_id, to_label_id).
#     """
#     num_labels = len(labels)
#     start_tag = num_labels
#     end_tag = num_labels + 1
#     labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]
#
#     num_total_labels = len(labels_with_boundaries)
#     total_labels = [v for k, v in labels_with_boundaries]
#
#     allowed_matrix = pd.DataFrame(
#         data=np.zeros([num_total_labels, num_total_labels]),
#         index=total_labels,
#         columns=total_labels,
#         dtype=np.bool
#     )
#
#     allowed = []
#     for from_label_index, from_label in labels_with_boundaries:
#         from_tag, from_entity = extract_tag_entity(from_label)
#
#         for to_label_index, to_label in labels_with_boundaries:
#             to_tag, to_entity = extract_tag_entity(to_label)
#
#             if is_transition_allowed(constraint_type, from_tag, from_entity,
#                                      to_tag, to_entity):
#                 allowed.append((from_label_index, to_label_index))
#                 allowed_matrix.at[from_label, to_label] = True
#     return allowed, allowed_matrix


def allowed_transitions(constraint_type: str, labels: Dict[int, str], tag_set: Dict[int, str] = None) -> (List[Tuple[int, int]], pd.DataFrame):
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : ``Dict[int, str]``, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    ``List[Tuple[int, int]]``
        The allowed transitions (from_label_id, to_label_id).
    """
    if not tag_set:
        tag_set = labels

    num_labels = len(tag_set)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    tag_set_with_boundaries = list(tag_set.items()) + [(start_tag, "START"), (end_tag, "END")]

    num_total_labels = len(tag_set_with_boundaries)
    total_labels = [v for k, v in tag_set_with_boundaries]

    allowed_matrix = pd.DataFrame(
        data=np.zeros([num_total_labels, num_total_labels]),
        index=total_labels,
        columns=total_labels,
        dtype=np.bool
    )

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        from_tag, from_entity = extract_tag_entity(from_label)

        for to_label_index, to_label in labels_with_boundaries:
            to_tag, to_entity = extract_tag_entity(to_label)

            if is_transition_allowed(constraint_type, from_tag, from_entity,
                                     to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
                allowed_matrix.iloc[from_label_index, to_label_index] = True
    return allowed, allowed_matrix


def is_transition_allowed(constraint_type: str,
                          from_tag: str,
                          from_entity: str,
                          to_tag: str,
                          to_entity: str):
    """
    Given a constraint type and strings ``from_tag`` and ``to_tag`` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : ``str``, required
        The tag that the transition originates from. For example, if the
        label is ``I-PER``, the ``from_tag`` is ``I``.
    from_entity: ``str``, required
        The entity corresponding to the ``from_tag``. For example, if the
        label is ``I-PER``, the ``from_entity`` is ``PER``.
    to_tag : ``str``, required
        The tag that the transition leads to. For example, if the
        label is ``I-PER``, the ``to_tag`` is ``I``.
    to_entity: ``str``, required
        The entity corresponding to the ``to_tag``. For example, if the
        label is ``I-PER``, the ``to_entity`` is ``PER``.

    Returns
    -------
    ``bool``
        Whether the transition is allowed under the given ``constraint_type``.
    """
    # pylint: disable=too-many-return-statements
    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ('O', 'B', 'U')
        if to_tag == "END":
            return from_tag in ('O', 'L', 'U')
        return any([
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ('O', 'L', 'U') and to_tag in ('O', 'B', 'U'),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ('B', 'I') and to_tag in ('I', 'L') and from_entity == to_entity
        ])
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ('O', 'B')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
                # Can always transition to O or B-x
                to_tag in ('O', 'B'),
                # Can only transition to I-x from B-x or I-x
                to_tag == 'I' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ('O', 'I')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
                # Can always transition to O or I-x
                to_tag in ('O', 'I'),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == 'B' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ('B', 'S')
        if to_tag == "END":
            return from_tag in ('E', 'S')
        return any([
                # Can only transition to B or S from E or S.
                to_tag in ('B', 'S') and from_tag in ('E', 'S'),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == 'M' and from_tag in ('B', 'M') and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == 'E' and from_tag in ('B', 'M') and from_entity == to_entity,
        ])
    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")