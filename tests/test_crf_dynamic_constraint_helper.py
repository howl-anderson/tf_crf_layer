import numpy as np

from tf_crf_layer.crf_dynamic_constraint_helper import generate_constraint_table


def test_generate_constraint_table():
    constraint_mapping = [
        ["X"],  # domain #1 only contain entity X
        ["Y"]   # domain #2 only contain entity Y
    ]

    tag_dict = dict(enumerate(["O", "B-X", "I-X", "L-X", "U-X", "B-Y", "I-Y", "L-Y", "U-Y"]))

    actual = generate_constraint_table(constraint_mapping, tag_dict, tag_scheme='BIOUL')

    expected = np.array([
        [
        # only contain allows for entity X
            #     O     B-X    I-X    L-X    U-X    B-Y    I-Y    L-Y    U-Y  start   end
            [     1,     1,     0,     0,     1,     0,     0,     0,     0,    0,     1],  # O
            [     0,     0,     1,     1,     0,     0,     0,     0,     0,    0,     0],  # B-X
            [     0,     0,     1,     1,     0,     0,     0,     0,     0,    0,     0],  # I-X
            [     1,     1,     0,     0,     1,     0,     0,     0,     0,    0,     1],  # L-X
            [     1,     1,     0,     0,     1,     0,     0,     0,     0,    0,     1],  # U-X
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # B-Y
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # I-Y
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # L-Y
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # U-Y
            [     1,     1,     0,     0,     1,     0,     0,     0,     0,    0,     0],  # start
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # end
        ],
        [
        # only contain allows for entity Y
            #     O     B-X    I-X    L-X    U-X    B-Y    I-Y    L-Y    U-Y  start   end
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     1],  # O
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # B-X
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # I-X
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # L-X
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # U-X
            [     0,     0,     0,     0,     0,     0,     1,     1,     0,    0,     0],  # B-Y
            [     0,     0,     0,     0,     0,     0,     1,     1,     0,    0,     0],  # I-Y
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     1],  # L-Y
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     1],  # U-Y
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     0],  # start
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # end
        ]
    ], dtype=np.bool)

    diff = np.bitwise_xor(actual, expected)

    np.testing.assert_equal(actual, expected)


def test_generate_constraint_table_for_multiple_entity_each_domain():
    constraint_mapping = [
        ["X", "Y"],  # domain #1 contain entity X and Y
        ["Y"]        # domain #2 only contain entity Y
    ]

    tag_dict = dict(enumerate(["O", "B-X", "I-X", "L-X", "U-X", "B-Y", "I-Y", "L-Y", "U-Y"]))

    actual = generate_constraint_table(constraint_mapping, tag_dict, tag_scheme='BIOUL')

    expected = np.array([
        [
        # only contain allows for entity X and Y
            #     O     B-X    I-X    L-X    U-X    B-Y    I-Y    L-Y    U-Y  start   end
            [     1,     1,     0,     0,     1,     1,     0,     0,     1,    0,     1],  # O
            [     0,     0,     1,     1,     0,     0,     0,     0,     0,    0,     0],  # B-X
            [     0,     0,     1,     1,     0,     0,     0,     0,     0,    0,     0],  # I-X
            [     1,     1,     0,     0,     1,     1,     0,     0,     1,    0,     1],  # L-X
            [     1,     1,     0,     0,     1,     1,     0,     0,     1,    0,     1],  # U-X
            [     0,     0,     0,     0,     0,     0,     1,     1,     0,    0,     0],  # B-Y
            [     0,     0,     0,     0,     0,     0,     1,     1,     0,    0,     0],  # I-Y
            [     1,     1,     0,     0,     1,     1,     0,     0,     1,    0,     1],  # L-Y
            [     1,     1,     0,     0,     1,     1,     0,     0,     1,    0,     1],  # U-Y
            [     1,     1,     0,     0,     1,     1,     0,     0,     1,    0,     0],  # start
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # end
        ],
        [
        # only contain allows for entity Y
            #     O     B-X    I-X    L-X    U-X    B-Y    I-Y    L-Y    U-Y  start   end
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     1],  # O
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # B-X
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # I-X
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # L-X
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # U-X
            [     0,     0,     0,     0,     0,     0,     1,     1,     0,    0,     0],  # B-Y
            [     0,     0,     0,     0,     0,     0,     1,     1,     0,    0,     0],  # I-Y
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     1],  # L-Y
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     1],  # U-Y
            [     1,     0,     0,     0,     0,     1,     0,     0,     1,    0,     0],  # start
            [     0,     0,     0,     0,     0,     0,     0,     0,     0,    0,     0],  # end
        ]
    ], dtype=np.bool)

    diff = np.bitwise_xor(actual, expected)

    np.testing.assert_equal(actual, expected)
