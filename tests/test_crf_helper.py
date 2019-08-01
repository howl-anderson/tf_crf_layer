from pytest import raises

from tf_crf_layer.crf_helper import allowed_transitions
from tf_crf_layer.exceptions import ConfigurationError


def test_allowed_transitions():
    # pylint: disable=bad-whitespace,bad-continuation
    bio_labels = ['O', 'B-X', 'I-X', 'B-Y', 'I-Y']  # start tag, end tag
    #              0     1      2      3      4         5          6
    allowed = allowed_transitions("BIO", dict(enumerate(bio_labels)))

    # The empty spaces in this matrix indicate disallowed transitions.
    assert set(allowed) == {                         # Extra column for end tag.
        (0, 0), (0, 1),         (0, 3),              (0, 6),
        (1, 0), (1, 1), (1, 2), (1, 3),              (1, 6),
        (2, 0), (2, 1), (2, 2), (2, 3),              (2, 6),
        (3, 0), (3, 1),         (3, 3), (3, 4),      (3, 6),
        (4, 0), (4, 1),         (4, 3), (4, 4),      (4, 6),
        (5, 0), (5, 1),         (5, 3)                      # Extra row for start tag
    }

    bioul_labels = ['O', 'B-X', 'I-X', 'L-X', 'U-X', 'B-Y', 'I-Y', 'L-Y', 'U-Y']  # start tag, end tag
    #                0     1      2      3      4      5      6      7      8          9        10
    allowed = allowed_transitions("BIOUL", dict(enumerate(bioul_labels)))

    # The empty spaces in this matrix indicate disallowed transitions.
    assert set(allowed) == {                                                   # Extra column for end tag.
        (0, 0), (0, 1),                 (0, 4), (0, 5),                 (0, 8),       (0, 10),
                        (1, 2), (1, 3),
                        (2, 2), (2, 3),
        (3, 0), (3, 1),                 (3, 4), (3, 5),                 (3, 8),       (3, 10),
        (4, 0), (4, 1),                 (4, 4), (4, 5),                 (4, 8),       (4, 10),
                                                        (5, 6), (5, 7),
                                                        (6, 6), (6, 7),
        (7, 0), (7, 1),                 (7, 4), (7, 5),                 (7, 8),       (7, 10),
        (8, 0), (8, 1),                 (8, 4), (8, 5),                 (8, 8),       (8, 10),
        # Extra row for start tag.
        (9, 0), (9, 1),                 (9, 4), (9, 5),                 (9, 8)
    }

    iob1_labels = ['O', 'B-X', 'I-X', 'B-Y', 'I-Y']  # start tag, end tag
    #              0     1      2      3      4         5          6
    allowed = allowed_transitions("IOB1", dict(enumerate(iob1_labels)))

    # The empty spaces in this matrix indicate disallowed transitions.
    assert set(allowed) == {                            # Extra column for end tag.
        (0, 0),         (0, 2),         (0, 4),         (0, 6),
        (1, 0), (1, 1), (1, 2),         (1, 4),         (1, 6),
        (2, 0), (2, 1), (2, 2),         (2, 4),         (2, 6),
        (3, 0),         (3, 2), (3, 3), (3, 4),         (3, 6),
        (4, 0),         (4, 2), (4, 3), (4, 4),         (4, 6),
        (5, 0),         (5, 2),         (5, 4),                # Extra row for start tag
    }
    with raises(ConfigurationError):
        allowed_transitions("allennlp", {})

    bmes_labels = ['B-X', 'M-X', 'E-X', 'S-X', 'B-Y', 'M-Y', 'E-Y', 'S-Y']  # start tag, end tag
    #               0      1      2      3      4      5      6      7       8          9
    allowed = allowed_transitions("BMES", dict(enumerate(bmes_labels)))
    assert set(allowed) == {
                (0, 1), (0, 2),
                (1, 1), (1, 2),                                         # Extra column for end tag.
        (2, 0),                 (2, 3), (2, 4),                 (2, 7), (2, 9),
        (3, 0),                 (3, 3), (3, 4),                 (3, 7), (3, 9),
                                                (4, 5), (4, 6),
                                                (5, 5), (5, 6),
        (6, 0),                 (6, 3), (6, 4),                 (6, 7), (6, 9),
        (7, 0),                 (7, 3), (7, 4),                 (7, 7), (7, 9),
        (8, 0),                 (8, 3), (8, 4),                 (8, 7),  # Extra row for start tag
    }
