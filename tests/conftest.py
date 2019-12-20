import itertools
import math
from typing import Tuple, List

import numpy as np
import pytest


def _get_random_data(nb_samples, timesteps, x_low=1, x_high=12, y_low=0, y_high=5):
    x = np.random.randint(low=x_low, high=x_high,
                          size=nb_samples * timesteps)
    x = x.reshape((nb_samples, timesteps))
    # x[0, -4:] = 0  # right padding
    # x[1, :5] = 0  # left padding, currently not supported by crf layer

    y = np.random.randint(low=y_low, high=y_high, size=nb_samples * timesteps)
    y = y.reshape((nb_samples, timesteps))

    return x, y


@pytest.fixture
def get_random_data():
    return _get_random_data


class NumpyCRF:
    def __init__(self, logits, mask, transitions, transitions_from_start, transitions_to_end):
        self.transitions_from_start = transitions_from_start
        self.transitions_to_end = transitions_to_end
        self.transitions = transitions
        self.logits = logits
        self.mask = mask if mask is not None else np.ones_like(self.logits)

    def compute_log_likehood(self, tags) -> float:
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i in zip(self.logits, tags):
            numerator = self._score(logits_i, tags_i)
            all_scores = [self._score(logits_i, tags_j)
                          for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        return manual_log_likelihood

    def decode(self) -> Tuple[List[any], List[any]]:
        # We can also iterate over all possible tag sequences and use self.score
        # to check the likelihood of each. The most likely sequence should be the
        # same as what we get from viterbi_tags.
        most_likely_tags = []
        best_scores = []

        for logit, mas in zip(self.logits, self.mask):
            sequence_length = np.sum(mas)
            most_likely, most_likelihood = None, -float('inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self._score(logit, tags)
                if score > most_likelihood:
                    # padding tags to sequence length
                    tag_len_diff = 3 - len(tags)
                    if tag_len_diff:
                        tags = list(tags) + [0] * tag_len_diff

                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)

        return most_likely_tags, best_scores

    def _score(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total


@pytest.fixture
def numpy_crf():
    return NumpyCRF
