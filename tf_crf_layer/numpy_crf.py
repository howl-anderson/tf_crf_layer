import itertools
import math

import numpy as np


class CRF(object):
    def __init__(self, transitions, transitions_from_start, transitions_to_end):
        self.transitions = transitions
        self.transitions_from_start = transitions_from_start
        self.transitions_to_end = transitions_to_end

    def score(self, logits, tags):
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

    def compute_log_likelihood_without_mask(logits, tags):
        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i in zip(logits, tags):
            numerator = self.score(logits_i, tags_i)
            all_scores = [self.score(logits_i, tags_j)
                          for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        return manual_log_likelihood

    def compute_log_likelihood_with_mask(self, logits, tags, mask):
        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        #   (which is just the score for the logits and actual tags)
        # and the denominator
        #   (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i, mask_i in zip(logits, tags, mask):
            # Find the sequence length for this input and only look at that much of each sequence.
            sequence_length = np.sum(mask_i)
            logits_i = logits_i[:sequence_length]
            tags_i = tags_i[:sequence_length]

            numerator = self.score(logits_i, tags_i)
            all_scores = [self.score(logits_i, tags_j)
                          for tags_j in itertools.product(range(5), repeat=sequence_length)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        return manual_log_likelihood

    def viterbi_decode(self, logits, mask):
        most_likely_tags = []
        best_scores = []

        for logit, mas in zip(logits, mask):
            sequence_length = np.sum(mas.detach())
            most_likely, most_likelihood = None, -float('inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit.data, tags)
                if score > most_likelihood:
                    # padding tags to sequence length
                    tag_len_diff = 3 - len(tags)
                    if tag_len_diff:
                        tags = list(tags) + [0] * tag_len_diff
                        
                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)
