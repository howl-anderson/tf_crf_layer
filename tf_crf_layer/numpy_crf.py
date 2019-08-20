import itertools
import math

import numpy as np


class CRF(object):
    def __init__(self, units, transitions, transitions_from_start, transitions_to_end, transition_constraint_mask=None, dynamic_transition_constraint=None):
        self.units = units
        self.transitions = transitions
        self.transitions_from_start = transitions_from_start
        self.transitions_to_end = transitions_to_end
        self.transition_constraint_mask = transition_constraint_mask
        self.dynamic_transition_constraint = dynamic_transition_constraint

    def score(self, logits, tags, transitions_from_start=None, transitions_to_end=None, transitions=None):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        if transitions_from_start is None:
            transitions_from_start = self.transitions_from_start

        if transitions_to_end is None:
            transitions_to_end = self.transitions_to_end

        if transitions is None:
            transitions = self.transitions

        # Start with transitions from START and to END
        total = transitions_from_start[tags[0]] + transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    def compute_log_likelihood_without_mask(self, logits, tags):
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

    def compute_effective_boundary(self):
        """

        :return: Tensor(shape=(B, 1, n)) or Tensor(shape=(1, 1, n))
        """
        left_boundary = self.transitions_from_start
        right_boundary = self.transitions_to_end

        if self.transition_constraint_mask is not None:
            start_tag = self.units
            end_tag = self.units + 1

            left_boundary = (
                    self.transitions_from_start * self.transition_constraint_mask[start_tag, :self.units] +
                    -10000.0 * (1 - self.transition_constraint_mask[start_tag, :self.units])
            )
            right_boundary = (
                    self.transitions_to_end * self.transition_constraint_mask[self.units, end_tag] +
                    -10000.0 * (1 - self.transition_constraint_mask[self.units, end_tag])
            )

        if self.dynamic_transition_constraint is not None:
            start_tag = self.units
            end_tag = self.units + 1

            # shape: (B, 1, n)
            left_dynamic_constraint = self.dynamic_transition_constraint[:, start_tag, :self.units]

            # shape: (B, 1, n)
            left_boundary = (
                self.transitions_from_start * left_dynamic_constraint +
                -10000.0 * (1 - left_dynamic_constraint)
            )

            # shape: (B, 1, n)
            right_dynamic_constraint = self.dynamic_transition_constraint[:, :self.units, end_tag]

            # shape: (B, 1, n)
            right_boundary = (
                self.transitions_to_end * right_dynamic_constraint +
                -10000.0 * (1 - right_dynamic_constraint)
            )

        return left_boundary, right_boundary

    def compute_effective_chain_kernel(self):
        """

        :return: Tensor(shape=(B, n, n)) or Tensor(shape=(n, n))
        """
        chain_kernel = self.transitions

        if self.transition_constraint_mask is not None:
            chain_kernel_mask = self.transition_constraint_mask[:self.units, :self.units]
            chain_kernel = self.transitions * chain_kernel_mask + -10000.0 * (1 - chain_kernel_mask)

        if self.dynamic_transition_constraint is not None:
            dynamic_chain_kernel_mask = self.dynamic_transition_constraint[:, :self.units, :self.units]
            chain_kernel = self.transitions * dynamic_chain_kernel_mask + -10000.0 * (1 - dynamic_chain_kernel_mask)

        return chain_kernel

    def viterbi_decode(self, logits, mask=None):
        if mask is None:
            mask = np.ones(np.shape(logits)[:2], dtype=np.bool)

        transitions_from_start, transitions_to_end = self.compute_effective_boundary()
        transitions = self.compute_effective_chain_kernel()

        most_likely_tags = []
        best_scores = []

        for logit, mas, from_start, to_end, transition in zip(logits, mask, transitions_from_start, transitions_to_end, transitions):
            sequence_length = np.sum(mas)
            most_likely, most_likelihood = None, -float('inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit, tags, from_start, to_end, transition)
                if score > most_likelihood:
                    # padding tags to sequence length
                    tag_len_diff = 3 - len(tags)
                    if tag_len_diff:
                        tags = list(tags) + [0] * tag_len_diff
                        
                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)

        return most_likely_tags
