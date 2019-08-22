#!/usr/bin/env python

from tokenizer_tools.conllz.tag_collector import collect_label_to_file


collect_label_to_file(['data/train.conllx'], 'data/label.txt')
