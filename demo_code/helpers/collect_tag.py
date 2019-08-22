#!/usr/bin/env python

from tokenizer_tools.conllz.tag_collector import collect_entity_to_file


collect_entity_to_file(['data/train.conllx'], 'data/entity.txt')
