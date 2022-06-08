# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:28:36 2022

@author: Zhenzi Weng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np

class DeepSpeechDecoder(object):
    def __init__(self, labels, blank_index=28):
        self.labels = labels
        self.blank_index = blank_index
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        
    def convert_to_string(self, sequence):
        return "".join([self.int_to_char[i] for i in sequence])
        
    def decode(self, logits):
        # choose the class with maximimum probability
        best = list(np.argmax(logits, axis=1))
        # merge repeated chars
        merge = [k for k, _ in itertools.groupby(best)]
        # remove the blank index in the decoded sequence
        merge_remove_blank = []
        for k in merge:
            if k != self.blank_index:
                merge_remove_blank.append(k)
                
        return self.convert_to_string(merge_remove_blank)

