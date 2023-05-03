# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function
import pdb

import numpy as np
import tensorflow as tf

from basenji import dna_io


class PredStreamGen:
    """ Interface to acquire predictions via a buffered stream mechanism
          rather than getting them all at once and using excessive memory.
          Accepts generator and constructs stream batches from it. """

    def __init__(self, model, seqs_gen, batch_size, stream_seqs=64, verbose=False):
        self.model = model
        self.seqs_gen = seqs_gen
        self.stream_seqs = stream_seqs
        self.batch_size = batch_size
        self.verbose = verbose

        self.stream_start = 0
        self.stream_end = 0

    def __getitem__(self, i):
        # acquire predictions, if needed
        if i >= self.stream_end:
            # update start
            self.stream_start = self.stream_end

            if self.verbose:
                print('Predicting from %d' % self.stream_start, flush=True)

            # predict
            self.stream_preds = self.model.predict(self.make_dataset())

            # update end
            self.stream_end = self.stream_start + self.stream_preds.shape[0]

        return self.stream_preds[i - self.stream_start]

    def make_dataset(self):
        """ Construct Dataset object for this stream chunk. """
        seqs_1hot = []
        stream_end = self.stream_start + self.stream_seqs
        for si in range(self.stream_start, stream_end):
            try:
                seqs_1hot.append(self.seqs_gen.__next__())
            except StopIteration:
                continue

        seqs_1hot = np.array(seqs_1hot)

        dataset = tf.data.Dataset.from_tensor_slices((seqs_1hot,))
        dataset = dataset.batch(self.batch_size)
        return dataset


class PredStreamIter:
    """ Interface to acquire predictions via a buffered stream mechanism
          rather than getting them all at once and using excessive memory.
          Accepts iterator and constructs stream batches from it.
          [I don't recall whether I've ever gotten this one working."""

    def __init__(self, model, dataset_iter, stream_seqs=128, verbose=False):
        self.model = model
        self.dataset_iter = dataset_iter
        self.stream_seqs = stream_seqs
        self.verbose = verbose

        self.stream_start = 0
        self.stream_end = 0

    def __getitem__(self, i):
        # acquire predictions, if needed
        if i >= self.stream_end:
            # update start
            self.stream_start = self.stream_end

            if self.verbose:
                print('Predicting from %d' % self.stream_start, flush=True)

            # predict
            self.stream_preds = self.model.predict(self.fetch_batch())

            # update end
            self.stream_end = self.stream_start + self.stream_preds.shape[0]

        return self.stream_preds[i - self.stream_start]

    def fetch_batch(self):
        """Fetch a batch of data from the dataset iterator."""
        x = [next(self.dataset_iter)]
        while x[-1] and len(x) < self.stream_seqs:
            x.append(next(self.dataset_iter))
        return x
