# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import torch.nn.functional as F

from fairseq.models import FairseqEncoder
from fairseq.modules import LayerNorm


class CTCDecoder(FairseqEncoder):
    def __init__(self, dictionary, in_dim):
        super().__init__(dictionary)
        self.proj = nn.Linear(in_dim, len(dictionary))
        
        self.linear_adapter = nn.Sequential(
               LayerNorm(in_dim),
               nn.Linear(in_dim, 2 * in_dim),
               nn.ReLU(),
               nn.Linear(2 * in_dim, in_dim),
           )
        
        if False:
            self.out_ln = LayerNorm(in_dim)
        else:
            self.out_ln = None
        
        self.distribution_temperature = 1.0
        self.output_projection = None

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        encoder_out = self.proj(src_tokens)
        return {"encoder_out": encoder_out}
