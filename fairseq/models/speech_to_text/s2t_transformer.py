# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import numpy as np
import copy

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.speech_to_text.modules.convolution import (
    Conv1dSubsampler,
    Conv2dSubsampler,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    MultiheadAttention,
)

logger = logging.getLogger(__name__)


@register_model("s2t_transformer")
class S2TTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/s2t"
        model_ids = [
            "s2t_transformer_s-en-asr-librispeech",
            "s2t_transformer_m-en-asr-librispeech",
            "s2t_transformer_l-en-asr-librispeech",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            **kwargs,
        )
        return S2THubInterface(x["args"], x["task"], x["models"][0])

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="STR",
            help="kernel sizes of Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv2d (convtransformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-version",
            type=str,
            default="s2t_transformer",
            choices=["s2t_transformer", "convtransformer"],
            help="version of frontend convolutional layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        parser.add_argument(
            "--use-ctc-shrink",
            type=bool,
            default=False,
            help="using ctc prediction to shrink sequence"
        )
        parser.add_argument(
            "--use-prefix-token",
            type=bool,
            default=False,
            help="using prefix token"
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TTransformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        args.tgt_dict_size = len(task.target_dictionary)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def get_ctc_target(self, sample: Optional[Dict[str, Tensor]]):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        sample: Optional[Dict[str, Tensor]],
    ):
        encoder_out = net_output[1]["encoder_out"]["encoder_out"][0]
        logits = self.encoder.ctc_proj(encoder_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = net_output[1]["encoder_out"]["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out

class LookBackModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder_attn = MultiheadAttention(
            cfg.encoder_embed_dim,
            cfg.encoder_attention_heads,
            kdim=cfg.encoder_embed_dim,
            vdim=cfg.encoder_embed_dim,
            dropout=cfg.dropout,
            encoder_decoder_attention=True
        )
        self.atten_layer_norm = LayerNorm(cfg.encoder_embed_dim)
        self.fc1 = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_embed_dim, cfg.encoder_embed_dim)
        self.activation_fn = nn.SiLU() #utils.get_activation_fn(activation="swish")
        self.ffn_layer_norm = LayerNorm(cfg.encoder_embed_dim)
        self.lb_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, wav_feature, bf_shrink_padding_mask):

        residual = x
        x, _ = self.encoder_attn(
            query=x,
            key=wav_feature,
            value=wav_feature,
            key_padding_mask=bf_shrink_padding_mask,
            incremental_state=None,
            static_kv=True,
            need_weights=False,
            #attn_mask=padding_mask,
        )
        x += residual
        x = self.lb_dropout(x)
        x = self.atten_layer_norm(x)
        residual = x
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x += residual
        x = self.lb_dropout(x)
        x = self.ffn_layer_norm(x)
        return x


class S2TTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.conv_version = args.conv_version
        if self.conv_version == "s2t_transformer":
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        elif self.conv_version == "convtransformer":
            self.subsample = Conv2dSubsampler(
                args.input_channels,
                args.input_feat_per_channel,
                args.conv_out_channels,
                args.encoder_embed_dim,
            )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.ctc_proj = None
        if getattr(args, "ctc_weight", 0.0) > 0.0:
            self.ctc_proj = nn.Linear(args.encoder_embed_dim, args.tgt_dict_size)
        self.use_ctc_shrink = getattr(args, "use_ctc_shrink", False)
        if self.use_ctc_shrink:
           self.shrink_layer_norm = LayerNorm(args.encoder_embed_dim)
           self.lbm = None
           # self.lbm = LookBackModule(args)
           print("LookBackModule:", self.lbm)
        self.use_prefix_token = getattr(args, "use_prefix_token", False)
        if self.use_prefix_token:
            self.pre_tokens = nn.Parameter(torch.randn(2, args.encoder_embed_dim))
            # self.pre_tokens.requires_grad = False         # ft
            self.add_prefix_layer = getattr(args, "add_prefix_layer", "999,999")
            self.add_prefix_layer = [ int(i)-1 for i in self.add_prefix_layer.split(',', 1)]
            print(f"Using prefixe tokens nn.Parameter {self.pre_tokens.size()} BEHIND Layer:{self.add_prefix_layer} (not influence task on current layer), if update: {self.pre_tokens.requires_grad}")
            self.both_activate = False
        else:
            self.pre_tokens = None

    
    def cal_inner_ctc(self, x, src_lengths, encoder_padding_mask, layer):
        
        # ctc_model
        ctc_model = self.inner_ctc[self.inner_ctc_layer[layer]][0]

        # make input data
        non_padding_mask = ~encoder_padding_mask
        input_lengths = non_padding_mask.long().sum(-1)
        if len(encoder_padding_mask) <= 0:
            assert False, encoder_padding_mask
    
        seq_len, bsz, _ = x.size()
        probs = ctc_model(x, input_lengths)["encoder_out"]
        
        vocab_size = probs.size(-1)
        if ctc_model.training:
            distribution = F.gumbel_softmax(probs, tau=ctc_model.encoder.distribution_temperature, hard=False,)
        else:
            # assert False, "ctc model is not training!!!"
            distribution = F.softmax(probs / ctc_model.encoder.distribution_temperature, dim=-1)
            
            # for name, param in ctc_model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            
        distribution_2d = distribution.contiguous().view(-1, vocab_size)
        
        soft_out = torch.mm(distribution_2d, ctc_model.encoder.output_projection.weight).view(seq_len, bsz, -1)
        linear_out = ctc_model.encoder.linear_adapter(x)
        # assert soft_out.size() == linear_out.size()
        x = linear_out + soft_out
        
        if ctc_model.encoder.out_ln:
            print("[Inner CTC] do layer post norm")
            ctc_model.encoder.out_ln(x)
        
        return x

    def cal_ctc_shrink(self, x, src_lengths, padding_mask, layer):
        ctc_model = self.inner_ctc[self.inner_ctc_layer[layer]][0]

        # make input data
        non_padding_mask = ~padding_mask
        input_lengths = non_padding_mask.long().sum(-1)
        if len(padding_mask) <= 0:
            assert False, padding_mask
        seq_len, bsz, _ = x.size()
        probs = ctc_model(x, input_lengths)["encoder_out"]

        vocab_size = probs.size(-1)
        if ctc_model.training:
            distribution = F.gumbel_softmax(probs, tau=ctc_model.encoder.distribution_temperature, hard=False,)
        else:
            # assert False, "ctc model is not training!!!"
            distribution = F.softmax(probs / ctc_model.encoder.distribution_temperature, dim=-1)


        tokens = distribution.argmax(dim=-1).transpose(0, 1)
        shrink_mask = tokens.roll(1) != tokens
        shrink_mask[:,0] = True
        shrink_mask = shrink_mask & (~padding_mask)
        lengths = shrink_mask.long().sum(-1)

        max_len = lengths.max()
        shrink_2d = x.transpose(0,1)[shrink_mask]
        s_x = shrink_2d.new_zeros(x.size(1), max_len, x.size(-1))
        #s_x = shrink_2d.new_ones(wav_feature.size(0), max_len, wav_feature.size(-1))
        l_index = 0
        for i, v in enumerate(lengths):
            s_x[i, :v] = shrink_2d[l_index:l_index+v]
            l_index += v
        new_padding_mask = lengths_to_padding_mask(lengths)

        s_x = self.shrink_layer_norm(s_x.transpose(0, 1))
        if self.lbm:
            s_x = self.lbm(s_x, x, padding_mask)

        return s_x, new_padding_mask

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        # save_path="/workspace/data/zhangyuhao/fairseq-S2ST/egs/speech_to_unit_translation/encoder_tmp/fren/double-wo_ft"
        encoder_states = []
        # attention_in = x.detach()
        # attention_mask = encoder_padding_mask.detach()
        old_padding_mask = encoder_padding_mask
        
        has_prefix = False
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, encoder_padding_mask)
            # x, attn_weight = layer(x, encoder_padding_mask)
            # with open(f"{save_path}/encoder_states_{i+1}_attn_weight","a") as write_file:
            #     for weight in attn_weight:
            #         np.savetxt(write_file, weight.cpu(), fmt="%.8f",newline=" ",footer="\n",comments="")
            
            if return_all_hiddens:
                encoder_states.append(x)
                
            # inner CTC
            # x = FFN(Norm(x)) + distribution * Embeddings
            if i in self.inner_ctc_layer:
                x = self.cal_inner_ctc(x, src_lengths, encoder_padding_mask, i)
                if self.use_ctc_shrink:
                    x, encoder_padding_mask = self.cal_ctc_shrink(x, src_lengths, encoder_padding_mask, i)

            # prefix
            if self.use_prefix_token and self.pre_tokens is not None:
                if i in self.add_prefix_layer:
                    seq_len, bsz, dim = x.size()
                    if i == self.add_prefix_layer[0]:
                        prefix = self.pre_tokens[0].repeat(bsz).reshape(1, bsz, dim)
                        has_prefix = True
                    elif i == self.add_prefix_layer[1]:
                        prefix = self.pre_tokens[1].repeat(bsz).reshape(1, bsz, dim)
                    else:
                        assert False, "Error prefix_token layer set"
                    if has_prefix:
                        x[0] = prefix
                        self.both_activate = True
                    else:
                        x = torch.cat((prefix, x), dim=0)
                        encoder_padding_mask = lengths_to_padding_mask(input_lengths + 1)

            # if i == 5 or True:
            #     if i+1 < len(self.transformer_layers):
            #         old_x = copy.deepcopy(x)
            #         encoder_layer_states = self.transformer_layers[i+1].self_attn_layer_norm(x).transpose(0,1).cpu()
            #         for feature in encoder_layer_states:
            #             with open(f"{save_path}/encoder_states_{i+1}_norm","a") as write_file:
            #                 np.savetxt(write_file, feature, fmt="%.8f",newline=" ",footer="\n",comments="")
            #         assert (old_x == x).all()
            
        if self.layer_norm is not None:
            x = self.layer_norm(x)

            # encoder_layer_states = x.transpose(0,1).cpu()
            # for feature in encoder_layer_states:
            #     with open(f"{save_path}/encoder_states_12_norm","a") as write_file:
            #         np.savetxt(write_file, feature, fmt="%.8f",newline=" ",footer="\n",comments="")
        

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "old_padding_mask": [old_padding_mask],
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            # "attention_in": attention_in,
            # "attention_mask": attention_mask,
        }

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        extra = {"encoder_out": encoder_out} if incremental_state is None else None
        return x, extra


@register_model_architecture(model_name="s2t_transformer", arch_name="s2t_transformer")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.input_channels = getattr(args, "input_channels", 1)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")  # for Conv1d
    args.conv_channels = getattr(args, "conv_channels", 1024)  # for Conv1d
    args.conv_out_channels = getattr(args, "conv_out_channels", 256)  # for Conv2d
    args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("s2t_transformer", "s2t_transformer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_xs")
def s2t_transformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_sp")
def s2t_transformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_m")
def s2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_mp")
def s2t_transformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_m(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_l")
def s2t_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_lp")
def s2t_transformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_l(args)
    