########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import math
import numpy as np
import torch.nn as nn

import functools
import os, math, gc, importlib
import torch

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


from ijepa.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)

from src.dataset import process_sequence_array
from model_mjepa import Encoder, Predictor, apply_masks
#
def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: tensor containing indices of patches to keep for each batch, shape [B, M] where M is the number of patches to keep
    """
    B, N, D = x.shape  # Batch size, Number of patches, Feature dimension
    M = masks.size(1)  # Number of patches to keep

    # Initialize a tensor to store the selected patches
    selected_x = torch.empty(B, M, D, dtype=x.dtype, device=x.device)

    # Apply the mask to each item in the batch individually
    for i in range(B):
        mask_keep = masks[i].unsqueeze(-1).expand(-1, D)  # Expand mask for feature dimension
        selected_x[i] = torch.gather(x[i], dim=0, index=mask_keep)

    return selected_x
# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

LORA_CONFIG = {
    "r": 0,
    "alpha": 0,
    "dropout": 0,
    "parts": {"att", "ln", "time"},
}


try:
    print("RWKV_MY_TESTING", os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ""


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(
        name=f"wkv_{T_MAX}_bf16",
        sources=[
            "cuda/wkv_op_bf16.cpp",
            "cuda/wkv_cuda_bf16.cu",
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-t 4",
            "-std=c++17",
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-DTmax={T_MAX}",
        ],
    )

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y = torch.empty(
                (B, T, C),
                device=w.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            return y

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty(
                (B, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            gu = torch.empty(
                (B, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                memory_format=torch.contiguous_format,
                dtype=torch.bfloat16,
            )
            wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

else:
    wkv_cuda = load(
        name=f"wkv_{T_MAX}",
        sources=[
            "cuda/wkv_op.cpp",
            "cuda/wkv_cuda.cu",
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-DTmax={T_MAX}",
        ],
    )

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            # print('BC!!!!',B,C)
            assert B * C % min(C, 32) == 0
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            y = torch.empty(
                (B, T, C), device=w.device, memory_format=torch.contiguous_format
            )
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return y
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return y.half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return y.bfloat16()

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty(
                (B, C), device=gy.device, memory_format=torch.contiguous_format
            )
            gu = torch.empty(
                (B, C), device=gy.device, memory_format=torch.contiguous_format
            )
            gk = torch.empty(
                (B, T, C), device=gy.device, memory_format=torch.contiguous_format
            )
            gv = torch.empty(
                (B, T, C), device=gy.device, memory_format=torch.contiguous_format
            )
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(
                    B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv
                )
            else:
                wkv_cuda.backward(
                    B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv
                )
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (
                    None,
                    None,
                    None,
                    gw.bfloat16(),
                    gu.bfloat16(),
                    gk.bfloat16(),
                    gv.bfloat16(),
                )


def RUN_CUDA(B, T, C, w, u, k, v):
    # print('run cuda!!!', B, T,C)
    return WKV.apply(B, T, C, w, u, k, v)


########################################################################################################
# LoRA
########################################################################################################


class LoraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r, alpha, dropout = (
            LORA_CONFIG["r"],
            LORA_CONFIG["alpha"],
            LORA_CONFIG["dropout"],
        )
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return F.linear(x, self.weight) + self.scaling * F.linear(
            F.linear(self.lora_dropout(x), self.lora_A), self.lora_B
        )


@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    if "att" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    if "ffn" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################
g = torch.Generator()


class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(
                torch.ones(args.dim_att) * math.log(0.3) + zigzag
            )

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
            )
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = make_linear_att(args.n_embd, args.dim_att, bias=False)
        self.value = make_linear_att(args.n_embd, args.dim_att, bias=False)
        self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False)

        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        if "a" in os.environ["RWKV_MY_TESTING"]:
            self.register_buffer(
                "att_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len))
            )
            d_qkv = args.n_embd // 16
            self.qq = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.kk = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.vv = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.oo = nn.Linear(d_qkv, args.n_embd, bias=False)
            with torch.no_grad():
                self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_vv = nn.Parameter(
                    torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                )

    if "a" not in os.environ["RWKV_MY_TESTING"]:

        @MyFunction
        def jit_func(self, x):
            xx = self.time_shift(
                x
            )  # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            return sr, k, v

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v = self.jit_func(x)
            # print('within Timemix',x.shape)
            rwkv = sr * RUN_CUDA(
                B, T, self.args.dim_att, self.time_decay, self.time_first, k, v
            )
            return self.output(rwkv)

    if "a" in os.environ["RWKV_MY_TESTING"]:

        @MyFunction
        def QKV(self, q, k, v):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.att_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            x = att @ v
            return x

        @MyFunction
        def jit_funcQKV(self, x):
            xx = self.time_shift(
                x
            )  # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
            xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
            xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            qq = self.qq(xqq)
            kk = self.kk(xkk)
            vv = self.vv(xvv)
            return sr, k, v, qq, kk, vv

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v, qq, kk, vv = self.jit_funcQKV(x)
            rwkv = sr * RUN_CUDA(
                B, T, self.args.dim_att, self.time_decay, self.time_first, k, v
            )
            rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))
            return rwkv


########################################################################################################


class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False)
        self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(
                    torch.zeros((1, args.my_pos_emb, args.n_embd))
                )
                self.pos_emb_y = nn.Parameter(
                    torch.zeros((args.my_pos_emb, 1, args.n_embd))
                )

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_TimeMix(args, layer_id)

        if "g" in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer(
                "tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len))
            )

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                x = x + pos_emb

        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            # print('within block!!!!',x.shape)
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
#


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.ln_concat = nn.LayerNorm(args.n_embd*2)

        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        if args.load_jepa_path:
            print('Using MJEPA!!!:')
            self.jepa_encoder = Encoder(args)
            self.jepa_predictor = Predictor(args)
            # print(self.jepa_encoder.state_dict())
            checkpoint = torch.load(args.load_jepa_path, map_location=torch.device(args.device))
            pretrained_dict = checkpoint['encoder']
            self.jepa_encoder.load_state_dict(pretrained_dict)
            pretrained_dict = checkpoint['predictor']
            self.jepa_predictor.load_state_dict(pretrained_dict)
            print('sucessfully loading MJEPA from:', args.load_jepa_path)
            # print(self.jepa_encoder.state_dict())

            for model in [self.emb,self.blocks, self.ln_out, self.ln_concat,self.head,self.jepa_encoder, self.jepa_predictor]:
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()

            self.linear1 = nn.Linear(args.embed_dim,32)

            self.linear = nn.Linear(args.n_embd+32,args.n_embd)
        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer(
                "copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len))
            )

    def configure_optimizers(self):
        args = self.args
        if args.layerwise_lr > 0:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif "time_decay" in n:
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            if args.my_pile_stage == 2:
                optim_groups = [
                    {
                        "params": [param_dict[n] for n in lr_1x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 1.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_2x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 2e-3 / args.lr_init},
                    {
                        "params": [param_dict[n] for n in lr_3x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {
                        "params": [param_dict[n] for n in lr_1x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 1.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_2x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 2.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_3x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 3.0,
                    },
                ]
        else:
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters()],
                    "weight_decay": 0.0,
                },
            ]

        for g in optim_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        optim_groups = [g for g in optim_groups if len(g["params"]) > 0]

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adamw_mode=False,
                weight_decay=0,
                amsgrad=False,
            )
        return FusedAdam(
            optim_groups,
            lr=self.args.lr_init,
            betas=self.args.betas,
            eps=self.args.adam_eps,
            bias_correction=True,
            adam_w_mode=False,
            weight_decay=0,
            amsgrad=False,
        )
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False



    def forward(self, idx,patch=None):
        args = self.args
        # print('idx!!!!!!',idx.shape)
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        # print('x shape:',x.shape)
        x_emb = x  # bs length embed

        if patch is not None:
            aa, masks_pred,masks_enc = patch
            with torch.no_grad():  # Ensure no gradients are computed for the fixed parts
                z = self.jepa_encoder(aa, masks_enc)
                z1 = self.jepa_predictor(z, aa, masks_enc, masks_pred)
            # temp=create_feature(idx,x,z1)  #bs length dim
            # temp=temp.to(x.dtype)
            z1 = torch.mean(z1, dim=1, keepdim=True)  # Resulting shape [bs, 1, dim]
            z2 = z1.expand(-1, x.size(1), -1)  # Resulting shape [bs, l2, dim]
            # print('z2 shape:',z2.shape,z2[0,:3])
            # z2= self.linear(z1)
            z2=self.linear1(z2)
            x=torch.concatenate([x,z2],2)
            # print('x shape:',x.shape,x[0,:3])

            # x= self.ln_concat(x)
            # print('x shape after ln:',x.shape,x[0,:3])

            x= self.linear(x)

        # print('x_emb before!!!!!!',x_emb.shape)  bs length embed nn.Embedding(args.vocab_size, args.n_embd)

        if args.tiny_att_dim > 0:
            for block in self.blocks:
                if args.grad_cp == 1:
                    if args.lora:
                        x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
                    else:
                        x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in self.blocks:
                if args.grad_cp == 1:
                    if args.lora:
                        x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
                    else:
                        x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    # print('x_block!!!!!!',x.shape)
                    x = block(x)
        # print('block after!!!!!!',x.shape)

        x = self.ln_out(x)
        # print('lnout after!!!!!!',x.shape)

        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

            x = self.head(x) + c
        else:
            x = self.head(x)   # bs,length,tokens  nn.Linear(args.n_embd, args.vocab_size, bias=False)
        # print('head after!!!!!!',x.shape)

        return x
    def training_step(self, batch, batch_idx):
        # print(f'Batch Index: {batch_idx}, GPU ID: {batch.device.index}, Batch Size: {batch.shape[0]},Batch Value: {batch[:,:3]}')
        args = self.args
        if args.load_jepa_path:
            aa = batch
            g.manual_seed(batch_idx)
            final_sequence, patched, non_patched=process_sequence_array(aa.to('cpu'),args.patch_number,args.ctx_len)
            dix=torch.tensor(np.array(final_sequence)).to(args.device)
            masks_pred=torch.tensor(np.array(patched)).to(args.device)
            masks_enc=torch.tensor(np.array(non_patched)).to(args.device)
            aa=aa.to(args.device)

            idx = dix[:,:-1]
            targets = dix[:,1:]
            patch= (aa,masks_pred,masks_enc)
            # print(idx[1][:10],idx[1][-10:])
            # print(targets[1][:10],targets[1][-10:])
            # print(torch.where(idx[1]==20096)[0],torch.where(idx[1]==20097)[0],torch.where(idx[1]==20098)[0])
            # print(torch.where(targets[1]==20096)[0],torch.where(targets[1]==20097)[0],torch.where(targets[1]==20098)[0])
            # print(masks_pred[1].shape,masks_enc[1].shape)

            logits = self(idx,patch)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        else:
            aa = batch
            g.manual_seed(batch_idx)
            final_sequence, _, _=process_sequence_array(aa.to('cpu'),args.patch_number,args.ctx_len)
            dix=torch.tensor(np.array(final_sequence)).to(args.device)
            idx = dix[:,:-1]
            targets = dix[:,1:]
            logits = self(idx)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return L2Wrap.apply(loss, logits)

    def validation_step(self, batch, batch_idx):
        args = self.args
        if args.load_jepa_path:
            aa = batch
            g.manual_seed(batch_idx)
            final_sequence, patched, non_patched=process_sequence_array(aa.to('cpu'),args.patch_number,args.ctx_len)
            dix=torch.tensor(np.array(final_sequence)).to(args.device)
            masks_pred=torch.tensor(np.array(patched)).to(args.device)
            masks_enc=torch.tensor(np.array(non_patched)).to(args.device)
            aa=aa.to(args.device)

            idx = dix[:,:-1]
            targets = dix[:,1:]
            patch= (aa,masks_pred,masks_enc)
            # print(idx[1][:10],idx[1][-10:])
            # print(targets[1][:10],targets[1][-10:])
            # print(torch.where(idx[1]==20096)[0],torch.where(idx[1]==20097)[0],torch.where(idx[1]==20098)[0])
            # print(torch.where(targets[1]==20096)[0],torch.where(targets[1]==20097)[0],torch.where(targets[1]==20098)[0])
            # print(masks_pred[1].shape,masks_enc[1].shape)

            logits = self(idx,patch)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            aa = batch
            torch.manual_seed(batch_idx)  # Ensure reproducibility if needed

            final_sequence, _, _ = process_sequence_array(aa.to('cpu'), args.patch_number, args.ctx_len)
            dix = torch.tensor(np.array(final_sequence)).to(args.device)

            idx = dix[:, :-1]
            targets = dix[:, 1:]
            logits = self(idx)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        # print(f"Test Loss: {loss.item()}")

        # Return the loss tensor which can be accumulated in `test_epoch_end`
        return {"test_loss": loss}

    def validation_epoch_end(self, outputs):
        # Accumulate all the test losses from each `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        print(f"Average Test Loss: {avg_loss.item()}")

        # Optionally, log the average test loss
        # self.log('avg_test_loss', avg_loss)

    def on_train_epoch_start(self):
        # Update sampler's epoch each time an epoch starts
        self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
            ):
                m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    for kk in [
                        ".att.key.",
                        ".att.receptance.",
                        ".att.output.",
                        ".att.key.",
                        ".ffn.value.",
                        ".ffn.receptance.",
                        ".ffnPre.value.",
                        ".ffnPre.receptance.",
                        "head_q.",
                        ".oo.",
                        ".rr.",
                    ]:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(
                    f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}"
                )

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
