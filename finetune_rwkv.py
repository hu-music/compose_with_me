########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

    rank_zero_info("########## work in progress ##########")

    ########################################################################################################
    #
    # example: train a simple L12-D768 RWKV on dummy data
    #
    # python train.py --load_model "" --wandb "" --proj_dir "out" \
    # --data_file "" --data_type "dummy" --vocab_size 0 \
    # --ctx_len 128 --epoch_steps 1000 --epoch_count 20 --epoch_begin 0 --epoch_save 10 \
    # --micro_bsz 16 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 \
    # --lr_init 6e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0

    # example: train a simple L6-D512 RWKV from scratch on enwik8
    #
    # python train.py --load_model "" --wandb "" --proj_dir "out" \
    # --data_file "../data/enwik8" --data_type "utf-8" --vocab_size 0 \
    # --ctx_len 512 --epoch_steps 5000 --epoch_count 500 --epoch_begin 0 --epoch_save 5 \
    # --micro_bsz 12 --n_layer 6 --n_embd 512 --pre_ffn 0 --head_qk 0 \
    # --lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0

    # example: fine-tune RWKV 1.5B using 8xA100 40G = 1.76it/s = 115k token/s, VRAM 37477M
    #
    # python train.py --load_model "/fsx/BlinkDL/CODE/FP16/out_1b2/all-8040.pth" --wandb "" --proj_dir "out" \
    # --data_file "../data/train.npy" --data_type "numpy" --vocab_size 50277 \
    # --ctx_len 1024 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 5 \
    # --micro_bsz 8 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    # --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    # --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0

    # example: fine-tune RWKV 1.5B using 1 GPU fp16 (VRAM 16G) NOTE: fp16 might overflow
    #
    # python train.py --load_model "/fsx/BlinkDL/CODE/FP16/out_1b2/all-8040.pth" --wandb "" --proj_dir "out" \
    # --data_file "../data/train.npy" --data_type "numpy" --vocab_size 50277 \
    # --ctx_len 1024 --epoch_steps 200 --epoch_count 1000 --epoch_begin 0 --epoch_save 1 \
    # --micro_bsz 11 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    # --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    # --accelerator gpu --devices 1 --precision fp16 --strategy deepspeed_stage_2_offload --grad_cp 1

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument(
        "--wandb", default="", type=str
    )  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument(
        "--vocab_size", default=0, type=int
    )  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument(
        "--epoch_steps", default=1000, type=int
    )  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument(
        "--epoch_count", default=500, type=int
    )  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument(
        "--epoch_begin", default=0, type=int
    )  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument(
        "--epoch_save", default=5, type=int
    )  # save the model every [epoch_save] "epochs"

    parser.add_argument(
        "--micro_bsz", default=12, type=int
    )  # micro batch size (batch size per GPU)



    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument(
        "--pre_ffn", default=0, type=int
    )  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument(
        "--tiny_att_layer", default=-999, type=int
    )  # tiny attention @ which layer

    parser.add_argument(
        "--lr_init", default=6e-4, type=float
    )  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument(
        "--warmup_steps", default=50, type=int
    )  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument(
        "--beta2", default=0.99, type=float
    )  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    parser.add_argument(
        "--grad_cp", default=0, type=int
    )  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument(
        "--my_pile_shift", default=-1, type=int
    )  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument(
        "--layerwise_lr", default=1, type=int
    )  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument(
        "--ds_bucket_mb", default=200, type=int
    )  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_img_version", default=0, type=str)
    parser.add_argument("--my_img_size", default=0, type=int)
    parser.add_argument("--my_img_bit", default=0, type=int)
    parser.add_argument("--my_img_clip", default="x", type=str)
    parser.add_argument("--my_img_clip_scale", default=1, type=float)
    parser.add_argument("--my_img_l1_scale", default=0, type=float)
    parser.add_argument("--my_img_encoder", default="x", type=str)
    # parser.add_argument("--my_img_noise_scale", default=0, type=float)
    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_testing", default="", type=str)

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_load", default="", type=str)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=float)
    parser.add_argument("--lora_dropout", default=0.01, type=float)
    parser.add_argument("--lora_parts", default="att,ln,time", type=str)


    # from ijepa
    # parser.add_argument('--load_jepa_path', type=str, default="/media/bruce/ssd41/zhejing/infilling/models/ijepa/jepa-latest.pth.tar", help='ijepa model path.')
    parser.add_argument('--load_jepa_path', type=str, default="", help='ijepa model path.')

    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension.')
    parser.add_argument('--predictor_embed_dim', type=int, default=256, help='Predictor embedding dimension.')

    parser.add_argument('--encoder_depth', type=int, default=6, help='Encoder depth.')
    parser.add_argument('--predictor_depth', type=int, default=3, help='Predictor depth.')

    parser.add_argument('--encoder_num_heads', type=int, default=4, help='Encoder number of heads.')
    parser.add_argument('--predictor_num_heads', type=int, default=4, help='Predictor number of heads.')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio.')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--init_std', type=float, default=0.02, help='Initialization standard deviation.')
    parser.add_argument('--use_bfloat16', action='store_true', help='Whether to use bfloat16.')
    parser.add_argument('--pin_mem', action='store_true', help='Whether to use pin memory.')
    parser.add_argument('--device', type=str, default='cuda', help='device.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument("--patch_number", default=1, type=int)

    # You can add more arguments here as needed
    # parser.add_argument("--actual_steps", default=4000, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time, importlib
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if "deepspeed" in args.strategy:
        import deepspeed
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(
            f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n"
            * 3
        )
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings(
        "ignore", ".*Consider increasing the value of the `num_workers` argument*"
    )
    warnings.filterwarnings(
        "ignore", ".*The progress bar already tracks a metric with the*"
    )
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = -1
    args.check_val_every_n_epoch = 1
    # args.limit_val_batches= 20
    args.log_every_n_steps = int(1e20)
    args.max_epochs = args.epoch_count  # -1 continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_T_MAX"] = str(args.ctx_len)
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = args.n_embd * 4

    if args.data_type == "wds_img":
        args.run_name = f"v{args.my_img_version}-{args.my_img_size}-{args.my_img_bit}bit-{args.my_img_clip}x{args.my_img_clip_scale}"
        args.proj_dir = f"{args.proj_dir}-{args.run_name}"
    else:
        args.run_name = (
            f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
        )
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if args.my_pile_stage > 0:
        magic_prime_bak = args.magic_prime
        if args.ctx_len == 1024:
            args.magic_prime = 324331313
            args.epoch_count = 8043
        elif args.ctx_len == 2048:
            args.magic_prime = 162165671
            args.epoch_count = 4021
        elif args.ctx_len == 4096:
            args.magic_prime = 81082817
            args.epoch_count = 2010
        if args.my_pile_shift < 0:
            if args.ctx_len == 1024:
                args.my_pile_shift = 0
            elif args.ctx_len == 2048:
                args.my_pile_shift = 512
            elif args.ctx_len == 4096:
                args.my_pile_shift = 768

        if magic_prime_bak > 0:
            args.magic_prime = magic_prime_bak

        # args.epoch_steps = 40320 // args.real_bsz
        # assert args.epoch_steps * args.real_bsz == 40320
        if args.my_pile_stage == 2:
            assert args.lr_final == args.lr_init
        if args.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
            list_p.sort()
            max_p = list_p[-1]
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.my_pile_stage == 2:
                    args.warmup_steps = 10
                else:
                    args.warmup_steps = 30
            args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-4 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1}, save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
# LoRA = {f'enabled, {args.lora_r} r, {args.lora_alpha} alpha, {args.lora_dropout} dropout, on {args.lora_parts}' if args.lora else 'disabled'}
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed.__version__ if importlib.util.find_spec('deepspeed') else 'None'}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.1 or newer
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in [
        "utf-8",
        "utf-16le",
        "numpy",
        "binidx",
        "dummy",
        "wds_img",
        "uint16",
    ]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info(
            "\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n"
        )

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info(
                "\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n"
            )
    if args.precision == "fp16":
        rank_zero_info(
            "\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n"
        )

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"
    if args.lora and args.grad_cp == 1:
        print(
            "!!!!! LoRA Warning: Gradient Checkpointing requires JIT off, disabling it"
        )
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    # if args.load_jepa_path:
    #     from src.dataset import MyDataset
    #     train_data = MyDataset(args)
    # else:
    #     from src.dataset import MyDataset_vanilla
    #     train_data = MyDataset_vanilla(args)
    from src.dataset import MyDataset
    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size
    # args.ct_len = len(train_data.__getitem__(0)[0])

    if args.data_type == "wds_img":
        from src.model_img import RWKV_IMG

        assert args.lora, "LoRA not yet supported for RWKV_IMG"
        model = RWKV_IMG(args)
    else:
        from src.model import RWKV, LORA_CONFIG, LoraLinear

        if args.lora:
            assert args.lora_r > 0, "LoRA should have its `r` > 0"
            LORA_CONFIG["r"] = args.lora_r
            LORA_CONFIG["alpha"] = args.lora_alpha
            LORA_CONFIG["dropout"] = args.lora_dropout
            LORA_CONFIG["parts"] = set(str(args.lora_parts).split(","))
            enable_time_finetune = "time" in LORA_CONFIG["parts"]
            enable_ln_finetune = "ln" in LORA_CONFIG["parts"]
        model = RWKV(args)
        # only train lora parameters
        if args.lora:
            model.requires_grad_(False)
            for name, module in model.named_modules():
                # have to check param name since it may have been wrapped by torchscript
                if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                    print(f"  LoRA training module {name}")
                    for pname, param in module.named_parameters():
                        param.requires_grad = "lora_" in pname
                elif enable_ln_finetune and ".ln" in name:
                    print(f"  LoRA additionally training module {name}")
                    for param in module.parameters():
                        param.requires_grad = True
                elif enable_time_finetune and any(
                    n.startswith("time") for n, _ in module.named_parameters()
                ):
                    for pname, param in module.named_parameters():
                        if pname.startswith("time"):
                            print(f"  LoRA additionally training parameter {pname}")
                            param.requires_grad = True

    if (
        len(args.load_model) == 0 or args.my_pile_stage == 1
    ):  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        tt=0
        new_state_dict = {}
        for name, param in model.named_parameters():
            if name in load_dict and param.size() == load_dict[name].size():
                print('load very sucess:',name)
                tt+=1
                new_state_dict[name] = load_dict[name]
            elif name in load_dict and  param.size()[-1] == load_dict[name].size()[-1] and param.size()[0] > load_dict[name].size()[0]:
                print('load sucess:',name)
                tt+=1
                # Create a new parameter tensor based on the model's current parameter
                new_param = param.clone().detach()
                print(new_param.shape,load_dict[name].shape)
                # Replace the first part of the model's parameter with the loaded parameter
                new_param[:load_dict[name].size()[0], ...] = load_dict[name]
                # Assign this modified parameter to the new state dictionary
                new_state_dict[name] = new_param
            # else:
            #     # For new or incompatible parameters, use the original model's parameters
            #     new_state_dict[name] = param

        # Now load the adapted state dict
        model.load_state_dict(new_state_dict, strict=False)
        print('successfully load', tt, 'layers')

        # model.load_state_dict(load_dict, strict=(not args.lora))
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.my_pile_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(args.load_model, map_location="cpu")
            model.load_state_dict(load_dict, strict=(not args.lora))

    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]
        model.load_state_dict(load_dict, strict=(not args.lora))
    # If using LoRA, the LoRA keys might be missing in the original model
    # model.load_state_dict(load_dict, strict=(not args.lora))
    if os.path.isfile(args.lora_load):
        model.load_state_dict(
            torch.load(args.lora_load, map_location="cpu"), strict=False
        )

    trainer: Trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)]
    )

    if (
        args.lr_init > 1e-4
        or trainer.world_size * args.micro_bsz * trainer.accumulate_grad_batches < 1
    ):
        if "I_KNOW_WHAT_IM_DOING" in os.environ:
            if trainer.global_rank == 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(
                    f"  WARNING: you are using too large LR ({args.lr_init} > 1e-4) or too small global batch size ({trainer.world_size} * {args.micro_bsz} * {trainer.accumulate_grad_batches} < 8)"
                )
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            if trainer.global_rank == 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(
                    f"  ERROR: you are using too large LR ({args.lr_init} > 1e-4) or too small global batch size ({trainer.world_size} * {args.micro_bsz} * {trainer.accumulate_grad_batches} < 8)"
                )
                print(
                    f"  Unless you are sure this is what you want, adjust them accordingly"
                )
                print(
                    f'  (to suppress this, set environment variable "I_KNOW_WHAT_IM_DOING")'
                )
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            exit(0)

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
            else:
                print(f"{str(shape[0]).ljust(5)}       {n}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = (
            args.ds_bucket_mb * 1000 * 1000
        )
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = (
            args.ds_bucket_mb * 1000 * 1000
        )

    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import Sampler
    import numpy as np

    class CustomSampler(Sampler):
        def __init__(self, data_source, num_samples, num_replicas=None, rank=None):
            self.data_source = data_source
            self.num_samples = num_samples
            self.num_replicas = num_replicas
            self.rank = rank
            self.epoch = 0

        def __iter__(self):
            # Calculate the total indices (first 100*4)
            g = torch.Generator()
            g.manual_seed(self.epoch)  # Ensures different shuffle each epoch
            total_indices = len(self.data_source)

            indices = torch.randperm(total_indices, generator=g)[:self.num_samples].tolist()

            # In a distributed setting, split the indices by the replica rank
            if self.num_replicas is not None and self.rank is not None:
                indices = indices[self.rank::self.num_replicas]

            return iter(indices)

        def __len__(self):
            return self.num_samples // self.num_replicas if self.num_replicas else self.num_samples

        def set_epoch(self, epoch):
            self.epoch = epoch  # Allows the sampler to shuffle differently each epoch.

    sampler = CustomSampler(train_data, num_samples=args.epoch_steps * args.micro_bsz, num_replicas=4, rank=trainer.global_rank)
    # sampler = DistributedSampler(train_data, num_replicas=4, rank=trainer.global_rank)
    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(
        train_data,
        shuffle=False,
        pin_memory=True,
        batch_size=args.micro_bsz,
        num_workers=4,
        persistent_workers=False,
        drop_last=True,
        sampler=sampler

    )
    # from src.dataset import MyDataset_test
    # test_data = MyDataset_test()
    # sampler = DistributedSampler(test_data, num_replicas=4, rank=trainer.global_rank)
    # test_loader = DataLoader(
    #     test_data,
    #     shuffle=False,
    #     pin_memory=True,
    #     batch_size=4,
    #     num_workers=4,
    #     persistent_workers=False,
    #     drop_last=False,
    #     sampler=sampler
    #
    # )

    # for batch_idx, data in enumerate(data_loader):
    #     # print(f"Batch index: {batch_idx}")
    #     # Assuming your dataset returns a tuple of features and labels
    #     features, labels = data
    #     # print(f"Features shape: {features.shape}")  # bs * 4096
    #     # print(f"Labels shape: {labels.shape}")      # bs * 4096
    #
    #     # Optionally, print the actual data (might be too large, depending on your batch size and data)
    #     # print(f"Features data: {features[0][-200:]}")  #input
    #     # print(f"Labels data: {labels}")               #output one token shift right
    #     if batch_idx==1:
    #     # Break after the first batch to prevent printing the entire dataset
    #         break
# Configure the Trainer


    # trainer.fit(model, data_loader, test_loader)

    trainer.fit(model, data_loader)
