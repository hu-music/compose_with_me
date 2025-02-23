import os

import copy
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler

# from preprocessing.dataset import MyDataset, process_sequence_array
from src.dataset import MyDataset,process_sequence_array

from model_mjepa import Encoder, Predictor, apply_masks
from ijepa.helper import load_checkpoint, init_opt
from ijepa.utils.distributed import AllReduce
from ijepa.utils.logging import AverageMeter, gpu_timer, grad_logger
from ijepa.utils.tensors import trunc_normal_, repeat_interleave_batch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on multiple GPUs using DistributedDataParallel.")
    parser.add_argument('--data_file', type=str, default="./data/pop909_document", help='datafile.')
    parser.add_argument('--data_type', type=str, default="binidx", help='data type')
    parser.add_argument('--proj_dir', type=str, default="out", help='data type')
    parser.add_argument('--root_path', type=str, default="./", help='data type')
    parser.add_argument('--log_folder', type=str, default="./models/mjepa/", help='data type')
    parser.add_argument('--write_tag', type=str, default="jepa", help='data type')

    parser.add_argument('--vocab_size', type=int, default=20099, help='Vocabulary size.')
    parser.add_argument('--ctx_len', type=int, default=4096, help='Context length.')

    parser.add_argument('--epoch_steps', type=int, default=100000, help='Steps per epoch.')
    parser.add_argument('--micro_bsz', type=int, default=3, help='Micro batch size.')

    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension.')
    parser.add_argument('--encoder_depth', type=int, default=6, help='Encoder depth.')
    parser.add_argument('--predictor_embed_dim', type=int, default=256, help='Predictor embedding dimension.')
    parser.add_argument('--predictor_depth', type=int, default=3, help='Predictor depth.')
    parser.add_argument('--encoder_num_heads', type=int, default=4, help='Encoder number of heads.')
    parser.add_argument('--predictor_num_heads', type=int, default=4, help='Predictor number of heads.')
    parser.add_argument('--patch_number', type=int, default=1, help='Number of patches.')

    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio.')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--init_std', type=float, default=0.02, help='Initialization standard deviation.')
    parser.add_argument('--ipe_scale', type=int, default=1, help='IPE scale.')
    parser.add_argument('--ema', nargs=2, type=float, default=[0.996, 1.0], help='EMA bounds.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--final_lr', type=float, default=1e-5, help='Final learning rate.')
    parser.add_argument('--final_weight_decay', type=float, default=0.4, help='Final weight decay.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--start_lr', type=float, default=1e-4, help='Start learning rate.')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup steps.')
    parser.add_argument('--weight_decay', type=float, default=0.04, help='Weight decay.')
    parser.add_argument('--use_bfloat16', action='store_true', help='Whether to use bfloat16.')
    parser.add_argument('--pin_mem', action='store_true', help='Whether to use pin memory.')

    # parser.add_argument('--load_checkpoint', type=str, default='../../mjepa/logs/jepa-latest.pth.tar', help='Path to the checkpoint to load.')
    parser.add_argument('--load_checkpoint', type=str, default='', help='Path to the checkpoint to load.')

    parser.add_argument('--device', type=str, default='cuda', help='device.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    # You can add more arguments here as needed

    args = parser.parse_args()
    return args

# Initialize distributed training
def init_distributed(port=12351):
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size




def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    rank, world_size = init_distributed()

    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR)
    logger = logging.getLogger()

    # Setup data
    train_data = MyDataset(args)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(train_data, batch_size=args.micro_bsz, sampler=train_sampler, pin_memory=args.pin_mem, drop_last=True)

    folder = args.log_folder
    tag = args.write_tag
    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # Setup model and optimizer
    encoder = Encoder(args).to(args.device)
    predictor = Predictor(args).to(args.device)
    target_encoder = copy.deepcopy(encoder)
    loss_function=nn.MSELoss()
    log_freq =1
    checkpoint_freq = 1
    log_timings = True

    _GLOBAL_SEED = 0
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    if world_size > 1:
        encoder = DistributedDataParallel(encoder, device_ids=[rank])
        predictor = DistributedDataParallel(predictor, device_ids=[rank])
        target_encoder = DistributedDataParallel(target_encoder,device_ids=[rank])
    for p in target_encoder.parameters():
        p.requires_grad = False
    # -- init optimizer and scheduler
    ipe = len(data_loader)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=args.weight_decay,
        final_wd=args.final_weight_decay,
        start_lr=args.start_lr,
        ref_lr=args.lr,
        final_lr=args.final_lr,
        iterations_per_epoch=len(data_loader),
        warmup=args.warmup,
        num_epochs= args.epochs,
        ipe_scale=args.ipe_scale,
        use_bfloat16=args.use_bfloat16)

    momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(ipe*args.epochs*args.ipe_scale)
                  for i in range(int(ipe*args.epochs*args.ipe_scale)+1))

    start_epoch = 0

    # -- load training checkpoint
    if args.load_checkpoint:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=args.device,
            r_path=args.load_checkpoint,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            # mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': args.micro_bsz,
            'world_size': world_size,
            'lr': args.lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()
        g = torch.Generator()

        for itr, data in enumerate(data_loader):
            g.manual_seed(itr)
            final_sequence, patched, non_patched=process_sequence_array(data,args.patch_number,args.ctx_len)
            dix=torch.tensor(np.array(final_sequence)).to(device)
            masks_pred=torch.tensor(np.array(patched)).to(device)
            masks_enc=torch.tensor(np.array(non_patched)).to(device)
            data=data.to(device)
            maskA_meter.update(len(masks_pred[0]))
            maskB_meter.update(len(masks_enc[0]))
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --
                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(data)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = h.mean(dim=1, keepdim=True)
                        return h

                def forward_context():
                    z = encoder(data, masks_enc)
                    z = predictor(z, data,masks_enc, masks_pred)
                    z = z.mean(dim=1, keepdim=True)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h,beta=0.01,reduction='sum')
                    # loss= loss_function(z,h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if args.use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()
                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                # csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))
            if itr % 100 ==0:
                log_stats()

            assert not np.isnan(loss), 'loss is nan'
            pass
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
        # Save checkpoints and logging
        if rank == 0:
            # Save model and other necessary items
            pass

if __name__ == "__main__":
    main()
