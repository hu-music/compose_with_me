########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
# from .utils import MaybeIsPrime
from torch.nn.functional import pad
# Set seed for reproducibility
np.random.seed(42)
def create_answer_sequence_array(sequence, start_indices, patch_size, sep_token=20098):
    """
    Create the answer sequence from selected patches, separated by `sep_token` using NumPy.
    """
    answer = []
    for start_idx in start_indices:
        patch = sequence[start_idx:start_idx + patch_size]
        answer.extend(patch)
        answer.append(sep_token)
    # Remove the last sep_token
    answer = answer[:]
    return np.array(answer)



def get_non_patched_indices(sequence_length, patched_indices):
    """
    Determine non-patched indices from the sequence based on patched indices.
    """
    non_patched_indices = [i for i in range(sequence_length) if i not in patched_indices]
    return non_patched_indices


def process_sequence_array(sequence, num_patches, ctx_len, mask_token=20096, sep_token=20097, ans_token=20098):
    """
    Apply modifications to a single sequence according to specified rules using NumPy.
    Returns the final modified sequence along with patched and non-patched indices.
    """

    # patched_size=128
    patched_indices_all=[]
    non_patched_indices_all=[]
    final_sequence_all=[]

    non_zero_all=[]
    for i in range(len(sequence)):
        if len(np.where(sequence[i]==0)[0])==0:
            non_zero=ctx_len-3
        else:
            non_zero=np.where(sequence[i]==0)[0][0]
        non_zero_all.append(non_zero)
    patch_size = np.random.randint(16, min(min(non_zero_all)-3,int(ctx_len*0.4)))
    available_index=non_zero-patch_size-1
    start_indices = np.random.choice(range(1,available_index))
    for i in range(len(sequence)):
        # if len(np.where(sequence[i]==0)[0])==0:
        #     non_zero=ctx_len-3
        # else:
        #     non_zero=np.where(sequence[i]==0)[0][0]
        # # print('non_zero:',non_zero)
        # patch_size = np.random.randint(16, min(non_zero-3,int(ctx_len*0.4)))
        # available_index=non_zero-patch_size-1
        # start_indices = 1
        # start_indices = available_index

        # if len(np.where(sequence[i]==0)[0]) == 0:
        #     start_indices = np.array([4096-patch_size])
        # else:
        #     start_indices = np.array([list(sequence[i]).index(0)-patch_size])
        # print('available_index:',available_index)
        patched_indices=list(range(start_indices,start_indices+patch_size))
        non_patched_indices=get_non_patched_indices(len(sequence[i]), patched_indices)
        answer_seq = create_answer_sequence_array(sequence[i], [start_indices], patch_size, sep_token=20098)
        new_sequence=torch.concat([sequence[i][:start_indices],torch.tensor([20096]),sequence[i][start_indices+patch_size:non_zero_all[i]-1],
             torch.tensor([20097]),torch.tensor(answer_seq),torch.tensor([2])])
        if len(np.where(sequence[i]==0)[0])!=0:
            final_sequence=np.array(torch.concat([new_sequence,torch.zeros(4096-new_sequence.shape[0],dtype=int)]))
        else:
            final_sequence = np.array(new_sequence)
        # print('tessas!',final_sequence.shape)
        final_sequence_all.append(final_sequence)
        patched_indices_all.append(patched_indices)
        non_patched_indices_all.append(non_patched_indices)

    return final_sequence_all, patched_indices_all, non_patched_indices_all




class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args


        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)"
        )

        if args.data_file.endswith("/"):
            d_all = []
            for p in os.listdir(args.data_file):
                if p.endswith(".idx"):
                    d_all += [p[:-4]]
            d_all.sort()
            rank_zero_info(d_all)
            exit(0)
        else:
            self.data = MMapIndexedDataset(args.data_file)
            # self.data_size = (
            #     len(self.data._bin_buffer) // self.data._index._dtype_size
            # )
            # rank_zero_info(f"Data has {self.data_size} tokens.")
            self.filtered_indices = []
            for idx in range(len(self.data)):
                if len(self.data.get(idx=idx)) >= 0:  # and len(self.data.get(idx=idx)) < 4096+4096:
                    self.filtered_indices.append(idx)

            self.data_size = len(self.filtered_indices)
            rank_zero_info(f"Train Data has {self.data_size} samples longer than 512 tokens.")

    def __len__(self):
        # return self.args.epoch_steps * self.args.micro_bsz
        return min(self.args.epoch_steps * self.args.micro_bsz, len(self.filtered_indices))
    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        data = self.data.get(idx=actual_idx)
        ctx_len = self.args.ctx_len - 1 - 2 * self.args.patch_number
        req_len = ctx_len + 1

        if len(data) <= req_len:
            padding_length = req_len - len(data)
            # Convert data to int64 before concatenation to match types
            data = np.concatenate((data.astype(np.int64), np.zeros(padding_length, dtype=np.int64)))
        else:
            start_index = 0
            data = data[start_index:start_index + req_len]

        # Ensure data is converted to a supported type (e.g., int64) before conversion to tensor
        x_origin = torch.tensor(data.astype(np.int64), dtype=torch.long)
        return x_origin

    # def __getitem__(self, idx):
    #     args = self.args
    #     ctx_len = args.ctx_len - 1- 2*args.patch_number
    #     req_len = ctx_len+1
    #     data = self.data
    #
    #     actual_len = len(data.get(idx=idx))
    #     if actual_len <= req_len:
    #         padding_length = req_len - actual_len
    #         dix = data.get(idx=idx, offset=0, length=actual_len).astype(int) # always start from begining
    #         dix = np.concatenate((dix, np.zeros(padding_length, dtype=int)))
    #
    #     else:
    #         i = np.random.randint(0, actual_len - req_len)
    #         dix = data.get(idx=idx, offset=0, length=req_len).astype(int)
    #
    #     x_origin = torch.tensor(dix[:], dtype=torch.long)
    #     # y = torch.tensor(dix[1:], dtype=torch.long)
    #
    #     # dix,patches,non_patches=process_sequence_array(dix,args.patch_number)
    #     # x = torch.tensor(dix[:-1], dtype=torch.long)
    #     # y = torch.tensor(dix[1:], dtype=torch.long)
    #     #
    #     # #
    #     # # x = torch.tensor(dix[:-1], dtype=torch.long)
    #     # # y = torch.tensor(dix[1:], dtype=torch.long)
    #     # max_length=x_origin.size(0)
    #     #
    #     # patchesr = torch.tensor(patches, dtype=torch.long)
    #     # non_patches = torch.tensor(non_patches, dtype=torch.long)
    #     # patches = pad(patches_tensor, (0, max_length - patches_tensor.size(0)), "constant", 0)
    #     # non_patches = pad(non_patches_tensor, (0, max_length - non_patches_tensor.size(0)), "constant", 0)
    #
    #     return x_origin
        # return x,y, x_origin, patches, non_patches

class MyDataset_test(Dataset):
    def __init__(self, data_file="data/pop909_document",epoch_steps=200,micro_bsz=4,ctx_len=4096,vocab_size=20099,patch_number=1):
        self.vocab_size = vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)"
        )

        self.data = MMapIndexedDataset(data_file)
        self.filtered_indices = []
        for idx in range(len(self.data)):
            if len(self.data.get(idx=idx)) > 512:  # and len(self.data.get(idx=idx)) < 4096+4096:
                self.filtered_indices.append(idx)

        self.data_size = len(self.filtered_indices)
        self.epoch_steps =epoch_steps
        self.micro_bsz = micro_bsz
        self.ctx_len=ctx_len
        self.patch_number=patch_number
        rank_zero_info(f"Test Data has {self.data_size} samples longer than 512 tokens.")

    def __len__(self):
        return min(self.epoch_steps * self.micro_bsz, len(self.filtered_indices))
    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        data = self.data.get(idx=actual_idx)
        ctx_len = self.ctx_len - 1 - 2 * self.patch_number
        req_len = ctx_len + 1

        if len(data) <= req_len:
            padding_length = req_len - len(data)
            # Convert data to int64 before concatenation to match types
            data = np.concatenate((data.astype(np.int64), np.zeros(padding_length, dtype=np.int64)))
        else:
            start_index = 0
            data = data[start_index:start_index + req_len]

        # Ensure data is converted to a supported type (e.g., int64) before conversion to tensor
        x_origin = torch.tensor(data.astype(np.int64), dtype=torch.long)
        return x_origin


class MyDataset_vanilla(Dataset):
    def __init__(self, args):
        self.args = args


        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)"
        )

        if args.data_file.endswith("/"):
            d_all = []
            for p in os.listdir(args.data_file):
                if p.endswith(".idx"):
                    d_all += [p[:-4]]
            d_all.sort()
            rank_zero_info(d_all)
            exit(0)
        else:
            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = (
                len(self.data._bin_buffer) // self.data._index._dtype_size
            )
            rank_zero_info(f"Data has {self.data_size} tokens.")


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        ctx_len = args.ctx_len - 1- 2*args.patch_number
        req_len = ctx_len+1
        data = self.data

        actual_len = len(data.get(idx=idx))
        if actual_len <= req_len:
            padding_length = req_len - actual_len
            dix = data.get(idx=idx, offset=0, length=actual_len).astype(int)
            dix = np.concatenate((dix, np.zeros(padding_length, dtype=int)))

        else:
            i = np.random.randint(0, actual_len - req_len)
            dix = data.get(idx=idx, offset=i, length=req_len).astype(int)

        x_origin = torch.tensor(dix[:], dtype=torch.long)
        dix,patches,non_patches=process_sequence_array(x_origin.unsqueeze(0).to('cpu'),args.patch_number,args.ctx_len)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x,y
