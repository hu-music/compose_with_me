# Compose with Me: Collaborative Music Inpainter for Symbolic Music Infilling

This repository contains the code for the AAAI 2025 paper: **"Compose with Me: Collaborative Music Inpainter for Symbolic Music Infilling"**.

## Prerequisites

Ensure the following dependencies are installed:

```bash
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```


## Model Checkpoints

You can download the model checkpoints from the following link:  
[Download Model Checkpoints](https://drive.google.com/drive/folders/12xQxQF0JQiTBMwhNL7mKB1IMZk3Rcs5i?usp=sharing)

## Datasets

- **Bread Dataset:** [Download Bread Dataset](https://huggingface.co/datasets/breadlicker45/bread-midi-dataset/tree/main)  
- **LMD Dataset:** [Download LMD Dataset](https://colinraffel.com/projects/lmd/)

## Data Preparation

### Step 0: Prepare Data from MIDI Format

**0.1 Convert MIDI to JSON**  

Follow the instructions in [MIDI-LLM-tokenizer](https://github.com/briansemrau/MIDI-LLM-tokenizer) for converting MIDI files to JSON format (under MIDI-LLM-tokenizer folder):

```bash
python ./midi_to_jsonl.py --path test.zip --output test-midi.jsonl --workers 4
```
**0.2 Convert JSON to IDXBIN**

Follow the instructions in [Json2Binidx](https://github.com/Abel2076/json2binidx_tool) for converting the JSON files to IDXBIN format (under json2binidx_tool folder):

```bash
python tools/preprocess_data.py --input data/test-midi.jsonl --output-prefix data/test --vocab tokenizer-midi.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod
```


## Step 1: Pre-train MJEPA to Extract Musical Features

Run the following command to pre-train MJEPA:

```bash
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train_mjepa.py --micro_bsz 2 --ctx_len 4096 --epoch_steps 5000 --log_folder "./check_points/mjepa/"
```

## Step 2: Fine-tune the Autoregressive Generative Architecture (AGA), RWKV-based

**Option 1: Fine-tune the Complete Model**

```bash
python finetune_rwkv.py --load_model "check_points/RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth" \
--proj_dir "check_points/rwkv/" --data_file "data/pop909_document" --data_type binidx \
--vocab_size 20099 --ctx_len 4096 --epoch_steps 100 --epoch_count 500 --epoch_begin 0 \
--epoch_save 1 --micro_bsz 1 --n_layer 64 --n_embd 800 --pre_ffn 0 --head_qk 0 \
--lr_init 1e-5 --lr_final 1e-7 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
--accelerator gpu --devices 2 --precision fp16 --strategy deepspeed_stage_2 --grad_cp 0 \
--patch_number 1 --load_jepa_path "check_points/mjepa/jepa-latest.pth.tar"
```

**Option 2: Fine-tune with LoRA (Low-Rank Adaptation)**

```bash
python finetune_rwkv.py --load_model "check_points/RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth" \
--proj_dir "check_points/rwkv/" --data_file "data/pop909_document" --data_type binidx \
--vocab_size 20099 --ctx_len 4096 --epoch_steps 100 --epoch_count 500 --epoch_begin 0 \
--epoch_save 1 --micro_bsz 1 --n_layer 64 --n_embd 800 --pre_ffn 0 --head_qk 0 \
--lr_init 1e-5 --lr_final 1e-7 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
--accelerator gpu --devices 2 --precision fp16 --strategy deepspeed_stage_2 --grad_cp 0 \
--patch_number 1 --lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln \
--load_jepa_path "check_points/mjepa/jepa-latest.pth.tar"
```

***Merge Fine-tuned Models with LoRA***

```bash
python merge_lora.py --use-gpu 16 "check_points/RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth" \
"check_points/rwkv/rwkv-1.pth" "check_points/finetuned/lora_out_0.pth"
```

## Step 3: Fine-tune MJEPA Based on Preference Fine-tuning

To perform personalization and preference fine-tuning, run the following commands:

```bash
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train_personalization_iteration.py
```

```bash
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train_personalization_sample.py
```

Alternatively, you can run the script from the Jupyter notebook HITL_training.ipynb.

## Step 4: Test the Model

Once the model is fine-tuned, you can test it with the following command:

```bash
python test.py --model_path "check_points/finetuned/lora_out_0.pth" \
--save_path "./results/" --mjepa "check_points/mjepa/jepa-latest.pth.tar" --fixed_location 2
```
