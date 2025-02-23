import os
import numpy as np
import torch
from model_inference import RWKV
from rwkv.utils import PIPELINE
from MIDI import midi_util
from MIDI.midi_util import VocabConfig, FilterConfig
import argparse
from src.dataset import MyDataset
from model_mjepa import Encoder, Predictor, apply_masks
import time
np.random.seed(42)
def text_to_midi(input_text, output_path, vocab_config_path,bpm):
    cfg = VocabConfig.from_json(vocab_config_path)
    text = input_text.strip()
    mid = midi_util.convert_str_to_midi(cfg, text,bpm)
    mid.save(os.path.abspath(output_path))
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

    patch_size=args.patch_size
    patched_indices_all=[]
    non_patched_indices_all=[]
    final_sequence_all=[]
    for i in range(len(sequence)):
        if len(np.where(sequence[i]==0)[0])==0:
            non_zero=ctx_len-3
        else:
            non_zero=np.where(sequence[i]==0)[0][0]
        available_index=non_zero-patch_size-1
        if args.fixed_location==1:
            start_indices = 128
        else:
            start_indices = np.random.choice(range(1,available_index))

        patched_indices=list(range(start_indices,start_indices+patch_size))
        non_patched_indices=get_non_patched_indices(len(sequence[i]), patched_indices)
        answer_seq = create_answer_sequence_array(sequence[i], [start_indices], patch_size, sep_token=20098)
        new_sequence=torch.concat([sequence[i][:start_indices],torch.tensor([20096]),sequence[i][start_indices+patch_size:non_zero-1],
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





def main(args):
    # Set environment variables
    os.environ['RWKV_JIT_ON'] = '1'
    model_path = args.model_path
    EOS_ID = 0
    TOKEN_SEP = ' '
    TRIAL = 0
    ANS_ID = 20098

    # Load model and tokenizer
    model = RWKV(model=model_path, strategy='cuda:0 fp16 *65 -> cpu fp32')
    pipeline = PIPELINE(model, "./json2binidx_tool/tools/tokenizer-midi.json")
    tokenizer = pipeline

    train_data = MyDataset(args)
    if args.mjepa:
        print('Using MJEPA!!!:')
        jepa_encoder = Encoder(args)
        jepa_predictor = Predictor(args)
        # print(self.jepa_encoder.state_dict())
        checkpoint = torch.load(args.mjepa, map_location=torch.device(args.device))
        pretrained_dict = checkpoint['encoder']
        jepa_encoder.load_state_dict(pretrained_dict)
        pretrained_dict = checkpoint['predictor']
        jepa_predictor.load_state_dict(pretrained_dict)
        jepa_encoder=jepa_encoder.to(args.device)
        jepa_predictor=jepa_predictor.to(args.device)
        print('sucessfully loading MJEPA from:', args.mjepa)

    lis = np.random.randint(0, 900, 10) # test on 10 samples
    for j in lis:
        for qq in range(3):
            try:
                a = train_data.__getitem__(j)
                final_sequence, patched, non_patched = process_sequence_array(a.unsqueeze(0), 1, 4096)
                masks_pred = torch.tensor(np.array(patched))
                masks_enc = torch.tensor(np.array(non_patched))
                a = a.to(args.device)
                masks_pred = masks_pred.to(args.device)
                masks_enc = masks_enc.to(args.device)

                ccc = list(final_sequence[0][:np.where(final_sequence[0]==20097)[0][0]+1])
                gt= list(final_sequence[0][np.where(final_sequence[0]==20097)[0][0]+1:np.where(final_sequence[0]==20098)[0][0]])
                occurrence = {}
                state = None
                ccc1=[]
                if args.mjepa:
                    with torch.no_grad():  # Ensure no gradients are computed for the fixed parts
                        z = jepa_encoder(a.unsqueeze(0), masks_enc)
                        z1 = jepa_predictor(z, a.unsqueeze(0), masks_enc, masks_pred)
                else:
                    z1=None
                for i in range(args.ctx_len):
                    start_time = time.time()  # Start timing for this token generation

                    if i == 0:
                        out, state = model.forward(ccc, state,z1)
                    else:
                        out, state = model.forward([token], state,z1)
                # Timing end after generating the token
                    end_time = time.time()
                    token_generation_time = end_time - start_time  # Time taken for this token
                    print(f"Time to generate token {i}: {token_generation_time:.4f} seconds")

                    # MIDI mode adjustments
                    for n in occurrence:
                        out[n] -= (0 + occurrence[n] * 0.5)

                    out[0] += (i - 5000/1) / (500/1)  # not too short, not too long
                    if args.patch_size>1000:
                        out[20098] += (i - 1000/1) / (500/1)  # not too short, not too long
                    else:
                        out[20098] += (i - 1000/4) / (500/4)  # not too short, not too long
                    # out[20098] += (i - 1000/1) / (500/1)  # not too short, not too long

                    out[127] -= 1  # avoid "t125"
                    out[20097] -= 1
                    out[20096] -= 1

                    # token = pipeline.sample_logits(out, temperature=1.0, top_k=8, top_p=0.8)
                    token = pipeline.sample_logits(out, temperature=1.0, top_k=8, top_p=0.8)
                    if token == ANS_ID:
                        print('finish!!!')
                        break
                    if token == EOS_ID: break

                    for n in occurrence: occurrence[n] *= 0.997  #### decay repetition penalty
                    if token >= 128 or token == 127:
                        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
                    else:
                        occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)

                    ccc += [token]
                    ccc1+=[token]
                ccc+= [ANS_ID]
                idx=np.where(np.array(ccc)==20096)[0][0]
                idx1=np.where(np.array(ccc)==20097)[0][0]
                new_seq=ccc[:idx]+ccc1+ccc[idx+1:idx1]
                gt_new_seq=ccc[:idx]+gt+ccc[idx+1:idx1]
                print(len(ccc1),len(gt))
                patched_new=[list(range(len(ccc[:idx]),(len(ccc[:idx])+len(ccc1))))]
                non_patched_new=[list(range(len(ccc[:idx]))) + list(range(patched_new[0][-1]+1,len(new_seq)))]
                masks_pred_new=torch.tensor(np.array(patched_new))
                masks_enc_new=torch.tensor(np.array(non_patched_new))

                seq='<start>'
                for c in new_seq:
                    seq+=' '+tokenizer.decode([c])
                seq+=' <end>'


                gt_seq='<start>'
                for c in gt_new_seq:
                    gt_seq+=' '+tokenizer.decode([c])
                gt_seq+=' <end>'

                data_to_save = np.array([ccc[:idx], ccc1, gt, ccc[idx+1:idx1]], dtype=object)
                np.save(args.save_path+'txt/' + str(j)+'_'+str(qq), data_to_save)
                text_to_midi(seq,args.save_path+'mid/'+str(j)+'_'+str(qq)+'.mid',"./MIDI/vocab_config.json",120)
                text_to_midi(gt_seq,args.save_path+'/mid/gt_'+str(j)+'_'+str(qq)+'.mid',"./MIDI/vocab_config.json",120)

            except Exception as e:
                print(f'Error: {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for generating music')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation (default: cuda:1)')
    parser.add_argument("--mjepa", default="", type=str)
    parser.add_argument('--fixed_location', type=int, default=1, help='fixed location')
    parser.add_argument('--vocab_config_path', type=str, default='MIDI/vocab_config.json', help='Path to vocab config file')
    parser.add_argument('--bpm', type=int, default=120, help='Beats per minute for the music (default: 120)')
    parser.add_argument('--data_file', type=str, default="json2binidx_tool/data/pop909_document", help='Root path for file operations')
    parser.add_argument('--epoch_steps', type=int, default=1000, help='Beats per minute for the music (default: 120)')
    parser.add_argument('--micro_bsz', type=int, default=2, help='Beats per minute for the music (default: 120)')
    parser.add_argument('--patch_number', type=int, default=1, help='Beats per minute for the music (default: 120)')
    parser.add_argument('--patch_size', type=int, default=128, help='modify size')
    parser.add_argument('--ctx_len', type=int, default=4096, help='Beats per minute for the music (default: 120)')
    parser.add_argument('--vocab_size', type=int, default=20099, help='Beats per minute for the music (default: 120)')
    parser.add_argument('--model_path', type=str, default='check_points/finetuned/lora_out_0.pth', help='Root path for file operations')
    parser.add_argument('--save_path', type=str, default='results/', help='Root path for file operations')


    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension.')
    parser.add_argument('--predictor_embed_dim', type=int, default=256, help='Predictor embedding dimension.')

    parser.add_argument('--encoder_depth', type=int, default=6, help='Encoder depth.')
    parser.add_argument('--predictor_depth', type=int, default=3, help='Predictor depth.')

    parser.add_argument('--encoder_num_heads', type=int, default=4, help='Encoder number of heads.')
    parser.add_argument('--predictor_num_heads', type=int, default=4, help='Predictor number of heads.')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio.')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--init_std', type=float, default=0.02, help='Initialization standard deviation.')


    args = parser.parse_args()
    main(args)
