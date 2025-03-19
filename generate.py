from transformers import AutoTokenizer, logging
import torch
import random
import argparse
import os
import glob
import csv
import datasets
import shutil
import numpy as np
from tools.swfzip import check_if_valid
from tools.bytes_tools import bos, eos, pad, hex_to_char, char_to_hex
from vllm import LLM, SamplingParams
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings
logging.set_verbosity_error()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def main_fn(hparams):

    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(hparams.model)

    # VLLM
    if hparams.min_p is not None: 
        sampling_params = SamplingParams(temperature=hparams.temperature, min_p=hparams.min_p, 
            max_tokens=hparams.max_model_len, n=hparams.n_answers_per_prompts,
            stop_token_ids=[204640], allowed_token_ids=[i for i in range(151665, 260079+1)], skip_special_tokens=False) # stop at <eos-bytes> and only allow byte tokens
    else:
        sampling_params = SamplingParams(temperature=hparams.temperature, top_p=hparams.top_p, 
            max_tokens=hparams.max_model_len, n=hparams.n_answers_per_prompts,
            stop_token_ids=[204640], allowed_token_ids=[i for i in range(151665, 260079+1)], skip_special_tokens=False) # stop at <eos-bytes> and only allow byte tokens
    # Load model
    print('Load LLM')
    model = LLM(model=hparams.model,
        max_num_seqs=hparams.max_num_seqs,
        max_model_len=hparams.max_model_len,
        dtype='bfloat16') # float16 breaks down the model entirely, it doesnt work anymore, it only generates "!!!!!!!!", as if its angry or something.

    # Write output text responses in one file (each line is a response) and save the file from bytes is outputted
    def save_output(output_bytes, i, j): 
        output_bytes = output_bytes.replace('Ŧ','')
        output_bytes_char = output_bytes.split(bos)[1].split(eos)[0]
        readable = False
        if output_bytes_char is not None and output_bytes_char != '' and output_bytes_char.endswith('ő000'): # must end with 'ő000' for normal completion
            newfile_loc = os.path.join(hparams.output_dir, f'prompt_{i}_file{j}.swf')
            output_bytes = char_to_hex(output_bytes_char) # remove the dummy and transform back to actual bytes
            with open(newfile_loc, 'wb') as f:
                f.write(bytes(output_bytes))
            try:
                readable = check_if_valid(newfile_loc)
            except Exception as e:
                print(e)
            if not readable:
                os.remove(newfile_loc)
        return readable

    def generate(prompts):

        my_texts = []
        n_readable = 0
        game_ids = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}, # the default Qwen prompt, I never changed it.
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + "Bytes: "#ŦŌ"
            my_texts += [text]

        outputs = model.generate(my_texts, sampling_params)
        for i, o1 in enumerate(outputs):
            for j, o2 in enumerate(o1.outputs):
                output_bytes = o2.text
                try:
                    n_readable_ = save_output(output_bytes, i, j)
                    n_readable += n_readable_
                except Exception as e:
                    print(e)
                    n_readable += 0

        return n_readable, game_ids

    if hparams.prompts_hf != '':
        assert len(hparams.prompts) == 0
        assert hparams.prompt_file == ''
        ds = datasets.load_dataset(hparams.prompts_hf, streaming=False)['train']
        prompts = []
        for example in ds:
            prompts += [example['instruction']]
    elif hparams.prompt_file != '': # prompts through txt files
        assert len(hparams.prompts) == 0
        with open(hparams.prompt_file, 'r', encoding='utf-8') as f:
            prompts = f.read()
        prompts = prompts.split('\nŦ\n')
    else:
        prompts = hparams.prompts.split('Ŧ')
    assert len(prompts) > 0
    if hparams.max_n_prompts is not None:
        if hparams.randomize_order:
            random.shuffle(prompts)
        prompts = prompts[:hparams.max_n_prompts]
    print(prompts)

    set_seed(hparams.seed)
    print('Processing prompts')
    n_readable, game_ids = generate(prompts)

    total_prompts = len(prompts)*hparams.n_answers_per_prompts
    print(f'% readable (automatic, only obvious problems) = {n_readable}/{total_prompts} = {n_readable/total_prompts}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='SamsungSAILMontreal/ByteCraft')
    parser.add_argument("--output_dir", type=str, default='mygames')
    parser.add_argument("--max_num_seqs", type=int, default=10) # max batch-size (number of concurrent prompts generated); base yourself on the "Maximum concurrency for x tokens per request: y" printed by vLLM in the console and reduce it if you have memory problems.
    parser.add_argument("--max_model_len", type=int, default=32768) # max number of tokens (ByteCraft supports 32768, anything lower will cause problems due to unfinished files and higher may or may not work)
    parser.add_argument("--n_answers_per_prompts", type=int, default=10) # number of files to generate per prompt

    ## Different ways of giving an input prompt
    parser.add_argument("--max_n_prompts", type=int, default=None) # If given, limits the number of prompts processed (in case your prompt file or parquet file is big)
    
    # Option 1: give your prompts directly in the comand line
    parser.add_argument("--prompts", type=str, default="") # list of prompts seperated by Ŧ
    
    # Option 2: use a directory containing .txt with prompts
    parser.add_argument("--prompt_file", type=str, default='') # txt file with prompts seperated by \nŦ\n (I didn't want to use commas like csv since prompts can have commas, the weird Ŧ symbol is perfect for this)
    # Example:
    # -------------
    # Generate me blablabla.
    # Ŧ
    # Generate me this other thing that
    # looks like a turtle.
    # Ŧ
    # Generate me this final thing.
    # -------------

    # Option 3: Use a hugging-face dataset
    parser.add_argument("--prompts_hf", type=str, default='')
    parser.add_argument("--randomize_order", type=str2bool, default=False) # randomize order of prompts

    # sampling parameters
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9) # 1 for all tokens
    parser.add_argument("--min_p", type=float, default=None) # -1 for all tokens; if provided, top_p is ignored

    hparams = parser.parse_args()
    if not os.path.exists(hparams.output_dir):
        os.mkdir(hparams.output_dir)
    main_fn(hparams)
