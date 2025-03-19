# 🎮ByteCraft: Generating video games and animations through bytes

This repo contains inference code for [ByteCraft](https://huggingface.co/EmyAze/ByteCraft), the world's first generative model of SWF video games and animations through bytes conditional on prompt.

<p align="center">
  <img src="https://emygervais.github.io/assets/images/screenshots.png" alt="Screenshots of files generated by ByteCraft"  style="width: 100%;">
  Figure: Screenshots of files generated by ByteCraft
</p>

You can find examples of generated files in our [blog](https://emygervais.github.io/2025/03/15/bytecraft.html). We also provide a [Paper/Tech-report](https://github.com/SamsungSAILMontreal/ByteCraft/blob/main/paper/ByteCraft.pdf).

Since ByteCraft is the first of its kind, expect most generated files to be broken. Treat it as a lootbox, generate as many files as possible, open them one by one and sometimes you will find something interesting. This is a research prototype on what can be achieved by modeling bytes of complex files.

## Setting up the environment

Install [Ruffle](https://ruffle.rs) to read the generated files.

You must install all the requirements below (adapted to your situation).
```
# assuming python/3.10, cuda/11.8
pip install --upgrade pip setuptools wheel ninja
pip install xformers https://github.com/vllm-project/vllm/releases/download/v0.7.3/vllm-0.7.3+cu118-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
pip install datasets==3.1.0 pylzma bitstring
MAX_JOBS=64 pip install flash-attn --no-build-isolation
```

## Generating bytes

Through vLLM, you can generate multiple files at a time (option max_num_seqs, set it to the maximum given to you by vLLM). On a A100 80Gb GPU, you can generate 25 files at a time in about 10min. Note that the model does not work when loaded in float16 (it returns "!!!!!!!!!"); it only works properly in bfloat16. It is not clear why this is the case.

You can provide prompts in various ways: 1) command-line, 2) text file containing multiple prompts, or 3) huggingface dataset such as our dataset of 10K synthetic prompts [GameAnimationPrompts10K](https://huggingface.co/datasets/SamsungSAILMontreal/GameAnimationPrompts10K). 

See [exp/generating_video_games.sh](https://github.com/SamsungSAILMontreal/ByteCraft/blob/main/exp/generating_video_games.sh) for examples on how to use generating SWF files with each of the 3 methods provided.

## Prompt format (examples)

You can write the prompt however you like, it's possible that there is an optimal way in order to improve output quality or diversity. We recommend the following two formats as example. Feel free to be creative and try to wild broken prompts.

Structured prompt:
```
"Please generate a video game with the following specifications: 
- Frame rate: 24.0
- Resolution: 500x400
- Language(s): English, Japanese
- Title: MegaTurtles
- Size: 35KB
- Release Date: 2011
- Description: This is a game about turtles fighting in an arena.
"
```

Unstructured prompt:
```
"Generate an animation titled MegaTurtles depicting turtles fighting in an arena. It should be 30Kb in size and in english language. Make sure that the resolution is 500x400.
```

## Prompt examples for a quick start

We provide 10K synthetic examples of prompts in [GameAnimationPrompts10K](https://huggingface.co/datasets/SamsungSAILMontreal/GameAnimationPrompts10K). Feel free to use this dataset as starting point. You can also also ask any LLM to generate new prompts based on some examples (few-shot prompting).

## Hyperparameters
```
n_answers_per_prompts=5 # number of randomly generated files per prompt
max_model_len=32768 # maximum number of tokens generated; should be left for optimal performance.
max_num_seqs=25 # maximum number of concurrent prompts generated; base yourself on the "Maximum concurrency for x tokens per request: y" printed by vLLM in the console and reduce it if you have memory problems.
temperature=0.1 # lower temperature = more deterministic and less diverse
min_p=0.1 # higher min_p = more deterministic and less diverse
```

## Reference

Soon
