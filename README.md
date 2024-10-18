# Word Swap Challenge

### _Measuring sequential logical reasoning in language models_

This test evaluates the ability of a language model to perform a series of operations in one shot. Each operation's effect depends on the previously-applied operations, which makes this test a proxy for assessing the logical complexity that a language model can handle.

A more elaborate description and discussion of the results are available in [this blogpost](https://www.goodai.com/breaking-the-chain-simple-word-swaps-expose-llms-reasoning-limits/).

## Setup and execution

This code should run with any version of Python `>=3.7`, but we suggest using version `3.12.4`. Model names must be compatible with `litellm`'s interface. Unless you just intend to load pre-computed results, you will also need to [set the API keys accordingly](https://docs.litellm.ai/docs/).

```bash
pip install -r requirements.txt
python main.py model_1 model_2 ...
```

Run `python main.py -h` to see how to change default parameters, like the number of repetitions or the global seed. You can also specify a `label` to save the results in a separate subfolder.

To reproduce the figure from the blogpost, first download the result files from [this link](https://drive.google.com/file/d/11QH-a3W6QxkIkqBeD1H7eQlqX8XKzyVN/view?usp=sharing) and place the distinct folders under `results` in the main directory (optional, but encouraged). Then run the following scripts.

```bash
# With temperature zero
python main.py --temperature 0 --label temp-0 \
    o1-mini-2024-09-12 \
    claude-3-opus-20240229 \
    claude-3-5-sonnet-20240620 \
    gpt-4-0613 \
    gpt-4-turbo-2024-04-09 \
    together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo \
    together_ai/meta-llama/Llama-3-70b-chat-hf \
    together_ai/google/gemma-2-27b-it \
    gpt-4o-2024-05-13 \
    gpt-4o-mini-2024-07-18 \
    together_ai/meta-llama/Llama-3-8b-chat-hf \
    together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    together_ai/google/gemma-2-9b-it \
    together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo
```

```bash
# With temperature one
python main.py --temperature 1 --label temp-0 \
    o1-mini-2024-09-12 \
    claude-3-opus-20240229 \
    claude-3-5-sonnet-20240620 \
    gpt-4-0613 \
    gpt-4-turbo-2024-04-09 \
    together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo \
    together_ai/meta-llama/Llama-3-70b-chat-hf \
    together_ai/google/gemma-2-27b-it \
    gpt-4o-2024-05-13 \
    gpt-4o-mini-2024-07-18 \
    together_ai/meta-llama/Llama-3-8b-chat-hf \
    together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    together_ai/google/gemma-2-9b-it \
    together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo
```

If you chose to not download the result files, now you will need to manually select the best result from the `temp-0` and `temp-1` result directories and copy them to another result folder named `temp-mix`.

```bash
# Selected best results
python main.py --label temp-mix \
    o1-mini-2024-09-12 \
    claude-3-opus-20240229 \
    claude-3-5-sonnet-20240620 \
    gpt-4-0613 \
    gpt-4-turbo-2024-04-09 \
    together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo \
    together_ai/meta-llama/Llama-3-70b-chat-hf \
    together_ai/google/gemma-2-27b-it \
    gpt-4o-2024-05-13 \
    gpt-4o-mini-2024-07-18 \
    together_ai/meta-llama/Llama-3-8b-chat-hf \
    together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    together_ai/google/gemma-2-9b-it \
    together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo
```