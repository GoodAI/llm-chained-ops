# Switch-Switch

### _Measuring sequential logical reasoning in language models_

This test evaluates the ability of a language model to perform a series of operations in one shot. Each operation's effect depends on the previously-applied operations, which makes this test a proxy for assessing the logical complexity that a language model can handle.

A more elaborate description and discussion of the results are available in [this blogpost]().

## Setup and execution

This code should run with any version of Python `>=3.7`, but we used version `3.12.4`. Model names must be compatible with `litellm`'s interface. Unless you just intend to load pre-computed results, you will also need to [set the API keys accordingly](https://docs.litellm.ai/docs/).

```bash
pip install -r requirements.txt
python main.py model_1 model_2 ...
```

Run `python main.py -h` to see how to change default parameters, like the number of repetitions or the global seed. You can also specify a `label` to save the results in a separate subfolder.

To reproduce the figure from the blogpost, first download the result files from [this link](https://drive.google.com/file/d/1LZEeqkuBjjDaIm-WkQ6fJ6p1LzFeEy66/view?usp=sharing) and place the `results` folder in the main directory. Then run the script as follows:

```bash
python main.py \
    gpt-4-0613 \
    gpt-4-turbo-2024-04-09 \
    gpt-4o-2024-05-13 \
    gpt-4o-mini-2024-07-18 \
    o1-mini-2024-09-12 \
    claude-3-5-sonnet-20240620 \
    claude-3-opus-20240229 \
    together_ai/google/gemma-2-9b-it \
    together_ai/google/gemma-2-27b-it \
    together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo \
    together_ai/meta-llama/Llama-3-8b-chat-hf \
    together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo \
    together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    together_ai/meta-llama/Llama-3-70b-chat-hf \
    together_ai/NousResearch/Hermes-3-Llama-3.1-405B-Turbo
```