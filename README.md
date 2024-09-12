# Switch-Switch

### _Measuring sequential logical reasoning in language models_

This test evaluates the ability of a language model to perform a series of operations in one shot. Each operation's effect depends on the previously-applied operations, which makes this test a proxy for assessing the logical complexity that a language model can handle.

A more elaborate description and display of the results are available in [this blogpost]().

## Setup and execution

This code should run with any version of Python `>=3.7`, but we used version `3.12.4`.

```bash
pip install -r requirements.txt
python main.py --ops 30 --reps 10 <model_1> <model_2> ...
```

To reproduce the figure from the blogpost, first download the result files from [this link](https://drive.google.com/file/d/1mEwz7BUowOJ-9cs-wiq3kcjHXQYTgBXd/view?usp=sharing) and place the `results` folder in the main directory. Then run the script as follows:

```bash
python main.py --ops 30 --reps 10 \
       claude-3-5-sonnet-20240620 \
       gpt-4 gpt-4-turbo gpt-4o gpt-4o-mini \
       together_ai/google/gemma-2-9b-it \
       together_ai/meta-llama/Llama-3-8b-chat-hf \
       together_ai/meta-llama/Llama-3-70b-chat-hf
```