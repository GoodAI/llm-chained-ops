# Testing the logical reasoning capabilities of LLMs

LLMs have improved a lot in the last few years. They are now capable of performing harder tasks, taking longer inputs, including other types of data… LLMs have become ubiquitous, and even your neighbours talk about them. Yet one thing is very unlikely to change: **we need to tell them what we want them to do**. For LLMs, this means writing prompts.

Entire books have been written about prompting techniques, which are an indispensable tool to grasp if one wants their wishes to be considered by the model. However, this scenario is not ideal. **A perfect language model should be able to understand any set of instructions that a human would**, but current implementations are still far from reaching that point. Imprecision in the attention mechanism, hallucinations, and interferences between different pieces of information are just some of the most recurring hurdles of current LLMs.

Right now, many benchmarks focus on evaluating the performance on tasks that require the LLM to ingest huge amounts of text and find key information in it. This is understandable, since such long contexts are a newly added feature and everyone is eager to find out how useful it actually is, and whether they can fit all the information they need, perform in-context learning, etc. However, **quite often LLMs are just incapable of making sense of extremely short and straightforward prompts**. During our efforts in [developing agents with Long-Term Memory (LTM)](https://github.com/GoodAI/goodai-ltm-benchmark), as well as in [integrating these agents in videogames](https://www.aipeoplegame.com/), we have found the **logical reasoning capabilities of LLMs to be one of the major pitfalls**.

If you have worked with LLMs yourself, you have surely faced the following situation: you want the LLM to perform a simple task (or so you think), so you write down a description of what the task is about, and enumerate a set of simple rules that the LLM has to follow in order to produce a reasonable output. And here is where you spend hours or days tweaking the prompt and finding absurdly verbose ways of explaining your intentions, because the LLM ignores some rules, interprets them in unexpected ways, or simply seems to not understand what you are talking about.

Transformer-based LLMs are composed of several layers, each of them performing a combination of cross-attention and transformation of the representation vectors. The more layers a model has, the more times it can transform the input sequence, and the more reasoning steps it can perform over it. Surely the LLM will have learned a good amount of shortcuts from frequent cases in the training data (just like chess players memorize plays), but that won’t work when facing unseen data. The underlying question is **how much can it do in one shot?** The LLM’s internal reasoning capability will ultimately **bottleneck what concepts the LLM can grasp, and how many interlinked rules it can apply at one time**.

## Do the switch-switch

This motivated us to design a test to evaluate **how many sequentially-dependent operations an LLM can perform in one shot**. Interestingly, we have found that the results correlate very well with the expected reasoning skills of each model, and we leverage the test to assess any new LLM before even attempting to integrate it in any of our systems.

The task consists in applying a series of transformations to a list of words. The LLM is given an initial list of five words, and then it is asked to perform a number of operations on the list, each of which switches one word in the list for another.

`Here is a list of words:`

`[rush;night;stone;chain;testing]`

`Respond with the list of words that results after applying the following operations in order:`

`1. Switch [rush] for [ghost].`  
`2. Switch [testing] for [population].`  
`3. Switch [ghost] for [rush].`  
`4. Switch [stone] for [ghost].`

`Respond only with the list of words. Use the same format as in the original list of words.`

The resulting prompts are very short and clear, and we carefully select the formatting and words so that: 1\) words are common; 2\) each of the words is represented as a single token. This way we isolate the internal reasoning mechanism from other phenomena like tokenization or [lost-in-the-middle](https://arxiv.org/abs/2307.03172) effects. In the case above, we would expect the list `[rush;night;ghost;chain;population]` as the final output.

## Results

Ranging from a single switch operation to 30, we perform 10 runs of each configuration and measure how often each LLM produces the right response. Additionally, and to further isolate the internal reasoning capabilities, we are not too strict about the formatting of the output. This is not an issue with the more performant LLMs, but the weaker ones do sometimes make formatting mistakes.

![Results of the switch-switch test](images/results1.png)

Interestingly, this otherwise very simple task turns out to be very challenging for the LLMs. Most of them start making mistakes as soon as 2 operations are given, and very few achieve a reasonable accuracy after 5 operations. The best LLM is Claude 3.5 Sonnet, and yet it cannot make it past the 10 operations mark without making some mistakes. We also observe that the *old* GPT-4 and GPT-4 turbo still hold up quite well in front of their more recent 4o alternatives.

One detail that caught our attention is a sudden performance dip which is present at the 4 operations point for 5 of all models tested. Upon closer inspection, this scenario seems to be related to a higher-than-usual occurrence of word reinstatements. It looks like most models have difficulty dealing with situations where a word is replaced, but later on brought back to the list. The bump is quite severe for the old GPT-4s.

## Conclusions

These results reveal a significant weak point of current LLMs, which struggle with prompts that require sequential thinking. While LLMs offer input lengths of many thousands of tokens, our tests require less than 340 tokens of input. These results have important implications for the understanding of how LLMs are limited in terms of logical thinking, and also offer a quick means for estimating an LLM’s reasoning capabilities, which should reflect on its ability to generalize to novel scenarios, for which it hasn’t learned reasoning shortcuts.

In future work, we want to analyze in detail the effects of compounding errors like the one discussed before, identifying cases that pose special challenges to LLMs. We also want to explore larger prompts, to see how accuracy ratios are affected by the prompt length.

You can find the code for these experiments in our [GitHub repository](https://github.com/GoodAI/llm-chained-ops), and the results shown here in [this link](https://drive.google.com/file/d/1LZEeqkuBjjDaIm-WkQ6fJ6p1LzFeEy66/view?usp=sharing).
