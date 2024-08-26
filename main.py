import re
import json
import argparse
from argparse import Namespace
from pathlib import Path
from random import Random
from nltk import edit_distance
from litellm import token_counter, completion


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    return parser.parse_args()


def run_sequential_ops(
    seed: int, model: str, num_initial_words: int, max_words: int, num_ops: int,
) -> int:

    model_str = model.replace("/", "-")
    save_path = Path(__file__).parent.joinpath(
        "results", model_str, f"{num_initial_words}_{max_words}-{num_ops}-{seed}.json",
    )
    if save_path.exists():
        with open(save_path) as fd:
            return json.load(fd)["dist"]

    rnd = Random(seed)

    with open("words.txt") as fd:
        words = fd.read().splitlines()
        rnd.shuffle(words)

    # We'll use these characters later for computing the Levenshtein distance
    equivalence_chars = [chr(i) for i in range(ord(" "), ord("z") + 1)]

    # We want to focus on the LLM's abilities to handle instructions, so we want to
    # avoid any possible artifacts related to the tokenization of words. That's why
    # we'll take only words that have a dedicated token: 1 token = 1 word.
    single_token_words = list()
    for w in words:
        s = f"[{w};{w};{w}]"
        if token_counter(model, text=s) == 7:
            single_token_words.append(w)
            if len(single_token_words) == len(equivalence_chars):
                break

    word_to_char = dict()
    for word, char in zip(single_token_words, equivalence_chars):
        word_to_char[word] = char

    # Generate: initial list of words, chain of operations, and final expected list.
    init_words = rnd.sample(single_token_words, num_initial_words)
    current_words = init_words[:]
    instructions = list()

    def new_word() -> str:
        while True:
            w = rnd.choice(single_token_words)
            if w not in current_words:
                return w

    for i in range(num_ops):
        ops = ["switch"]
        if len(current_words) > 1:
            ops.append("remove")
        if len(current_words) < max_words:
            ops.append("append")
        op = rnd.choice(ops)
        if op == "append":
            w = new_word()
            current_words.append(w)
            ins = f"Append [{w}]."
        elif op == "remove":
            w = rnd.choice(current_words)
            current_words.remove(w)
            ins = f"Remove [{w}]."
        else:
            w1 = rnd.choice(current_words)
            w2 = new_word()
            i = current_words.index(w1)
            current_words[i] = w2
            ins = f"Switch [{w1}] for [{w2}]."
        instructions.append(f"{i + 1}. {ins}")

    init_word_list = "[" + ";".join(init_words) + "]"
    instruction_list = "\n".join(instructions)
    context = [dict(role="user", content=(
        f"Here is a list of words:\n\n{init_word_list}\n\nRespond with the resulting "
        f"list of words after applying the following operations in order:\n\n"
        f"{instruction_list}\n\nRespond only with the list of words, in the same format"
        f" as given."
    ))]

    response = completion(model, messages=context, temperature=0)

    i = response.find("[")
    assert i >= 0
    j = response.find("]", i + 1)
    assert j > i
    reply_word_list = response[i:j + 1]

    m = re.match(r"^\[ *\w+ *(; *\w+ *)*]$", reply_word_list)
    reply_words = list()
    if m is not None:
        reply_words = [w.strip() for w in reply_word_list.strip("[]").split(";")]

    target_chain = "".join(word_to_char[word] for word in current_words)
    reply_chain = "".join(
        word_to_char.get(word, chr(ord("z") + 1)) for word in reply_words
    )
    dist = edit_distance(target_chain, reply_chain)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fd:
        json.dump(dict(
            seed=seed, model=model, num_initial_words=num_initial_words,
            max_words=max_words, num_ops=num_ops, prompt=context[0]["content"],
            response=response, reply_words=reply_words, expected=current_words,
            dist=dist,
        ), fd)
    return dist


def main(args: Namespace):
    import matplotlib.pyplot as plt

    num_ops = list()
    rates = list()
    distances = list()
    errors = list()

    for n in range(1, 100):
        num_ops.append(n)
        dist_list = list()
        for seed in range(5):
            dist = run_sequential_ops(
                seed=seed,
                model=args.model,
                num_initial_words=5,
                max_words=8,
                num_ops=n,
            )
            print(f"num_ops={n}; seed={seed}; dist={dist}")
            dist_list.append(dist)
        rate = sum(d == 0 for d in dist_list) / len(dist_list)
        rates.append(rate)
        avg_dist = sum(dist_list) / len(dist_list)
        var = [(s - avg_dist) ** 2 for s in dist_list]
        var = sum(var) / len(var)
        distances.append(avg_dist)
        errors.append(var ** 0.5)

    plt.figure()
    plt.errorbar(num_ops, distances, yerr=errors)
    plt.xlabel("Number of sequential operations")
    plt.ylabel("L-distance to reference")
    plt.title(args.model)
    plt.show()

    plt.figure()
    plt.bar(num_ops, rates)
    plt.xlabel("Number of sequential operations")
    plt.ylabel("Accuracy")
    plt.title(args.model)
    plt.show()


if __name__ == "__main__":
    main(get_args())
