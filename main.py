import re
import json
import argparse
from argparse import Namespace
from pathlib import Path
from random import Random
from nltk import edit_distance
from litellm import token_counter, completion, completion_cost


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=str, nargs="+",
                        help="Models to run the test on, as taken by litellm.")
    parser.add_argument("--max-ops", type=int, default=100,
                        help="Maximum number of chained operations to test.")
    parser.add_argument("--label", type=str, default=None,
                        help="Save the results in a subfolder with this name.")
    return parser.parse_args()


def run_sequential_ops(
    seed: int, model: str, num_initial_words: int, max_words: int, num_ops: int,
    label: str = None,
) -> dict:

    model_str = model.replace("/", "-")
    save_dir = Path(__file__).parent.joinpath("results")
    if label is not None:
        save_dir = save_dir.joinpath(label)
    save_path = save_dir.joinpath(
        model_str, f"{num_initial_words}_{max_words}-{num_ops}-{seed}.json",
    )
    if save_path.exists():
        with open(save_path) as fd:
            return json.load(fd)

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
            # If too many words are used, the final result depends less on the order of
            # the operations and more on the last operations applied.
            if len(single_token_words) == max_words * 2:
                break
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

    for op_idx in range(num_ops):
        ops = ["switch"]
        if len(current_words) > max_words // 2:
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
            idx = current_words.index(w1)
            current_words[idx] = w2
            ins = f"Switch [{w1}] for [{w2}]."
        instructions.append(f"{op_idx + 1}. {ins}")

    init_word_list = "[" + ";".join(init_words) + "]"
    instruction_list = "\n".join(instructions)
    context = [dict(role="user", content=(
        f"Here is a list of words:\n\n{init_word_list}\n\nRespond with the list of "
        f"words that results after applying the following operations in order:\n\n"
        f"{instruction_list}\n\nRespond only with the list of words. Use the same "
        f"format as in the original list of words."
    ))]

    response_obj = completion(
        model, messages=context, temperature=0, max_tokens=2 + 2 * max_words,
    )
    response = response_obj.choices[0].message.content
    # Mistral tends to partially ignore the format and use "," instead of ";"
    response = response.replace(",", ";")

    i = response.find("[")
    j = response.find("]", i + 1)
    reply_word_list = response[i:j + 1] if 0 <= i < j else ""

    m = re.match(r"^\[ *\w+ *(; *\w+ *)*]$", reply_word_list)
    reply_words = list()
    if m is not None:
        reply_words = [w.strip() for w in reply_word_list.strip("[]").split(";")]

    target_chain = "".join(word_to_char[word] for word in current_words)
    reply_chain = "".join(
        word_to_char.get(word, chr(ord("z") + 1)) for word in reply_words
    )
    dist = edit_distance(target_chain, reply_chain)

    result = dict(
        seed=seed, model=model, num_initial_words=num_initial_words,
        max_words=max_words, num_ops=num_ops, prompt=context[0]["content"],
        response=response, reply_words=reply_words, expected=current_words,
        dist=dist, cost=completion_cost(completion_response=response_obj),
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fd:
        json.dump(result, fd)
    return result


def main(args: Namespace):
    import matplotlib.pyplot as plt

    # Run tests and collect results
    results = dict()
    for model in args.models:
        num_ops = list()
        rates = list()
        distances = list()
        errors = list()
        model_cost = 0

        for n in range(1, args.max_ops):
            num_ops.append(n)
            dist_list = list()
            for seed in range(5):
                result = run_sequential_ops(
                    seed=seed,
                    model=model,
                    num_initial_words=5,
                    max_words=8,
                    num_ops=n,
                    label=args.label,
                )
                dist = result["dist"]
                print(f"num_ops={n}; seed={seed}; dist={dist}")
                dist_list.append(dist)
                model_cost += result["cost"]
            rate = sum(d == 0 for d in dist_list) / len(dist_list)
            rates.append(rate)
            avg_dist = sum(dist_list) / len(dist_list)
            var = [(s - avg_dist) ** 2 for s in dist_list]
            var = sum(var) / len(var)
            distances.append(avg_dist)
            errors.append(var ** 0.5)

        print(f"Total cost for model {model}: ${model_cost:.2f}")
        results[model] = dict(
            num_ops=num_ops, rates=rates, distances=distances, errors=errors,
        )

    # Plot figures
    plt.figure()
    for model in args.models:
        r = results[model]
        plot = plt.plot(r["num_ops"], r["distances"], label=model)
        upper_errors = [d + e for d, e in zip(r["distances"], r["errors"])]
        lower_errors = [max(d - e, 0) for d, e in zip(r["distances"], r["errors"])]
        plt.fill_between(
            r["num_ops"], upper_errors, lower_errors,
            color=plot[0].get_color(), alpha=0.2,
        )
    plt.xlabel("Number of sequential operations")
    plt.ylabel("L-distance to reference")
    plt.legend()
    plt.show()

    plt.figure()
    for model in args.models:
        r = results[model]
        plt.fill_between(r["num_ops"], r["rates"], [0] * len(r["rates"]), alpha=0.2, label=model)
    plt.xlabel("Number of sequential operations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(get_args())
