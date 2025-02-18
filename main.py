import re
import json
import time
import argparse
from argparse import Namespace
from pathlib import Path
from random import Random
from nltk import edit_distance
from litellm import token_counter, completion, completion_cost, ModelResponse


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=str, nargs="+",
                        help="Models to run the test on, as taken by litellm.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ops", type=int, default=30,
                        help="Maximum number of chained operations to test.")
    parser.add_argument("--reps", type=int, default=10,
                        help="Number of repetitions per configuration.")
    parser.add_argument("--label", type=str, default=None,
                        help="Save the results in a subfolder with this name.")
    parser.add_argument("--variable-len", action="store_true",
                        help="Allow operations that alter the list's length.")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Default temperature for the LLM calls.")
    return parser.parse_args()


def warn_once(warning: str) -> bool:
    if not hasattr(warn_once, "warned"):
        warn_once.warned = set()
    warned = False
    if warning not in warn_once.warned:
        print("WARNING:", warning)
        warn_once.warned.add(warning)
        warned = True
    return warned


def safe_completion(
    model: str, max_retries=3, retry_delay=5, **kwargs,
) -> ModelResponse:
    for r in range(max_retries):
        try:
            return completion(model, **kwargs)
        except Exception:
            if r == max_retries - 1:
                raise
            time.sleep(retry_delay)


def evaluate_response(
    response: str, expected: list[str], word_to_char: dict[str, str],
) -> tuple[int, list[str]]:
    # Mistral tends to partially ignore the format and use "," instead of ";"
    response = response.replace(",", ";")
    target_chain = "".join(word_to_char[word] for word in expected)
    dist = len(expected)
    reply_words = []
    for m in re.finditer(r"\[? *\w+ *(; *\w+ *)+]?", response):
        rwords = [w.strip() for w in m.group(0).strip("[]").split(";")]
        reply_chain = "".join(
            word_to_char.get(word, chr(ord("z") + 1)) for word in rwords
        )
        d = edit_distance(target_chain, reply_chain)
        if d < dist:
            dist = d
            reply_words = rwords
    return dist, reply_words


def run_sequential_ops(
    seed: int, model: str, num_initial_words: int, max_words: int, num_ops: int,
    variable_len: bool = False, label: str = None, temperature: float = 0,
) -> dict | None:

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
        s = f"[{w};{w};{w}]"  # 2 brackets, 2 semicolons, 3 words -> 7 tokens
        if token_counter(model, text=s) == 7:
            single_token_words.append(w)
            # If too many words are used, the final result depends less on the order of
            # the operations and more on the last operations applied.
            if len(single_token_words) == max_words * 2:
                break
            if len(single_token_words) == len(equivalence_chars):
                break

    # This dict will help us compute the Levenshtein distance at the word level.
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
        if variable_len:
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

    max_tokens = 2 + 2 * max_words
    if model.startswith("o1-"):
        warn_once("OpenAI o1 models require temperature=1 and no output limit.")
        temperature, max_tokens = 1, None
    try:
        response_obj = safe_completion(
            model, messages=context, temperature=temperature, max_tokens=max_tokens,
        )
    except Exception as exc:
        print(f"Unsuccessful completion. Error: {str(exc)}")
        return

    response = response_obj.choices[0].message.content
    dist, reply_words = evaluate_response(response, current_words, word_to_char)
    try:
        cost = completion_cost(completion_response=response_obj)
    except Exception as exc:
        if warn_once(f"Unable to calculate cost for model {model}."):
            print(f"Error: {str(exc)}")
        cost = 0

    result = dict(
        seed=seed, model=model, num_initial_words=num_initial_words,
        max_words=max_words, num_ops=num_ops, variable_len=variable_len,
        prompt=context[0]["content"], response=response, reply_words=reply_words,
        expected=current_words, dist=dist, temperature=temperature, cost=cost,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fd:
        json.dump(result, fd)
    return result


def get_label(model: str) -> str:
    label = model.split("/")[-1]
    if model.startswith("claude"):
        label = "-".join(label.split("-")[:-1])
    return label


def main(args: Namespace):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Run tests and collect results
    results = dict()
    for model in args.models:
        num_ops = list()
        rates = list()
        distances = list()
        errors = list()
        model_cost = 0
        temperature = None

        for n in range(1, args.ops + 1):
            num_ops.append(n)
            dist_list = list()
            for repetition in range(args.reps):
                seed = args.seed + repetition
                result = run_sequential_ops(
                    seed=seed,
                    model=model,
                    num_initial_words=5,
                    max_words=8,
                    num_ops=n,
                    label=args.label,
                    variable_len=args.variable_len,
                    temperature=args.temperature,
                )
                if result is None:
                    continue
                if temperature is None:
                    temperature = result["temperature"]
                assert result["temperature"] == temperature
                dist = result["dist"]
                print(f"num_ops={n}; seed={seed}; dist={dist}")
                dist_list.append(dist)
                model_cost += result["cost"] or 0
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
            temperature=temperature,
        )

    # Plot figures
    plt.figure()
    for model in args.models:
        r = results[model]
        plot = plt.plot(r["num_ops"], r["distances"], label=get_label(model))
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

    hot_ico = plt.imread("hot.png")
    cold_ico = plt.imread("cold.png")
    num_models = len(args.models)
    plt.rcParams.update({'font.size': max(5, round(10 - 0.25 * num_models))})
    colors = mpl.colormaps["tab10"].colors
    get_color = lambda i: colors[i % len(colors)]
    fig, axs = plt.subplots(num_models, 1, figsize=(6, 4 * num_models), sharex=True)
    for i, model in enumerate(args.models):
        r = results[model]
        ax = axs[i] if num_models > 1 else axs
        ax.fill_between(
            r["num_ops"], r["rates"], [0] * len(r["rates"]),
            alpha=0.5, color=get_color(i),
        )
        ax.plot(r["num_ops"], r["rates"], color=get_color(i))
        o1, o2 = r["num_ops"][0], r["num_ops"][-1]
        ax.text(o1 + 0.6 * (o2 - o1), 0.5, f"AuC = {sum(r["rates"]):.1f}")
        ax.set_ylabel("Accuracy")
        ax.set_ylim((0, 1.05))
        ax.set_xlim((o1, o2))
        ax.legend(markerfirst=False, handlelength=0, handleheight=0, handletextpad=0,
                  handles=[Patch(label=get_label(model))], loc="upper right")
        axin = ax.inset_axes([0.85, 0.4, 0.2, 0.2])
        axin.imshow(hot_ico if r["temperature"] > 0.5 else cold_ico)
        axin.axis('off')

    axs[-1].set_xlabel("Number of sequential operations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(get_args())
