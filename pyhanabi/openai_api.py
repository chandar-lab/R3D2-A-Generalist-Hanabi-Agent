#%%
import numpy as np
import pickle


def get_expected_gtruth(hist, move, mode):
    assert mode in ['color', 'rank']
    if f'Reveal player +1 {mode}' in hist:
        revealed_cards = hist.split("reveal ")[-1]
        assert revealed_cards.endswith(">")
        revealed_cards = map(int, revealed_cards[:-1].split(","))
        revealed_cards = [c for c in revealed_cards]
        if "(Play" not in move:
            return 0
        for revealed_card in revealed_cards:
            if str(revealed_card) in move:
                return 1
        return 0
    else:
        return int(f'Reveal player +1 {mode}' in move)


def generate_prompts(
    meta_prompt, hist2lang, move2lang, mode, include=None
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    prompts: dict[str, dict[str, int]] = {}
    hist_move_prompts: dict[str, dict[str, str]] = {}
    for hist, hist_lang in hist2lang.items():
        hist_move_prompts[hist] = {}

        if include is not None and include not in hist:
            continue

        for move, move_lang in move2lang.items():
            prompt = meta_prompt.format(prev_action=hist_lang, curr_action=move_lang)
            hist_move_prompts[hist][move] = prompt
            gtruth = get_expected_gtruth(hist, move, mode)
            prompts[prompt] = {"gtruth": gtruth}
    return prompts, hist_move_prompts


##########
# step1: load the action, observation generated by running gen_all_langs.py
##########
move_pkl = "exps/lang/move.pkl"
hist_pkl = "exps/lang/hist.pkl"
move2lang = pickle.load(open(move_pkl, "rb"))
hist2lang = pickle.load(open(hist_pkl, "rb"))


print(f"History Items: {len(hist2lang)}")
for k, v in hist2lang.items():
    if '<(Reveal player +1 color' in k and '<(Reveal player +1 color R' not in k:
        continue
    if '<(Reveal player +1 rank' in k and '<(Reveal player +1 rank 1' not in k:
        continue
    print(f"{k}\n\t--> {v}")
print("=" * 100)


#%%


##########
# >>> prompts tried during the development of the project <<<
# we use the meta_prompt4 and its simplfied version meta_prompt5 for our ICML paper.
# Some of the prompts are used for additional explorations in the appendix
##########


# this prompt solves color. it uses a simple prompt with an example
meta_prompt1_color = """\
Instruction: If my partner tells me the 'color' of some of my cards, I should 'play' those specific cards. Otherwise, I may 'hint color' to my partner. For example, if my partner tells me the 'rank' of my cards, I should 'hint color' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# this prompt solves rank
meta_prompt1_rank = """\
Instruction: If my partner tells me the 'rank' of some of my cards, I should 'play' those specific cards. Otherwise, I may 'hint rank' to my partner. For example, if my partner tells me the 'color' of my cards, I should 'hint rank' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# experimenting with this new one, not all correct
meta_prompt2 = """\
Instruction: If my partner tells me the 'color' of some of my cards, I should 'play' those specific cards. If my partner does something else, including telling me the 'rank' of some of my cards, I may 'hint color' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# icml appendix version, all correct
meta_prompt3 = """\
Instruction: If my partner tells me the 'color' of some of my cards, I should 'play' those specific cards. If my partner does something else, including telling me the 'rank' of some of my cards, playing or discarding their own card, I may 'hint color' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# icml version, all correct
meta_prompt4_color = """\
Instruction: If my partner tells me the 'color' of some of my cards, I should 'play' those specific cards. If my partner does something else, e.g. discards their card or tells me the 'rank' of my cards, then I may 'hint color' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# icml version, all correct
meta_prompt4_rank = """\
Instruction: If my partner tells me the 'rank' of some of my cards, I should 'play' those specific cards. If my partner does something else, e.g. discards their card or tells me the 'color' of my cards, then I may 'hint rank' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# to simplify icml version, still all correct, but less ''
meta_prompt5_rank = """\
Instruction: If my partner tells me the 'rank' of some of my cards, I should play those specific cards. If my partner does something else, e.g. discards their card or tells me the 'color' of my cards, then I may 'hint rank' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# to simplify icml version, still all correct, but less ''
meta_prompt5_color = """\
Instruction: If my partner tells me the 'color' of some of my cards, I should play those specific cards. If my partner does something else, e.g. discards their card or tells me the 'rank' of my cards, then I may 'hint color' to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# icml simplification -> 1/3852 mistake
meta_prompt6_rank = """\
Instruction: If my partner tells me the rank of some of my cards, I should play those specific cards. If my partner does something else, e.g. discards their card or tells me the color of my cards, then I should hint rank to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

# icml simplification -> 1/3852 mistake
meta_prompt6_color = """\
Instruction: If my partner tells me the color of some of my cards, I should play those specific cards. If my partner does something else, e.g. discards their card or tells me the rank of my cards, then I should hint color to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

meta_prompt7_rank = """\
Instruction: If my partner told me the rank of some of my cards, I should play those specific cards. Otherwise, I should hint rank to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""

meta_prompt7_color = """\
Instruction: If my partner told me the color of some of my cards, I should play those specific cards. Otherwise, I should hint color to my partner.

Previously: {prev_action}.

Question: Should I{curr_action}?

Answer:\
"""


##########
# step2: generate prompts. the `include` is used to generate a portion of all
# possible (observation, action) pairs so that we do not evaluate all prompt at once
##########


include = None
# include = "[null]"
# include = "<(Play"
# include = "<(Discard"
# include = "<(Reveal player +1 color R) by player 1 reveal 0"
# include = "<(Reveal player +1 rank 1) by player 1 reveal 0"
prompts, _ = generate_prompts(meta_prompt5_color, hist2lang, move2lang, "color", include=include)
print("num prompts:", len(prompts))


#%%
import os
from collections import defaultdict
import tqdm
from termcolor import cprint
import openai


##########
# step3: fill in your api_key
##########


openai.api_key = ""


def get_probs(response):
    if isinstance(response, dict) and "choices" in response:
        # openai
        all_logprobs: list[dict[str, float]] = response["choices"][0]["logprobs"]["top_logprobs"]
    elif hasattr(response, "tokens"):
        # helm
        all_logprobs = [t.top_logprobs for t in response.tokens]
    else:
        assert False

    target_probs = {"Yes": 0, "No": 0}
    for logprobs in all_logprobs:
        found = False
        for token, prob in logprobs.items():
            key = token.strip().capitalize()
            if key in target_probs:
                found = True
                target_probs[key] += np.exp(prob)
        if found:
            break
    return target_probs


def create_or_load_cache(cache_file):
    if not os.path.exists(os.path.dirname(cache_file)):
        print(f"Creating directory for {cache_file}")
        os.makedirs(os.path.dirname(cache_file))

    cache: defaultdict[str, dict] = defaultdict(dict)
    if os.path.exists(cache_file):
        print(f"loading cache from {cache_file}")
        cache = pickle.load(open(cache_file, "rb"))
    return cache


def evaluate_yes_or_no(prompts, cache_file, api_type, model="text-davinci-003", force_retry=False):
    cache = create_or_load_cache(cache_file)

    max_tokens = 2
    results = {}
    for prompt in tqdm.tqdm(prompts):
        response = cache[prompt].get(model, None)
        if response is None or force_retry:
            success = False
            max_retry = 3
            while not success and max_retry > 0:
                try:
                    if api_type == "openai":
                        response = openai.Completion.create(
                            model=model,
                            prompt=prompt,
                            temperature=0,
                            max_tokens=max_tokens,
                            logprobs=5,
                        )
                    else:
                        assert False, f"unknown api type {api_type}."
                    success = True
                except Exception as e:
                    # save computed results to cache and then quit
                    print("Error encountered, saving cache")
                    pickle.dump(cache, open(cache_file, "wb"))
                    max_retry -= 1
                    if max_retry == 0:
                        raise e
                    else:
                        print(e)

            cache[prompt][model] = response
            # print(response)
        else:
            # print(f"Getting from cache")
            pass
        results[prompt] = get_probs(response)

    pickle.dump(cache, open(cache_file, "wb"))
    return results


def check_match(prompts, results):
    all_match = True
    num_mismatch = 0
    for prompt, ret in results.items():
        if ret["Yes"] > ret["No"]:
            val = 1
        else:
            val = 0

        ref = prompts[prompt]['gtruth']
        color = ["red", "green"][int(ref == val)]
        all_match = all_match and (ref == val)
        if ref != val:
            num_mismatch += 1
            print(prompt)
            print(ret)
            print(f"ref: {ref}, pred: {val}", end=", ")
            cprint(f"match? {ref==val}", color=color)
            print("-" * 100)
    color = ["red", "green"][int(all_match)]
    cprint(f"All match? {all_match}", color=color)
    print(f"num mismatch: {num_mismatch}")


##########
# step 4: evaluate the prompts with openai-api and check the quality of the answers.
##########

cache_file = "/home/hhu/dev/cc-hanabi/pyhanabi/openai/dev_cache.pkl"
results = evaluate_yes_or_no(prompts, cache_file, "openai")
check_match(prompts, results)


#%%

##########
# step 5 (final): if the results is good, generate prior policy and dump results
# Note that we have generated prior located in `instruct-rl/pyhanabi/openai`
##########


def generate_full_prior(
    meta_inst, hist2lang, move2lang, mode
) -> dict[str, dict[str, tuple[str, float]]]:
    prompts, hist_move_prompts = generate_prompts(meta_inst, hist2lang, move2lang, mode=mode)
    results = evaluate_yes_or_no(prompts, cache_file=cache_file, api_type="helm")
    check_match(prompts, results)
    hist_move_logit: dict[str, dict[str,tuple[str, float]]] = {}
    for hist, move_prompts in hist_move_prompts.items():
        hist_move_logit[hist] = {}
        for move, prompt in move_prompts.items():
            ret = results[prompt]
            val = float((ret["Yes"] > ret["No"]))
            hist_move_logit[hist][move] = (prompt, val)
    return hist_move_logit


meta_prompt_name = "meta_prompt5_rank"
modes = {
    "meta_prompt1_color": "color",
    "meta_prompt1_rank": "rank",
    "meta_prompt2": "color",
    "meta_prompt3": "color",
    "meta_prompt4_color": "color",
    "meta_prompt4_rank": "rank",
    "meta_prompt5_color": "color",
    "meta_prompt5_rank": "rank",
    "meta_prompt6_color": "color",
    "meta_prompt6_rank": "rank",
    "meta_prompt7_rank": "rank",
    "meta_prompt7_color": "color",
}

hist_move_logit = generate_full_prior(
    eval(meta_prompt_name), hist2lang, move2lang, modes[meta_prompt_name]
)

save_dir = f"openai/{meta_prompt_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

output_path = os.path.join(save_dir, "prior.pkl")
print(f"writing to {output_path}")
pickle.dump(hist_move_logit, open(output_path, "wb"))
pickle.dump(move2lang, open(os.path.join(save_dir, "move2lang.pkl"), "wb"))
pickle.dump(hist2lang, open(os.path.join(save_dir, "hist2lang.pkl"), "wb"))
with open(os.path.join(save_dir, "prompt.txt"), "w") as f:
    f.write(eval(meta_prompt_name))
