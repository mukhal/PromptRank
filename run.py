from transformers import AutoModelForCausalLM, GPT2TokenizerFast, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import random
import csv, json, time
import os, sys
import sys
import torch
from argparse import ArgumentParser
sys.path.append("./path-retriever/")
from pipeline.tfidf_retriever import TfidfRetriever
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
from src.eval_utils import evaluate_retrieval
from src.prompt_utils import *
from src.data_utils import *

## set seed
def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_metrics(metrics, best=False, test=False):
    for k, v in metrics.items():
        if "count" in k or "stuff" in k:
            continue
        if best:
            k = "best " + k
        elif test:
            k = "test " + k

        if not isinstance(v, float):
            continue
        print("%s: %.4f" % (k, v))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="google/t5-small-lm-adapt")
    parser.add_argument("--demos_file", type=str, default=None)
    parser.add_argument(
        "--eval_data",
        type=str,
        default=
        "/home/khalifam/stuff/odqa/PromptRetriever/data/hotpotqa/dev1/dev1.json"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None)
    parser.add_argument("--save_retrieved_path", type=str, default=None)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=
        'Document: <P> Review previous documents and ask some question. Question:'
    )
    parser.add_argument("--ensemble_prompts", action="store_true")
    parser.add_argument("--n_ensemble_prompts", type=int, default=5)
    parser.add_argument("--instruction_template_file", type=str, default="prompt-templates/top_instructions.txt")
    parser.add_argument("--ensembling_method", type=str, default="mean")
    parser.add_argument("--demos_ids", type=str, default="0,1", help="demos ids to use")
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="conditional_plp",
        help="method used to compute relevance score",
        choices=["prompt_plp", "conditional_plp", "raw", "tfidf"])
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--n_eval_examples", type=int, default=None)
    parser.add_argument("--tfidf_pool_size", type=int, default=100)
    parser.add_argument("--top_k_second_hops",
                        type=int,
                        default=3)
    parser.add_argument("--top_k_first_hops", type=int, default=5)
    parser.add_argument("--use_raw_score", action="store_true", help="use unnormalized raw score (logits) instead of logprobs")
    parser.add_argument("--combine_hops_method", type=str, default="separate", choices=["separate", "sep-concat", "concat"], help="how to combine first and second hop prompts")
    parser.add_argument("--prepend_title",
                        action="store_true",
                        help="prepend titles to paragraphs")
    parser.add_argument("--max_doc_len", type=int, default=230)
    parser.add_argument("--demo_max_doc_len", type=int, default=100)
    parser.add_argument("--max_prompt_len", type=int, default=600)
    parser.add_argument("--demo_max_prompt_len", type=int, default=1024)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db_path", type=str, default="path-retriever/models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db")
    parser.add_argument("--tfidf_retriever_path", type=str,
    default="path-retriever/models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    parser.add_argument("--lower_case", action="store_true", help="lower case")
    parser.add_argument("--use_bm25", action="store_true", help="rerank with bm25")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reverse_path", action="store_true", help="reverse document paths")
    parser.add_argument("--truncate_demos",
                        action="store_true",
                        help="truncate demos to max_doc_len")
    parser.add_argument("--n_ensemble_demos", type=int, default=3)
    parser.add_argument("--bridge_only" , action="store_true", help="use bridge questions only")

    args = parser.parse_args()
    init_seed(args.seed)

    print("Reranking LM used: {} ".format(args.model))

    if 'gpt2' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token

    elif 't5' in args.model:
        model = T5ForConditionalGeneration.from_pretrained(args.model)
        tokenizer = T5Tokenizer.from_pretrained(args.model)

    model = model.to(args.device)
    prompt_template = args.prompt_template  # use first prompt format for now
    assert "<P>" in prompt_template, "invalid prompt format!"

    if args.demos_file: ## append in-context demos to prompt
        print("Loading in-context demos from {}".format(args.demos_file))

        with open(args.demos_file, "r") as f:
            demos = json.load(f)

        if args.truncate_demos:
            for d in demos:
                d["gold_path_texts"] = [shorten_paragraph_if_necessary(p, args.demo_max_doc_len, tokenizer=tokenizer) for p in d["gold_path_texts"]]

        if args.ensemble_prompts:
            print("Ensemble in-context demos!")
            ## sample
            prompt_template_ensemble = []
            even_ids = [i for i in range(0, len(demos), 2)] # for comparison questions
            odd_ids = [i for i in range(1, len(demos), 2)] # for bridge questions

            ## sample args.n_ensemble_demos even and odd ids
            even_ids = np.random.choice(even_ids, args.n_ensemble_demos, replace=False)
            odd_ids = np.random.choice(odd_ids, args.n_ensemble_demos, replace=False)
            ids = zip(even_ids, odd_ids)

            for ids_tuple in ids:
                exs = []
                print(ids_tuple)
                for i in ids_tuple:
                    exs.append(create_prompt(demos[i]["gold_path_texts"],
                                        prompt_template=prompt_template
                                        ) + " " + demos[i]["question"])
                t = " ".join(exs) + " " + prompt_template
                prompt_template_ensemble.append(t)

            prompt_template = prompt_template_ensemble

        else:
            print("Using in-context demonstrations with ids {} -- no ensemble".format(args.demos_ids))
            ids = [int(i) for i in args.demos_ids.split(",")]
            demos = [demos[i] for i in ids]
            examples = []
            for d in demos:
                example = create_prompt(d["gold_path_texts"],
                                        prompt_template=prompt_template
                                        ) + " " + d["question"]
                examples.append(example)
            prompt_template = " ".join(examples) + " " + prompt_template

        setattr(args, "max_prompt_len", args.demo_max_prompt_len)

    elif args.ensemble_prompts:
        ## read remplate from args.prompt_template_file 
        with open(args.instruction_template_file, "r") as f:
            prompt_template = [l.strip() for l in f.readlines()]
        print("Ensembling instructions -- no in-context demos used...")

    if isinstance(prompt_template, list):
        print("*" * 50)
        print("Instruction templates used: ", "\n".join(prompt_template))
        print("*" * 50)
    else:
        print("Instruction template used: ", prompt_template)
    setattr(args, "prompt_template", prompt_template)

    db_path = args.db_path
    tfidf_retriever = TfidfRetriever(
        db_path,
        args.tfidf_retriever_path,
    )

    if args.eval_data.endswith('.json'):
        with open(args.eval_data) as f:
            eval_data = json.load(f)

    metrics_docs = evaluate_retrieval(args, model,
    eval_data,
    tokenizer,
    tfidf_retriever=tfidf_retriever)
    print(json.dumps(metrics_docs, indent=4))

    if args.save_retrieved_path:
        with open(args.save_retrieved_path, "w") as f:
            json.dump(metrics_docs, f)

        with open(args.save_retrieved_path[:-5] + ".args", "w") as f:
            json.dump(vars(args), f)

