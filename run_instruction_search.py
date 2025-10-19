import json
import os
import random
import sys
import torch
import numpy as np
from argparse import ArgumentParser
import logging

from transformers import AutoModelForCausalLM, GPT2TokenizerFast, T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration

from pipeline.tfidf_retriever import TfidfRetriever
from src.eval_utils import compute_metrics
from src.prompt_utils import *
from src.data_utils import *
from src.promptrank import PromptRank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("transformers.tokenization_utils_base").setLevel(
    logging.ERROR)


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
    parser.add_argument("--model", type=str, default="google/t5-large-lm-adapt")
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        default=None
    )
    parser.add_argument("--ensemble_prompts", action="store_true")
    parser.add_argument("--ensembling_method", type=str, default="mean", choices=["mean", "max", "token"])
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="conditional_plp",
        help="method used to compute relevance score",
        choices=["prompt_plp", "conditional_plp", "raw", "tfidf"])
    parser.add_argument("--eval_mode",
                        type=str,
                        default="hotpot",
                        choices=["hotpot", "nq"],
                        help="eval model")
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument("--n_eval_examples", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tfidf_pool_size", type=int, default=40)
    parser.add_argument("--top_k_second_hops", type=int, default=3)
    parser.add_argument("--top_k_first_hops", type=int, default=5)
    parser.add_argument(
        "--use_raw_score",
        action="store_true",
        help="use unnormalized raw score (logits) instead of logprobs")
    parser.add_argument("--combine_hops_method",
                        type=str,
                        default="separate",
                        choices=["separate", "sep-concat", "concat"])
    parser.add_argument("--prepend_title",
                        action="store_true",
                        help="prepend titles to paragraphs")
    parser.add_argument("--max_doc_len", type=int, default=230)
    parser.add_argument("--max_prompt_len", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db_path", type=str, default="path-retriever/models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db")
    parser.add_argument("--tfidf_retriever_path", type=str, default="path-retriever/models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz")
    parser.add_argument("--lower_case", action="store_true", help="lower case")
    parser.add_argument("--use_bm25",
                        action="store_true",
                        help="rerank with bm25")
    parser.add_argument("--metric_out_path", type=str, required=True)

    args = parser.parse_args()
    init_seed(args.seed)

    ws_dir = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/data"

    print("Scoring Model used ====> {} ".format(args.model))

    if 'gpt2' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                     cache_dir=os.path.join(
                                                         ws_dir, "cache"))
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model,
                                                      cache_dir=os.path.join(
                                                          ws_dir, "cache"))
        tokenizer.pad_token = tokenizer.eos_token

    elif 't5' in args.model:
        model = T5ForConditionalGeneration.from_pretrained(
            args.model, cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer = T5Tokenizer.from_pretrained(args.model,
                                                cache_dir=os.path.join(
                                                    ws_dir, "cache"))

    elif 'bart' in args.model:
        model = BartForConditionalGeneration.from_pretrained(
            args.model, cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer = BartTokenizer.from_pretrained(args.model,
                                                  cache_dir=os.path.join(
                                                      ws_dir, "cache"))

    model = model.to(args.device)

    db_path = args.db_path
    tfidf_retriever = TfidfRetriever(
        db_path,
        args.tfidf_retriever_path,
    )

    MAX_TEMPLATES = 200
    ## open prmopt tempalte file
    with open(args.prompt_template_file, 'r') as f:
        prompt_templates = [l.replace("Task:","").strip() for l in f.readlines()]
        for i, t in enumerate(prompt_templates):
            if not t.endswith("Question:"):
                prompt_templates[i] = t + " Question:"
        ## adding baseline template
        #prompt_templates.insert(
        #    0,
        #    "Read the following documents and answer the question. Document: <P> Question:"
        #)
        prompt_templates = prompt_templates[:MAX_TEMPLATES]
        print ("Number of prompt templates = {}".format(len(prompt_templates)))

    print("Starting search...")

    best_rec_2 = 0
    best_template = None
    all_metrics = {}

    #prompt_templates = ["<P>"] + prompt_templates
    for i, template in enumerate(prompt_templates):
        print("***********************************************************")
        print("[{}] Template: {}".format(i+1, template))
        setattr(args, "prompt_template", template)
        assert "<P>" in template, "invalid prompt format!"

        if ".json" in args.val_data:
            with open(args.val_data) as f:
                val_data = json.load(f)

            if args.eval_mode == "nq":
                # convert to hotpot format
                val_data = convert_nq_to_hotpot(val_data)

        # Limit evaluation data if specified
        if args.n_eval_examples is not None:
            val_data = val_data[:args.n_eval_examples]

        # Create PromptRank instance
        ranker = PromptRank(tokenizer=tokenizer, 
                          tfidf_retriever=tfidf_retriever, 
                          model=model)

        # Retrieve documents using PromptRank
        retrieved_docs = ranker.retrieve(questions=val_data, args=args)

        # Compute metrics
        metrics = compute_metrics(retrieved_docs)

        rec_2 = metrics["recall@2"]
        print("REC@2: {}".format(round(rec_2, 2)))
        if rec_2 > best_rec_2:
            print("New best REC@2!")
            best_rec_2 = rec_2
            best_template = template

        all_metrics[template] = metrics
    
    print("Best template: {}".format(best_template))
    print("Best REC@2: {}".format(best_rec_2))

    ## save all metrics to json file
    with open(args.metric_out_path, 'w') as f:
        json.dump(all_metrics, f)
