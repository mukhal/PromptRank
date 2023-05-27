from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler, RandomSampler
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from nltk.tokenize import sent_tokenize
import json
from .prompt_utils import create_prompt, create_qa_prompt, shorten_paragraph_if_necessary, prepend_title_doc, trim_passages_for_reader
import torch

class PromptRetrievalDataset(Dataset):
    def __init__(
        self,
        processed_data,
        max_prompt_len,
        tokenizer,
        n_negs=4,
    ):

        self.tokenizer = tokenizer
        self.max_len = max_prompt_len
        self.n_negs = n_negs

        print("tokenizing data...")

        assert "T5" in tokenizer.__class__.__name__, "finetuning only supported for t5"
        for i, example in tqdm(enumerate(processed_data)):

            pos_prompt = example["pos_prompt"]
            neg_prompts = example["neg_prompts"]

            pos_input_ids = self.tokenizer(
                pos_prompt,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
            )

            example["pos_input_ids"] = pos_input_ids["input_ids"][0]
            example["pos_attn_masks"] = pos_input_ids["attention_mask"][0]

            example["question_input_ids"] = self.tokenizer(
                [example["question"]],
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
            )["input_ids"][0]

            ## easier negatives: prompts with irrelevant documents
            try:
                neg_input_ids = self.tokenizer(
                    neg_prompts,
                    return_tensors="pt",
                    max_length=self.max_len,
                    truncation=True,
                    padding="max_length",
                )
            except:
                import ipdb; ipdb.set_trace()
            example["neg_input_ids"] = neg_input_ids["input_ids"]
            example["neg_attn_masks"] = neg_input_ids["attention_mask"]

        self.examples = processed_data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        pos_ids = self.examples[idx]["pos_input_ids"]
        pos_attn_masks = self.examples[idx]["pos_attn_masks"]
        neg_ids = self.examples[idx]["neg_input_ids"]
        neg_attn_masks = self.examples[idx]["neg_attn_masks"]

        ## sample a set of negatives
        if len(neg_ids) >= self.n_negs:
            neg_idxs = np.random.choice(len(neg_ids), self.n_negs, replace=False)

        else: # only sample with replacement if you have fewer negatives than n_negs
            neg_idxs = np.random.choice(len(neg_ids), self.n_negs, replace=True)

        #neg_idxs = np.array(range(self.n_negs)) # for debugging

        neg_ids = [neg_ids[i] for i in neg_idxs]
        neg_attn_masks = [neg_attn_masks[i] for i in neg_idxs]

        hop = self.examples[idx]["hop"]
        return {
            "pos_input_ids": pos_ids,
            "pos_attn_masks": pos_attn_masks,
            "neg_input_ids": neg_ids,
            "neg_attn_masks": neg_attn_masks,
            "pos_p": self.examples[idx]["gold_path_texts"][hop - 1],
            "neg_p": [self.examples[idx]["negative_p"][i] for i in neg_idxs],
            "question_input_ids": self.examples[idx]["question_input_ids"],
        }

class PromptQADataset(Dataset):
    def __init__(
        self,
        processed_data,
        max_prompt_len,
        tokenizer,
    ):

        self.tokenizer = tokenizer
        self.max_len = max_prompt_len

        print("tokenizing data...")
        for _, example in tqdm(enumerate(processed_data)):
            question = example["question"]
            answer = example["answer"]
            prompt = example["prompt"]

            prompt_with_answer = prompt + " " + answer
            input_ids = self.tokenizer(
                prompt_with_answer,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding="max_length",

            )

            len_prompt_encode = len(self.tokenizer.encode(prompt))
            len_answer_encode = len(self.tokenizer.encode(' ' + answer))
            labels = input_ids["input_ids"][0].clone()

            labels[:len_prompt_encode] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            try:
                labels[len_prompt_encode + len_answer_encode] = tokenizer.eos_token_id # eos_token_id to mark answer end
            except:
                ValueError("prompt + answer are too long!")

            example["input_ids"] = input_ids["input_ids"][0]
            example["attn_masks"] = input_ids["attention_mask"][0]
            example["labels"] = labels

        self.examples = processed_data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]["input_ids"]
        attn_masks = self.examples[idx]["attn_masks"]
        labels = self.examples[idx]["labels"]

        return {
            "input_ids": ids,
            "attn_masks": attn_masks,
            "labels": labels,
        }

def process_data(args,
                json_data,
                tokenizer,
                retriever=None):
    """
    creates hops for training -- both positive and negative depending on the passed arguments
    """
    negative_method = args.negative_method
    combine_hops_method = args.combine_hops_method
    prompt_template = getattr(args, "prompt_template", getattr(args, "reranker_prompt_template", None))

    processed_data = []

    for example in json_data:
        question = example["question"]

        if not question.endswith("?"):
            question += "?"
            example["question"] = question

        gold_path = example["gold_path_texts"]
        gold_path_titles = example["gold_path_titles"]
        supporting_facts = example["supporting_facts"]
        tfidf_p = example[
                        "negative_paragraphs_tfidf_texts"]
        tfidf_t = example["negative_paragraphs_tfidf_titles"]

        ## prepend titles to all docs if necessary
        if args.prepend_title:
            ## append to gold
            for i,  (title, text) in enumerate(zip(gold_path_titles, gold_path)):
                if args.lower_case:
                    text = text.lower()

                gold_path[i] = prepend_title_doc(text, title)

            ## append to tfidf negatives
            for i, (title, text) in enumerate(zip(tfidf_t, tfidf_p)):
                if args.lower_case:
                    text = text.lower()
                tfidf_p[i] = prepend_title_doc(text, title)

        if combine_hops_method in ['separate', 'sep-concat']: # each hop in a separate prompt
            ## separate first hop prompts
            for i, (gold_p_title,
                    gold_p_text) in enumerate(zip(gold_path_titles,
                                                  gold_path)):

                neg_prompts = []
                example["negative_p"] = []

                gold_p_text = shorten_paragraph_if_necessary(gold_p_text, args.max_doc_len, tokenizer)
                pos_prompt = create_prompt(gold_p_text,
                                           prompt_template=prompt_template)

                if negative_method in ['sf_tfidf', 'tfidf', 'tfidf_links']:
                    tfidf_p = [shorten_paragraph_if_necessary(p, args.max_doc_len, tokenizer)
                    for p in tfidf_p]

                    for neg_p in tfidf_p[:args.n_negs]:
                        prompt = create_prompt(neg_p,
                                      prompt_template=prompt_template)
                        neg_prompts.append(prompt)
                        example["negative_p"].append(neg_p)

                    if 'sf' in negative_method:
                        # remove supporting facts from negative prompts
                        sf_idx = defaultdict(list)
                        for title, idx in supporting_facts:
                            sf_idx[title].append(idx + (1 if args.prepend_title else 0)) ## TODO needs checking

                        ## remove supporting fact sentence from positive paragraph to create negative paragraph
                        sf_sent_indices = sf_idx[gold_p_title]
                        sent_tokenized = sent_tokenize(gold_p_text)
                        neg_p = " ".join(sent_tokenize[idx]
                                        for idx, _ in enumerate(sent_tokenized)
                                        if idx not in sf_sent_indices)

                        neg_prompts.append(create_prompt(neg_p,
                                        prompt_template=prompt_template))

                        example["negative_p"].append(neg_p)

                assert len(example["negative_p"]) == len(neg_prompts)

                d = {
                    "pos_prompt": pos_prompt,
                    "neg_prompts": neg_prompts,
                    "hop": i + 1
                }
                d.update({
                    k: v
                    for k, v in example.items() if k not in [
                        'distractor_context', 'negative_paragraphs_tfidf',
                        'additional_context'
                    ]
                })

                processed_data.append(d)

        if combine_hops_method in ['sep-concat', 'concat']: # add multi-hop prompts if needed

            if "negative_p" not in example:
                example["negative_p"] = []

            gold_path= [shorten_paragraph_if_necessary(p, args.max_doc_len, tokenizer) for p in gold_path]
            pos_prompt = create_prompt(gold_path, prompt_template=prompt_template)

            neg_prompts = []

            ## add negatives
            if negative_method == 'tfidf_links':
                assert retriever is not None, "need retriever to find links for negative prompts "
                ## get links from first hop
                linked_paragraphs = retriever.get_hyperlinked_abstract_paragraphs(
                    title=gold_path_titles[0] + '_0')

                ## take only n_negs
                n = 0
                for neg_p_title, neg_p_text in linked_paragraphs.items():
                    if n >= args.n_negs:
                        break

                    if neg_p_title.replace("_0", "") == gold_path_titles[1]:
                        continue # skip if same as positive 2nd hop

                    n += 1
                    neg_p_text = shorten_paragraph_if_necessary(neg_p_text, args.max_doc_len, tokenizer)
                    if args.prepend_title:
                        neg_p_text = prepend_title_doc(neg_p_text, neg_p_title.replace("_0", ""))

                    ## sample random tfidf negative from the top 10
                    idx = np.random.choice(range(len(tfidf_p[:args.n_negs])), 1)[0]

                    #neg_first_hop = tfidf_p[idx]
                    neg_prompt = create_prompt([gold_path[0], neg_p_text],
                                                prompt_template=prompt_template)

                    neg_prompts.append(neg_prompt)
                    example["negative_p"].append(neg_p_text)

                if not neg_prompts:
                    ## insert a single placeholder negative prompt
                    #TODO maybe insert fill the reamining with tfidf negs
                    #neg_prompts.append(" Question:")
                    # insert tfidf neg instead
                    print("Found no linked paragraphs for {}".format(question))
                    print("Inserting tfidf negatives instead")
                    idx = np.random.choice(range(len(tfidf_p[:args.n_negs])),
                                           1)[0]

                    neg_prompt = create_prompt([gold_path[0], tfidf_p[idx]],
                                               prompt_template=prompt_template)

                    neg_prompts.append(neg_prompt)
                    example["negative_p"].append(neg_p_text)

            ## replace 2nd hop with tfidf neg -- should we replace only the 2nd hop? or all?
            elif negative_method == 'tfidf':
                for neg_p in tfidf_p[:args.n_negs]:
                    if args.prepend_title:
                        neg_p = prepend_title_doc(neg_p, tfidf_t[i])
                    neg_prompt = create_prompt([gold_path[0], neg_p],
                                            prompt_template=prompt_template)

                    neg_prompts.append(neg_prompt)
                    example["negative_p"].append(neg_p)

                    #neg_prompt2 = create_prompt(neg_p,
                    #    paragraph_2=gold_path[1],
                    #    prompt_template=prompt_template)

                    #neg_prompts.append(neg_prompt2)
                    #example["negative_p"].append(neg_p)

            dd = {
                    "pos_prompt": pos_prompt,
                    "neg_prompts": neg_prompts, # TODO negatives if needed
                    "hop": -1 # -1 means concat all hops
                }

            dd.update({
                k: v
                for k, v in example.items() if k not in [
                    'distractor_context', 'negative_paragraphs_tfidf',
                    'additional_context'
                ]
            })

            processed_data.append(dd)

    # return a dictionary of pos data, neg data, question to pos paragraphs
    return processed_data


def process_qa_data(args,
                data,
                tokenizer):
    """
    Loads QA examples (Gold Passages, question) from and prepares prompts for training.
    """

    n_examples = args.n_train_examples
    json_data = data
    prompt_template = getattr(args, "prompt_template", getattr(args, "reader_prompt_template", None))

    if n_examples is not None:
        if args.rand_train_data:
            print("randomly sampling {} examples".format(n_examples))
            json_data = np.random.choice(json_data, n_examples, replace=False)
        else:
            json_data = json_data[:n_examples] # take the first n_examples

    processed_data = []

    for example in json_data:
        question = example["question"]
        gold_path = example["gold_path_texts"]
        answer = example["answer"]

        if len(answer.split()) > 10:
            print("trimming too long answer in example: {}".format(example["_id"]))
            answer = answer.split()[:10]
            answer = " ".join(answer)
            example["answer"] = answer

        qid = example["_id"]
        gold_path = [shorten_paragraph_if_necessary(p, args.max_doc_len, tokenizer) for p in gold_path]

        prompt = create_qa_prompt(question,
                                        gold_path,
                                        prompt_template=prompt_template)

        d = {"prompt": prompt,
            "answer": answer}

        d.update({ k: v for k, v in example.items() if k not in [
                        'distractor_context', 'negative_paragraphs_tfidf',
                        'additional_context'
                    ]})

        processed_data.append(d)

        if args.add_negative_paragraphs:
            ## train model with negative paragraphs to reduce exposure bias
            N = args.tfidf_neg_examples
            neg_p = example["negative_paragraphs_tfidf_texts"][:N]
            all_p = neg_p[::-1] + gold_path # reverse sort using score

            passages = trim_passages_for_reader(question, all_p,
                args.max_prompt_len, args.max_doc_len, tokenizer)
            prompt = create_qa_prompt(question, passages, prompt_template=prompt_template)
            d = {"prompt": prompt,
                "answer": answer}

            d.update({ k: v for k, v in example.items() if k not in [
                    'distractor_context', 'negative_paragraphs_tfidf',
                    'additional_context'
                ]})

            processed_data.append(d)

            ## shuffle passages for more robust reader (data augmentation, basically)
            n_shuffle = args.neg_shuffle_times
            for _ in range(n_shuffle):
                np.random.shuffle(all_p)
                passages = trim_passages_for_reader(question, all_p,
                                                    args.max_prompt_len,
                                                    args.max_doc_len,
                                                    tokenizer)
                prompt = create_qa_prompt(question,
                                        passages,
                                        prompt_template=prompt_template)
                d = {"prompt": prompt, "answer": answer}

                d.update({
                    k: v
                    for k, v in example.items() if k not in [
                        'distractor_context', 'negative_paragraphs_tfidf',
                        'additional_context'
                    ]
                })
                processed_data.append(d)

    return processed_data

class ConcatBatchSampler(Sampler):
    '''
    Takes a ConcatDataset and samples from it a batch from one dataset at a time alternatively
    '''

    def __init__(self, dataset, batch_size_1, batch_size_2):
        self.dataset = dataset
        self.batch_size_1 = batch_size_1
        self.batch_size_2 = batch_size_2
        self.cur_dataset = 0 # dataset to sample from at the current time

    def __iter__(self):
        n_iter = len(self.dataset.datasets[0]) // self.batch_size_1 + len(self.dataset.datasets[1]) // self.batch_size_2
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        batch_size = self.batch_size_1 if self.cur_dataset == 0 else self.batch_size_2

        for _ in range(n_iter):
            size_cur_dataset = len(self.dataset.datasets[self.cur_dataset])
            indices = torch.randperm(size_cur_dataset, generator=generator)

            if self.cur_dataset == 1:
                indices += len(self.dataset.datasets[0])

            yield indices[:batch_size]
            self.cur_dataset = 1 - self.cur_dataset

    def __len__(self):
        return len(self.dataset)


def convert_nq_to_hotpot(data):

    new_data = []
    for i, example in enumerate(data):

        question = example["question"]
        if not question.endswith("?"):
            question += "?"

        answer = example["answers"][0]
        pos_ctxs = example["positive_ctxs"]
        neg_ctxs = example["negative_ctxs"]

        gold_path_texts = [t["text"] + ('. ' if not t["text"].strip().endswith('.') else '') for t in pos_ctxs]
        gold_path_titles = [t["title"] for t in pos_ctxs]

        neg_paragraph_tfidf_titles = [t["title"] for t in neg_ctxs]
        neg_paragraph_tfidf_texts = [
            t["text"] + ('. ' if not t["text"].strip().endswith('.') else '')
            for t in neg_ctxs
        ]

        qid = i + 1
        new_data.append({
            "question": question,
            "answer": answer,
            "gold_path_texts": gold_path_texts,
            "gold_path_titles": gold_path_titles,
            "negative_paragraphs_tfidf_texts": neg_paragraph_tfidf_texts,
            "negative_paragraphs_tfidf_titles": neg_paragraph_tfidf_titles,
            "_id": qid,
            "supporting_facts": None,
            "type": "bridge"
        })

    return new_data


def convert_fever_to_hotpot(data, retriever):

    new_data = []
    for i, example in enumerate(data):

        question = example["claim"]
        #if not question.endswith("?"):
        #    question += "?"

        print(question)

        answer = example["label"]

        gold_path_titles = []
        for doc in example["evidence"]:
            for ev in doc:
                gold_path_titles.append(ev[2])

        gold_path_titles = list(set([t for t in gold_path_titles]))
        assert len(gold_path_titles) >= 2, "Not enough supporting facts"

        if len(gold_path_titles) == 2: # only use 2-hop claims

            gold_path_texts = [get_text_from_title(retriever, t) for t in gold_path_titles]
            qid = i + 1
            new_data.append({
                "question": question,
                "answer": answer,
                "gold_path_texts": gold_path_texts,
                "gold_path_titles": gold_path_titles,
                "_id": qid,
                "supporting_facts": None,
                "type": "claim"
            })

    return new_data


def convert_beerqa_to_hotpot(data, retriever):
    new_data = []
    data = data["data"]
    for i, example in enumerate(data):

        question = example["question"]
        if not question.endswith("?"):
            question += "?"

        if "answers" in example:
            answer = example["answers"][0]
            
            title_to_text = defaultdict(lambda : '')
            for ti, tx in example["context"]:
                title_to_text[ti] = (title_to_text[ti] + ' ' + tx).strip()
            
            gold_path_titles = list(title_to_text.keys())
            gold_path_texts = list(title_to_text.values())
            source = example["src"]
        
        else:
            answer = "TEST"
            gold_path_titles = ["TEST"]
            gold_path_texts = ["TEST"]
            source = "TEST"
        
        id = example['id']

        new_data.append({
            "question": question,
            "answer": answer,
            "gold_path_texts": gold_path_texts,
            "gold_path_titles": gold_path_titles,
            "_id": id,
            "supporting_facts": None,
            "type": "UNKNOWN",
            "source": source
        })

    return new_data


def get_text_from_title(retriever, title):
    """
    Get the text from the title of the article.
    """
    t = list(retriever.load_abstract_para_text([title]).values())[0]
    if not isinstance(t, str):
        return ""

    return t