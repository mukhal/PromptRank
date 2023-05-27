from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json 

from .prompt_utils import get_question_begin_idx, get_paragraph_begin_idx, prepend_title_doc, create_prompt, shorten_paragraph_if_necessary, get_question_begin_idx_v2


def process_data(args,
                data_file,
                tokenizer):
    """
    Loads positive and negative retrieval examples from a csv file
    returns n_examples positive and n_examples negative examples
    if n_examples is None, returns all examples
    """

    prompt_format = args.prompt_template

    assert data_file.endswith(".jsonl")

    json_data = []
    
    MAX_EXAMPLES = args.n_train_examples
    with open(data_file) as f:
        cnt = 0
        for line in f:
            if cnt > MAX_EXAMPLES:
                break
            d = json.loads(line)
            if args.keep_question_sentence and d["sent_removed"]:
                continue # skip if question sentence is removed
            
            elif args.remove_question_sentence and not d["sent_removed"]:
                continue # skip if question sentence is not removed
            
            json_data.append(d)

            if args.use_sentence_as_question:
                cnt+=1
            else:
                cnt += len(d["questions"])

    processed_data = []

    for example in json_data:
        question_list = example["questions"]
        answer_list = example["answers"]
        passage = example["passage"]
        sentence = example["sentence"]
        title = example["title"]
                
        if args.lower_case:
            passage = passage.lower()
            sentence = sentence.lower()
            title = title.lower()
            question_list = [q.lower() for q in question_list]

        passage = shorten_paragraph_if_necessary(passage, args.max_doc_len, tokenizer)
        example["passage"] = passage
        example["sentence"] = sentence
        example["title"] = title
        example["questions"] = question_list

        if not args.use_sentence_as_question:
            for q, a in zip(question_list, answer_list):
                reranking_prompt = create_prompt(q,
                                            passage,
                                            prompt_format=prompt_format)
                ## create answer prompt as well 
                if args.prepend_title:
                    reranking_prompt = prepend_title_doc(reranking_prompt, title)

                d = {
                    "prompt": reranking_prompt,
                    "answer": a,
                }

                d.update(example)
                del d["questions"]
                del d["answers"]
                
                processed_data.append(d) 
        else:
            reranking_prompt = create_prompt(example["sentence"],
                                        passage,
                                        prompt_format=prompt_format)
            
            if args.prepend_title:
                reranking_prompt = prepend_title_doc(reranking_prompt, title)

            d = {
                "prompt": reranking_prompt,
                "answer": "",
            }
            d.update(example)
            
            processed_data.append(d)

    # return a dictionary of pos data, neg data, question to pos paragraphs
    return processed_data



class PretrainingDataset(Dataset):
    def __init__(
        self,
        args,
        processed_data,
        tokenizer,
    ):
        """
        n_negs: number of negative examples to sample at a time per positive example
        """

        self.tokenizer = tokenizer
        self.max_len = args.max_prompt_len

        print("tokenizing data...")
        tokenized_data = []
        for i, example in tqdm(enumerate(processed_data)):

            ## reranking prompt
            prompt = example["prompt"]

            pos_input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
            )

            example["pos_input_ids"] = pos_input_ids["input_ids"][0]
            example["pos_attn_masks"] = pos_input_ids["attention_mask"][0]
            
            q_begin_idx =  get_question_begin_idx_v2(example["pos_input_ids"].cpu().numpy().tolist())
            labels_r = example["pos_input_ids"].clone()
            labels_r [:q_begin_idx] = -100
            labels_r[labels_r == tokenizer.pad_token_id] = -100
            example["labels"] = labels_r

            tokenized_data.append(example)

            ## answer generation prompt
            if args.reader_pretraining:
                prompt_with_answer = prompt + " " + example["answer"]
                prompt_with_a_input_ids = self.tokenizer(
                    prompt_with_answer,
                    return_tensors="pt",
                    max_length=self.max_len,
                    truncation=True,
                    padding="max_length",   
                )
                len_prompt_encode = len(self.tokenizer.encode(prompt))
                len_answer_encode = len(self.tokenizer.encode(' ' + example["answer"]))
                labels_a = prompt_with_a_input_ids["input_ids"][0].clone()

                labels_a[:len_prompt_encode] = -100
                labels_a[labels_a == self.tokenizer.pad_token_id] = -100

                if len_prompt_encode + len_answer_encode >= len(labels_a):
                    #print("Skipping too long prompt with answer")
                    continue
                
                labels_a[len_prompt_encode + len_answer_encode] = tokenizer.eos_token_id # eos_token_id to mark answer end

                example["pos_input_ids"] = prompt_with_a_input_ids["input_ids"][0]
                example["pos_attn_masks"] = prompt_with_a_input_ids["attention_mask"][0]
                example["labels"] = labels_a

                tokenized_data.append(example)

        self.examples = tokenized_data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        pos_ids = self.examples[idx]["pos_input_ids"]
        pos_attn_masks = self.examples[idx]["pos_attn_masks"]
        labels = self.examples[idx]["labels"]

        return {
            "pos_input_ids": pos_ids,
            "pos_attn_masks": pos_attn_masks,
            "labels": labels,
            "passage": self.examples[idx]["passage"],
            "sentence": self.examples[idx]["sentence"],
            "title": self.examples[idx]["title"],
            "answer": self.examples[idx]["answer"],
        }