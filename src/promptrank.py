from tqdm import tqdm
from .prompt_utils import shorten_paragraph_if_necessary
from .prompt_utils import (shorten_paragraph_if_necessary, prepend_title_doc, score_paths)
import pickle
from rank_bm25 import BM25Okapi
import regex
import unicodedata
from retriever.tfidf_vectorizer_article import TopTfIdf
from nltk.tokenize import  word_tokenize

class PromptRank:
    def __init__(self, tokenizer, tfidf_retriever, model):
        self.tokenizer = tokenizer
        self.tfidf_retriever = tfidf_retriever
        self.model = model

    def retrieve(self, questions, args):
        self.model.eval()
        batch_size = args.eval_batch_size
        tfidf_pool_size = args.tfidf_pool_size
        use_bm25 = args.use_bm25
        prompt_template = args.prompt_template
        max_prompt_len = args.max_prompt_len
        all_retrieved_docs = []

        top_k_first_hops = args.top_k_first_hops
        top_k_second_hops = args.top_k_second_hops

        tfidf_pruner = TopTfIdf(n_to_select=top_k_second_hops,
                            filter_dist_one=True,
                            rank=True)
        
        for ex in questions:
            if "gold_path_titles" in ex:
                gold_docs = ex["gold_path_titles"]
            else:
                gold_docs = []
                for title, _ in ex.get("supporting_facts", [("NOFACT", []),
                                                            ("NOFACT", ())]):
                    if title not in gold_docs:
                        gold_docs.append(title)

        for ex in tqdm(questions):
            if not ex["question"].endswith("?"):
                ex["question"] += "?"

            question = ex["question"]
            answer = ex["answer"] if "answer" in ex else "NOANS"
            qid = ex["_id"]
            qtype = ex["type"] if "type" in ex else "NOTYPE"

            if "gold_path_titles" in ex:
                gold_docs = ex["gold_path_titles"]
            else:
                gold_docs = []
                for title, _ in ex.get("supporting_facts", [("NOFACT", []),
                                                            ("NOFACT", ())]):
                    if title not in gold_docs:
                        gold_docs.append(title)

            assert len(gold_docs) > 0, "No gold documents found for question {}".format(question)

            if "retrieved_docs_text" not in ex:  # if we haven't already pre-retrieved docs
                tfidf_paragraphs, _ = self.tfidf_retriever.ranker.closest_docs(
                    question, tfidf_pool_size)

                retrieved_titles = [
                    t.replace("_0", "") for t in tfidf_paragraphs
                ]
                retrieved_docs_text = self.tfidf_retriever.load_abstract_para_text(
                    retrieved_titles)
                retrieved_docs_text = list(retrieved_docs_text.items())

            else:  # use pre-retrieved docs
                retrieved_docs_text = ex["retrieved_docs_text"]
                retrieved_docs_text = retrieved_docs_text[:tfidf_pool_size]

            if use_bm25:
                ## rerank with bm25
                tokenized_corpus = [
                    word_tokenize(t[1]) for t in retrieved_docs_text
                ]
                bm25_index = BM25Okapi(tokenized_corpus, k1=2 , b=0.75)
                doc_scores = bm25_index.get_scores(word_tokenize(question))

                ## sort by score
                doc_scores = sorted(zip(doc_scores, retrieved_docs_text),
                                    key=lambda x: x[0],
                                    reverse=True)
                retrieved_docs_text = [t[1] for t in doc_scores]

            retrieved_docs_text = dict(retrieved_docs_text)
            title_score = {}
            title_to_text = {}

            if args.scoring_method == "tfidf" or args.scoring_method == "tfidf_bm25":
                retrieved_titles = [
                    t[0].replace("_0", "")
                    for t in retrieved_docs_text.items()
                ]
                title_to_text = retrieved_docs_text

                tfidf_ = TopTfIdf(n_to_select=len(retrieved_titles),
                            filter_dist_one=True,
                            rank=True)

                ## compute dists scores
                _, scores = tfidf_.prune(question, retrieved_docs_text.values(), return_scores=True)
                assert len(scores) == len(retrieved_titles)
                title_score = dict(zip(retrieved_titles, scores))
            
            else: # rank initial pool with LM
                retrieved_docs_text_dict = {}
                for doc_title, doc_text in retrieved_docs_text.items():
                    if not (isinstance(doc_text, str) and doc_text.strip()):
                        continue

                    doc_text = shorten_paragraph_if_necessary(
                        doc_text, args.max_doc_len, tokenizer=self.tokenizer)

                    doc_title = doc_title.replace("_0", "")
                    if args.prepend_title:
                        doc_text = prepend_title_doc(doc_text, doc_title)

                    retrieved_docs_text_dict[doc_title] = doc_text

                retrieved_docs_text = list(
                    t for t in retrieved_docs_text_dict.items()
                    if isinstance(t[1], str))  # ignore empty docs

                title_to_text.update(dict(retrieved_docs_text))

                # score docs
                for j in range(0, len(retrieved_docs_text), batch_size):
                    cur_batch = retrieved_docs_text[j:j + batch_size]
                    batch_titles = [t[0] for t in cur_batch]
                    batch_texts = [t[1] for t in cur_batch]

                    batch_scores = score_paths(self.model ,
                    prompt_template=prompt_template,
                    question=question,
                    path_list= [[t] for t in batch_texts],
                    tokenizer=self.tokenizer,
                    max_prompt_len=max_prompt_len,
                    temperature=args.temperature,
                    ensembling_method=args.ensembling_method,)

                    for score, title in zip(batch_scores, batch_titles):
                        title_score[title.replace("_0", "")] = score

                ## sort by score
                assert len(title_score) == len(retrieved_docs_text)

                title_score_sorted = sorted(
                    title_score.items(), key=lambda x: x[1],
                    reverse=True)

                scores= [t[1] for t in title_score_sorted]
                retrieved_titles = [t[0] for t in title_score_sorted]

            ## multi-hop prompts
            linked_title_to_text = ex["all_linked_paras_dic"]
            multi_prompts = []
            multi_prompts_titles = []
            
            for first_hop_title in retrieved_titles[:
                                                    top_k_first_hops]:  ## only expand top k2 docs
                first_hop_text = title_to_text[first_hop_title]

                first_hop_text = shorten_paragraph_if_necessary(first_hop_text, args.max_doc_len, tokenizer=self.tokenizer)
                linked_paragraph_titles = ex["all_linked_para_title_dic"][first_hop_title]
                linked_paragraph_texts = [linked_title_to_text[t] for t in linked_paragraph_titles]

                ## select top k links using tfidf pruner
                rank_idx = tfidf_pruner.prune(question=question,
                                                        paragraphs=linked_paragraph_texts,
                                                        return_scores=False)

                rank_idx = rank_idx[:top_k_second_hops]
                linked_paragraph_titles = [linked_paragraph_titles[i] for i in rank_idx]

                for t in linked_paragraph_titles:
                    second_hop_title = t.replace("_0","")
                    second_hop_text = linked_title_to_text[t]

                    ## shorten
                    second_hop_text = shorten_paragraph_if_necessary(second_hop_text, args.max_doc_len, tokenizer=self.tokenizer)
                    assert isinstance(second_hop_text, str)

                    title_to_text[second_hop_title] = linked_title_to_text[t]

                    if getattr(args, "reverse_path", False):
                        multi_prompts.append(
                            [second_hop_text, first_hop_text])
                        multi_prompts_titles.append((second_hop_title, first_hop_title))
                    else:
                        multi_prompts.append([first_hop_text, second_hop_text])
                        multi_prompts_titles.append((first_hop_title, second_hop_title))

                scores_mutlihop_titles = {}
                if len(multi_prompts) > 0:

                    if args.scoring_method == "tfidf_bm25":
                        ## score multi_hop prompts with bm25
                        tokenized_corpus = [
                            word_tokenize(d1 + " " + d2) for d1, d2 in multi_prompts]
                        bm25_index = BM25Okapi(tokenized_corpus, k1=2 , b=0.75)
                        doc_scores = bm25_index.get_scores(word_tokenize(question))
                        ## sort by score
                        assert len(doc_scores) == len(multi_prompts)
                        for i, (t1, t2) in enumerate(multi_prompts_titles):
                            scores_mutlihop_titles[(t1, t2)] = doc_scores[i]

                    else: ## LM-based scoring
                        for j in range(0, len(multi_prompts), batch_size):
                            paths_batch = multi_prompts[j:j + batch_size]
                            batch_titles = multi_prompts_titles[j:j + batch_size]

                            batch_scores = score_paths(
                                self.model,
                                prompt_template=prompt_template,
                                question=question,
                                path_list=paths_batch,
                                tokenizer=self.tokenizer,
                                max_prompt_len=max_prompt_len,
                                temperature=args.temperature,
                            )

                            for i, (t1, t2) in enumerate(batch_titles):
                                scores_mutlihop_titles[(t1, t2)] = batch_scores[i]     # score of full path

                scores_mutlihop_titles = sorted(scores_mutlihop_titles.items(), key=lambda x: x[1], reverse=True)

                retrieved_titles = []

                for (title1, title2), score in scores_mutlihop_titles:
                    for t in [title1, title2]:
                        title_score[t] = max(title_score.get(t, -1e5), score)

                title_score_sorted = sorted(
                    title_score.items(), key=lambda x: x[1],
                    reverse=True)
                retrieved_titles = [t[0] for t in title_score_sorted]

            retrieved_text_title = []
            for title in retrieved_titles:
                doc = {}
                doc["title"] = title.replace("_0", "")
                try:
                    cur_paragraph_text = title_to_text[title + '_0']
                except KeyError:
                    cur_paragraph_text = title_to_text[title]
                doc['text'] = cur_paragraph_text
                doc['score'] = title_score[doc["title"]]
                retrieved_text_title.append(doc)
            
            d = {
                "id": qid,
                "question": question,
                "answer": answer,
                "gold_docs": gold_docs,
                "reranked_docs": retrieved_text_title
            }
            all_retrieved_docs.append(d)
        
        return all_retrieved_docs
