Few-shot Reranking for Multi-hop QA via Language Model Prompting

We propose an approach for unsupervised re-ranking of multi-hop document paths for open-domain QA. PromptRank constructs a prompt that consists of **(i) an instruction** and **(ii) the path** and uses a language model to score paths as probability of generating the question given the path.

![main](./img/overview.png)




# Instructions 

#### 1. Download the TF-IDF retriever for HotpotQA provided by [PathRetriever](https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths) from [this link](https://drive.google.com/open?id=1ra37xtEXSROG_f90XxR4kgElGJWUHQyM) and place its contents in `path-retriever/models`

#### 2. Install requirements 
```
pip install -r requirements.txt
```

#### 3. Downloading processed HotpotQA data
The hotpotqa processed data can be downloaded from [Google Drive](https://drive.google.com/file/d/1vBjSe5dzEQBK_IHNTC1eEpLR3pIqDIU3/view?usp=sharing). Then unzip the data and place the content it in ```data/hotpotqa```

#### 4. Run and evaluate PromptRank
```
python run.py \
--model google/t5-small-lm-adapt \
--eval_batch_size=50 \
--max_prompt_len 600  \
--max_doc_len 230 \
--tfidf_pool_size 100  \
--n_eval_examples 1000 \
--temperature 1.0 \
--val_data data/hotpotqa/dev2/dev2.json \
```
