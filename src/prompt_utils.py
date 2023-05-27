import torch
import numpy as np


def create_prompt(
    paragraphs,
    prompt_template,
):
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]

    paragraphs = [p for p in paragraphs if p.strip() != ""]
    doc_repr = "Document:" if "document:" in prompt_template.lower() else ("Passage:" if "passage:" in prompt_template.lower() else "")
    for p in paragraphs:
        prompt_template = prompt_template.replace(
            "<P>", f"{p} {doc_repr} <P>"
        )

    prompt_template = prompt_template.replace(f"{doc_repr} <P>", "") # remove "<P>"

    #print(prompt_template)
    return prompt_template


def create_qa_prompt(question,
                  passages,
                  prompt_template,
                  passage_prefix="Passage: "):

    if not question.endswith("?"):
        question += "?"

    t = prompt_template.replace("<Q>", question)

    if "Passage: <P>" in t:
        t = t.replace("Passage: <P>", "<P>")

    for p in passages:
        t = t.replace("<P>", passage_prefix + p + " <P>")
    t = t.replace("<P>", "")
    return t

def batch_get_logprob_from_logits(logits,
                            labels,
                            temperature=1.0,
                            normalized=True,
                            shift=True,
                            average=True):
    '''
    Logits: (B, L, V)
    Labels: (B, L)
    '''

    if normalized:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    else:
        loss_fct = torch.nn.NLLLoss(ignore_index=-100, reduction='none')

    if shift:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    logits /= temperature

    B , L, V = logits.size()

    ## check if empty labels
    if labels.nelement() == 0 or torch.all(labels == -100):
        print("Empty/-100 labels")
        return torch.tensor(0.0, device=logits.device)

    log_probs = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # (B*L)
    log_probs = -log_probs.view(B, L)

    if average:
        log_probs = log_probs.mean(dim=1) # (B)

    assert not torch.isnan(log_probs).any(), "Log probs have a NaN value!"
    return log_probs


def get_logprob_from_logits(logits, labels, shift, temperature):
    '''
    Logits: (L, V)
    Labels: (L)
    '''
    loss_fct = torch.nn.CrossEntropyLoss()
    if shift:
        logits = logits[:-1, :].contiguous()
        labels = labels[1:].contiguous()
    
    logits /= temperature
    
    log_prob_i = -loss_fct(logits.view(-1, logits.size(-1)),
                           labels.view(-1))
    return log_prob_i



def score_paths(model,
                prompt_template,
                question,
                path_list,
                tokenizer,
                max_prompt_len,
                temperature=1.0,
                ensembling_method="mean",):
    '''
    Description: encapsulates both create_prompt() and score_prompt()
    Args:
        question: a question
        paths_list: list of paths of size N, where each path is a list of H (hops) of passage texts
        prompt_template: string or list of strings. If list, then use ensemble of templates to score each path

    returns a list of scores of size N
    '''

    if isinstance(prompt_template, str):
        prompt_template = [prompt_template]

    scores_per_template = []
    logits_per_template = []
    ensembling_method = ensembling_method.lower().strip()

    for prompt_t in prompt_template:
        prompts = [create_prompt(p, prompt_t) for p in path_list]

        max_len = max([len(tokenizer.encode(p)) for p in prompts])
        if max_len > max_prompt_len:
            print(f"Max prompt length {max_len} > {max_prompt_len}")

        scores = score_prompt_batch(model,
                                    prompts,
                                    prompt_t, [question] * len(prompts),
                                    tokenizer,
                                    method="conditional_plp",
                                    max_prompt_len=max_prompt_len,
                                    temperature=temperature,
                                    get_logits=True if "token" in ensembling_method else False,)

        if 'token' in ensembling_method:
            _, logits, labels = scores # (B, L, V)
            logits_per_template.append(logits)

        else:
            scores = scores.cpu().numpy().tolist()
            scores_per_template.append(scores)

    if "token" in ensembling_method:
        shift = True if "gpt" in model.config._name_or_path else False

        logprobs_per_template = []
        #lg = torch.stack(logits_per_template, dim=0).sum(dim=0) # (B, L, V)

        for lg in logits_per_template:
            logprobs = batch_get_logprob_from_logits(lg,
                                                    labels,
                                                    temperature=temperature,
                                                    shift=shift,
                                                    average=False) # B, L

            logprobs_per_template.append(logprobs)

        logprobs = torch.stack(logprobs_per_template, dim=0) # (N, B, L)

        if ensembling_method == 'token-max':
            logprobs = logprobs.max(dim=0)[0]
        elif ensembling_method == 'token-median':
            logprobs = logprobs.median(dim=0)[0]
        elif ensembling_method == 'token':
            logprobs = logprobs.mean(dim=0) # (B, L)
        else:
            raise ValueError(f"Unknown ensembling method {ensembling_method}")
        scores = logprobs.mean(dim=1).cpu().numpy().tolist() # (B)

    elif ensembling_method == 'mean':
        scores = np.array(scores_per_template) #* np.array(all_prompt_weights)[:, None]
        scores = scores.mean(axis=0).tolist() # max over templates

    elif ensembling_method == 'max':
        scores = np.array(scores_per_template)
        scores = scores.max(axis=0).tolist()

    else:
        raise ValueError(f"Unknown ensembling method: {ensembling_method}")

    return scores


def score_prompt_batch(model, prompt_list, prompt_template, question_list, tokenizer, method,
                       max_prompt_len, temperature=1.0, get_logits=False):

    ## prompt consists of paragraphs without questions.
    ## for gpt-2 we need to append the question to the prompt
    ## for t5, we will not

    #TODO refactor
    assert method == "conditional_plp", "Batch scoring is not implemented for {}".format(
        method)

    if "gpt" in model.config._name_or_path:

        passage_q_list = [p.strip() + " " + q for p, q in zip(prompt_list, question_list)]

        prompt_input_ids = tokenizer(
            passage_q_list,
            return_tensors="pt",
            max_length=max_prompt_len,
            truncation=True,
            padding='longest',
        ).to(model.device)

    elif "t5" in model.config._name_or_path or 'bart' in model.config._name_or_path or 'T0' in model.config._name_or_path:
        passage_input_ids = tokenizer(
            prompt_list,
            return_tensors="pt",
            max_length=max_prompt_len,
            truncation=True,
            padding='longest',
        ).to(model.device)

        question_input_ids = tokenizer(
            ["" + q for q in question_list],
            return_tensors="pt",
            max_length=max_prompt_len,
            truncation=True,
            padding='longest',
        ).to(model.device)
    else:
        raise NotImplementedError("Model {} not supported".format(model.config._name_or_path))

    bsz = len(prompt_list)
    log_probs = []

    with torch.no_grad():

        if "gpt" in model.config._name_or_path: ## gpt
            labels = prompt_input_ids["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            logits = model(**prompt_input_ids).logits

            #import ipdb; ipdb.set_trace()
            for i in range(bsz): ## ignore non-question labels
                idx = get_question_begin_idx_v2(prompt_input_ids.input_ids[i])
                #idx = len(tokenizer.encode(prompt_list[i])) ##  hack for the no-inst experiment
                if idx >= max_prompt_len:
                    raise ValueError("prompt too long!")
                labels[i, :idx] = -100

        elif "t5" in model.config._name_or_path or "bart" in model.config._name_or_path or 'T0' in model.config._name_or_path:
            labels = question_input_ids["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            logits = model(**passage_input_ids, labels=labels).logits

        shift = True if "gpt" in model.config._name_or_path else False

        for i in range(bsz): # FOR LOOP SINCE BATHCING PRODUCES INCONSISTENT RESULTS WITH DIFFERENT BATCH SIZES -- TODO: fix in final version
            log_prob_i = get_logprob_from_logits(logits[i], labels[i], shift=shift, temperature=temperature)
            log_probs.append(log_prob_i)

        log_probs = torch.stack(log_probs, dim=0)
        #log_probs = batch_get_logprob_from_logits(logits, labels, temperature=temperature, shift=shift)

    if get_logits:
        return log_probs, logits, labels

    return log_probs

def get_paragraph_begin_idx(tokenizer, prompt_template, question):
    '''
    In case question comes first
    '''
    assert prompt_template.find("<Q>") < prompt_template.find(
        "<P>"), "prompt_template must contain <Q> before <P>"

    prompt_with_q = prompt_template.replace("<Q>", question)
    prompt_without_p = prompt_with_q[:prompt_with_q.find("<P>")]
    p_start_idx = len(
        tokenizer.encode(prompt_without_p,
                         truncation=True))
    return p_start_idx

def get_question_begin_idx(tokenizer, prompt_template, paragraph):
    '''
    In case paragraph comes first
    '''
    assert prompt_template.find("<Q>") > prompt_template.find(
        "<P>"), "prompt_template must contain <P> before <Q>"

    prompt_with_p = prompt_template.replace("<P>", paragraph)
    prompt_without_q = prompt_with_p[:prompt_with_p.find("<Q>")].strip()
    q_start_idx = len(
        tokenizer.encode(prompt_without_q,
                         truncation=True))
    return q_start_idx

def get_question_begin_idx_v2(prompt_encoding):
    '''
    In case paragraph comes first
    '''
    if torch.is_tensor(prompt_encoding):
        prompt_encoding = prompt_encoding.cpu().numpy().tolist()

    #last_colon_idx = find_last(prompt_encoding, 25)

    question_start_idx = find_segment_in_ids(
        prompt_encoding,
        [18233, 25]) + 2  # find "Question: " in prompt_encoding

    return question_start_idx

def get_ans_begin_idx(prompt_encoding):
    '''
    In case paragraph comes first
    '''
    ## find last occurrence of ':'
    last_question_mark_idx = find_last(prompt_encoding, 30)
    if last_question_mark_idx is None:
        return None
    return last_question_mark_idx + 1

def shorten_paragraph_if_necessary(paragraph,
                                   max_tokens=400,
                                   tokenizer=None):
    ## make sure paragraph is shorter than max_par_tokens
    par_ids = tokenizer.encode(paragraph)
    if len(par_ids) > max_tokens:
        #print("Paragraph is too long. Shortening it")
        par = tokenizer.decode(par_ids[:max_tokens], skip_special_tokens=True)
        if not par.strip().endswith("."):
            par = par + ". " # add period if necessary
        return par

    return paragraph


def prepend_title_doc(doc, title):
    return "Title: " + title.split("_")[0] + ". " + doc


def find_last(lst, elm):
    '''
    Finds the last index of an element in a list
    '''
    idx = lst[::-1].index(elm)
    if idx < 0:
        return None
    return len(lst) - 1 - idx


def find_segment_in_ids(lst, ids_segment):
    '''
    returns the index of the last occurrence of the segment in the list
    '''
    # start from the end
    cur_i = len(lst) - 1 - len(ids_segment)

    while cur_i >= 0:
        if lst[cur_i:cur_i + len(ids_segment)] == ids_segment:
            return cur_i
        cur_i -= 1

    raise ValueError("Could not find 'Question: ' in prompt ids!")


def trim_passages_for_reader(question, passages, max_prompt_len, max_doc_len, tokenizer):
    '''
    returns the maximum number of truncated passages that can fit in the prompt
    '''
    max_passages = max_prompt_len // max_doc_len
    if max_passages == 0:
        raise ValueError("max_prompt_len must be greater than max_doc_len")
        return None

    ## trim all passages if necessary
    passages = [shorten_paragraph_if_necessary(p, max_doc_len, tokenizer) for p in passages]
    passages_max_len = max_prompt_len - len(tokenizer.encode(question)) - 20 # extra space for prompt and answer

    truncated_passages = []
    cur_total_passages_len = 0

    for passage in passages[::-1]: # go in reverse order to keep highest scoring passages
        if cur_total_passages_len + len(tokenizer.encode(passage)) > passages_max_len:
            break
        truncated_passages.append(passage)
        cur_total_passages_len += len(tokenizer.encode(passage))

    truncated_passages = truncated_passages[::-1]
    return truncated_passages
