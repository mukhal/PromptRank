import regex
import unicodedata

def compute_metrics(retrieved):
    metrics = {
            "recall@2": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "recall@50": 0.0,
            "ans_recall@2": 0.0,
            "ans_recall@10": 0.0,
            "ans_recall@20": 0.0,
            "ans_recall@50": 0.0,
        }

    n_bridge = 0

    for d in retrieved:
        gold_docs = d["gold_docs"]
        answer = d["answer"]
        retrieved_titles = [doc["title"] for doc in d["reranked_docs"]]
        retrieved_texts = [doc["text"] for doc in d["reranked_docs"]]
        title_score = {doc["title"]: doc["score"] for doc in d["reranked_docs"]}
        if set(retrieved_titles[:2]) & set(gold_docs) == set(gold_docs):
            metrics["recall@2"] += 1
        if set(retrieved_titles[:10]) & set(gold_docs) == set(gold_docs):
            metrics["recall@10"] += 1
        if set(retrieved_titles[:20]) & set(gold_docs) == set(gold_docs):
            metrics["recall@20"] += 1
        if set(retrieved_titles[:50]) & set(gold_docs) == set(gold_docs):
            metrics["recall@50"] += 1

        ## answer recall
        if answer not in ['yes', 'no']:
            n_bridge+= 1
            for i, text in enumerate(retrieved_texts):
                if has_answer(text, answer):
                    if i < 50:
                        metrics['ans_recall@50'] += 1
                    if i < 20:
                        metrics['ans_recall@20'] += 1
                    if i < 10:
                        metrics['ans_recall@10'] += 1
                    if i < 2:
                        metrics['ans_recall@2'] += 1
                    break
        
    for k in metrics:
        if "ans" in k:
            metrics[k] /= (n_bridge + 1e-10)
        else:
            metrics[k] /= len(retrieved)

        ## round to 3 decimal places
        metrics[k] = round(metrics[k], 3)
    
    return metrics

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def has_answer(text, answers):
    """
    Adapted from DPR: https://github.com/facebookresearch/DPR
    """

    if isinstance(answers, str):
        answers = [answers]

    tokenizer = SimpleTokenizer()
    text = _normalize(text)

    tokenizer = SimpleTokenizer()
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def _normalize(text):
    return unicodedata.normalize('NFD', text)
