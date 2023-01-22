import collections

def load_stopwords(source, max_num=None):
    """
    Args:
        source: [str] `nltk` or `anserini` predefined stopword list.
    """
    stopwords_list = []
    if source == 'nltk':
        from nltk.corpus import stopwords
        stopwords_list = [w.casefold() for w in stopwords.words('english')]
    elif source == 'anserini':
        with open("stopwords_en.txt", 'r') as f:
            for line in f:
                if "#" not in line.strip():
                    stopwords_list += [line.strip()]
    elif source is not None:
        with open(source, 'r') as f:
            stopwords_list = [line.strip() for line in f.readlines()]

    return [w.casefold() for w in stopwords_list][:max_num]

def load_master_dict(path, max_num=None):
    """
    Args:
        path: [str] path of pre-extracted dictionary (wordlist)
    """
    lexicon = collections.defaultdict(int)
    with open(path, 'r') as f:
        for line in f:
            keyword, word_cnt, doc_cnt = line.strip().split('\t')
            lexicon[keyword] = doc_cnt

    return [w.casefold() for w in lexicon][:max_num]
