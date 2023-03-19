# A Compare-and-contrast Multistage Pipeline for Uncovering Financial Signals in Financial Reports

This repo is the temporary anonymous repositary for double-blind reviews.

We releasd our FINAL (FINancial-ALpha) dataset, including the pseudo-labeled training data and the humna-annotated labels.

---

## FINAL_v1.0 Dataset
The data used in this paper includes one (pseudo) labeled training dataset and two set of evaluation data.

> The parsed dataset is based on [Software Repositary for Accounting and Finance](https://sraf.nd.edu/sec-edgar-data/), which its source contents are officially released from SEC/EDGAR. 

- The data definition and data statisitcs

Split  | Type       | Descrption       | Number of Pairs |
---    | ---        | ---              | ---    |
Train  | Revised    | Pseudo-label     | 30000  |
Eval   | Revised $\mathcal(T)^{\alpha}_1$   | Human-annotation | 200    |
Eval   | Mismatched $\mathcal(T)^{\alpha}_2$| Human-annotation | 200    |

- Data example 
Note that we use the 'jsonl' format; each line in files is an instance.
An instance is compiled into a 'dict' object as one line in the file.

Key     | contents | Descrption |
---     | ---      | ---- |              
`sentA`   | raw text (string) | the `reference` segment in a report.
`sentB`   | raw text (string) | the `target` segment in a report.
`wordsA`  | a list of strings | splitted tokens of `sentA`.
`wordsB`  | a list of strings | splitted tokens of `sentB`.
`words`   | A list of strings | splitted tokens of `sentB` and `sentB`, seperated by `<tag>`.
`labels`  | A list of labels (binary). | Human annotation: final binary labeling is based on agreement of annotators.
`probs`   | A list of labels (float).  | Human annotation: final fine-grained labeling is based on the average of annontated binary `labels`.
`keywordsA`  | a list of strings | the annotated tokens.
`keywordsB`  | a list of strings | the annotated tokens. 

```
{
    "sentA": "Net loss for fiscal year 2014 was $836 thousand ...", 
    "sentB": "Net income for fiscal year 2015 was $364 thousand ...",
    "type": 1, 
    "words": ["<tag1>", "Net", "loss", "for", "fiscal", "year", "2014", "was", "$836", "thousand", ..., ".", "<tag2>", "Net", "income", "for", "fiscal", "year", "2015", "was", "$364", "thousand", ..., ".", "<tag3>"], 
    "wordsA": ["Net", "loss", "for", "fiscal", "year", "2014", "was", "$836", "thousand", ..., "."], 
    "wordsB": ["Net", "income", "for", "fiscal", "year", "2015", "was", "$364", "thousand", ..., "."], 
    "keywordsA": [], 
    "keywordsB": ["Net", "income", "$364", "thousand", "increase", "of", "$1.2", "million"], 
    "labels": [-1, 0, 0, 0, , -1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, -1], 
    "probs": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3333333333333333, 1.0, 1.0, 0.0, -1.0]
}
```

## Financial signal highlighting
Formally, we are focusing tackle the financial signal highlighting task.
In document-level, we adopted the multistage pipeline.

Phase | Descrption | Summary |
---   | ---        | ---     |
S_0   | Document segmetation    | Using Cross-seg BERT to separate document (actually aggregate sentences into a segment)
S_1   | Relation recognition    | Using ROUGE and SBERT cosine score to identify the relationship of each semgnet pairs.
S_2 & S_2+| In-domain/Out-domain fine-tuning  | Two-stage domain-adaptive training using out-domain e-SNLI dataset and pseudo-labeld pairs with "revised" relations.

1. Document Segmentation

> TBD

2. Segments Alignment

> TBD

3. Sentence Highlighting
See [highlighting](highlighting/README.md) for detail.
