# CNC multistage pipeline
This repo is the temporary anonymous repositary for the submitted paper: 
'A Compare-and-contrast Multistage Pipeline for Uncovering Financial Signals in Financial Reports'

## FINAL_v1.0 Dataset
The data used in this paper includes 
one (pseudo) labeled training dataset and two set of evaluation data.

Here is the data statisitcs
Split  | Type       | Descrption       | Number of Pairs |
---    | ---        | ---              | ---    |
Train  | Revised    | Pseudo-labeling  | 30000  |
Eval   | Revised    | Human-annotation | 200    |
Eval   | Mismatched | Human-annotation | 200    |

## Financial signal highlighting
Formally, we are focusing tackle the financial signal highlighting task.
In document-level, we adopted the multistage pipeline.
Phase | Descrption | Summary |
---   | ---        | ---     |
S_0   | Document segmetation    | Using Cross-seg BERT to separate document (actually aggregate sentences into a segment)
S_1   | Relation recognition    | Using ROUGE and SBERT cosine score to identify the relationship of each semgnet pairs.
S_2 & S_2+| In-domain/Out-domain fine-tuning  | Two-stage domain-adaptive training using out-domain e-SNLI dataset and pseudo-labeld pairs with "revised" relations.

0. Document Segmentation
TBD

1. Segments Alignment
TBD 

2. Sentence Highlighting
See [highlighting](highlighting/) for detail.
