# Transformers-DeepBiaf


# Performance

| Pretrained Model  | Model | Dataset | UAS | LAS |
| ------------- | ------------- |------------- |------------- |-------------|
| ---  | SOTA | PTB  |  97.42 | 96.26		
| ---  | Deep Biaffine | PTB  |  95.87 | 94.22	
| roberta-large(supar)  | Deep Biaffine | PTB  |  97.33  | 95.86
| roberta-base**(Our)**  | Deep Biaffine | PTB  |  96.60  | 95.21
| ---| ---| ---| ---| ---|
| ---  | Deep Biaffine | CTB  |  89.30 | 88.23
| Bert(Fixed-8)  | Deep Biaffine | CTB  |  92.96 | 91.80
| electra-large(supar)  | Deep Biaffine | PTB  |  92.45  | 89.55
| electra-base**(Our)**  | Deep Biaffine | CTB  |  90.66  | 87.07

#  Related Repo
Bert(Fixed-8)  https://github.com/LindgeW/BiaffineParser
roberta-large(supar)  https://github.com/yzhangcs/parser
