# RoBERTa-DeepBiaf


# Performance

| Pretrained Model  | Model | Dataset | UAS | LAS |
| ------------- | ------------- |------------- |------------- |-------------|
| ---  | SOTA | PTB  |  97.42 | 96.26		
| ---  | Deep Biaffine | PTB  |  95.87 | 94.22	
| roberta_base  | Deep Biaffine | PTB  |  96.60  | 95.21
| ---| ---| ---| ---| ---|
| electra-base  | Deep Biaffine | CTB  |  90.66  | 87.07
| ---  | Deep Biaffine | CTB  |  89.30 | 88.23
| Bert(Fixed-8)  | Deep Biaffine | CTB  |  92.96 | 91.80

#  Related Repo
Bert(Fixed-8)  https://github.com/LindgeW/BiaffineParser
