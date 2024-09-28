# Transformers-DeepBiaf


# Performance

| Pretrained Model  | Model | Dataset | UAS | LAS |
| ------------- | ------------- |------------- |------------- |-------------|
| ---  | SOTA | PTB  |  97.42 | 96.26		
| ---  | Deep Biaffine | PTB  |  95.87 | 94.22	
| roberta-large(supar)  | Deep Biaffine | PTB  |  97.33  | 95.86
| roberta-base(Our)  | Deep Biaffine | PTB  |  96.60  | 95.21
| bert-base-cased(Our)  | Deep Biaffine | PTB  |  96.74  | 95.13
| bert-large-cased(Our)  | Deep Biaffine | PTB  |  96.15  | 93.94
| ---| ---| ---| ---| ---|
| ---  | Deep Biaffine | CTB-5  |  89.30 | 88.23
| Bert(Fixed-8)  | Deep Biaffine | CTB-5  |  92.96 | 91.80
| electra-large(supar)  | Deep Biaffine | CTB-5  |  92.45  | 89.55
| electra-base(Our)  | Deep Biaffine | CTB-5  |  90.66  | 87.07
| bert-base-multi(Our)  | Deep Biaffine | CTB-5  |  91.26  | 89.96

Roberta learing-rate： 2e-5， batch_size: 48, epoch: 20

Bert EN pretr_lr： 2e-5，other_lr: 5e-4, adatrans：3-layers, heads：8, batch_size: 48, epoch: 100

Bert CN pretr_lr： 2e-5，other_lr: 5e-4, adatrans：3-layers, heads：8, batch_size: 48, epoch: 120
large-large，情况很怪，到了100轮还在涨，但是轮数已到就停了。但是las还是远低于base的
#  Related Repo
Bert(Fixed-8)  https://github.com/LindgeW/BiaffineParser

roberta-large(supar)  https://github.com/yzhangcs/parser

# File: giga.100.txt
https://github.com/yzhangcs/parser/releases/download/v1.1.0/giga.100.zip

