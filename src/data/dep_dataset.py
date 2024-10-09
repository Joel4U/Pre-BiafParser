from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch
import numpy as np
from src.data.data_utils import build_deplabel_idx, root_dep_label, build_pos_idx
from logger import get_logger
from src.data import Instance
from transformers.tokenization_utils_base import BatchEncoding

logger = get_logger()

PAD_INDEX = 0
PUNCT_LABEL = 'punct'  # EN 依存关系标签中的标点符号标签
# PUNCT_LABEL = 'P'  # CN 依存关系标签中的标点符号标签


def convert_instances_to_feature_tensors(instances: List[Instance],
                                         tokenizer: PreTrainedTokenizerFast,
                                         deplabel2idx: Dict[str, int], pos2idx: Dict[str, int]) -> List[Dict]:
    features = []
    logger.info("[Data Info] We are not limiting the max length in tokenizer. You should be aware of that")
    for idx, inst in enumerate(instances):
        words = inst.ori_words
        orig_to_tok_index = []
        res = tokenizer.encode_plus(words, is_split_into_words=True)
        subword_idx2word_idx = res.word_ids(batch_index=0)
        prev_word_idx = -1
        for i, mapped_word_idx in enumerate(subword_idx2word_idx):
            """
            Note: by default, we use the first wordpiece/subword token to represent the word
            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
            """
            if mapped_word_idx is None: ## cls and sep token
                continue
            if mapped_word_idx != prev_word_idx:
                orig_to_tok_index.append(i)
                prev_word_idx = mapped_word_idx
        assert len(orig_to_tok_index) == len(words)
        deplabels = inst.deplabels
        deplabel_ids = [deplabel2idx[deplabel] for deplabel in deplabels] if deplabels else [-100] * len(words)
        dephead_ids = inst.depheads
        tags = inst.pos
        tag_ids = [pos2idx[tag] for tag in tags] if tags else [-100] * len(words)

        features.append({"input_ids": res["input_ids"],
                         "attention_mask": res["attention_mask"],
                         "orig_to_tok_index": orig_to_tok_index,
                         "tag_ids": tag_ids,
                         "dephead_ids": dephead_ids,
                         "deplabel_ids": deplabel_ids})
    return features

def batch_iter(dataset, batch_size, shuffle=False):
    data = dataset.insts_ids
    if shuffle:
        np.random.shuffle(data)

    nb_batch = int(np.ceil(len(dataset) / batch_size))
    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        yield batch_data


def batch_variable(batch:List[Dict], config):
    device = config.device
    batch_size = len(batch)
    word_seq_lens = [len(feature["orig_to_tok_index"]) for feature in batch]
    max_seq_len = max(word_seq_lens)
    max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])

    input_ids = torch.zeros((batch_size, max_wordpiece_length), dtype=torch.long, device=device)
    input_mask = torch.zeros((batch_size, max_wordpiece_length), dtype=torch.long, device=device)
    orig_to_tok_index = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    pos_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    head_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    rel_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    # non_pad_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
    punc_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
    for i, feature in enumerate(batch):
        input_ids[i, :len(feature["input_ids"])] = torch.tensor(feature["input_ids"], dtype=torch.long, device=device)
        input_mask[i, :len(feature["attention_mask"])] = torch.tensor(feature["attention_mask"], dtype=torch.long, device=device)
        orig_to_tok_index[i, :len(feature["orig_to_tok_index"])] = torch.tensor(feature["orig_to_tok_index"], dtype=torch.long, device=device)
        pos_ids[i, :len(feature["tag_ids"])] = torch.tensor(feature["tag_ids"], dtype=torch.long, device=device)
        head_ids[i, :len(feature["dephead_ids"])] = torch.tensor(feature["dephead_ids"], dtype=torch.long, device=device)
        rel_ids[i, :len(feature["deplabel_ids"])] = torch.tensor(feature["deplabel_ids"], dtype=torch.long, device=device)
        # non_pad_mask[i, :len(feature["orig_to_tok_index"])] = 1
        punc_mask[i, :len(feature["deplabel_ids"])] = torch.tensor(
            [deplabel_id == config.punctid for deplabel_id in feature['deplabel_ids']], dtype=torch.bool, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "orig_to_tok_index": orig_to_tok_index,
        "word_seq_lens": torch.tensor(word_seq_lens, dtype=torch.long, device=device),
        "pos_ids":  pos_ids,
        "dephead_ids": head_ids,
        "deplabel_ids": rel_ids,
        # "non_pad_mask": non_pad_mask,
        "punc_mask": punc_mask
    }


class DepDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizerFast,
                 is_train: bool,
                 sents: List[List[str]] = None,
                 deplabel2idx: Dict[str, int] = None, pos2idx: Dict[str, int] = None):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        insts = self.read_file(file=file) if sents is None else self.read_from_sentences(sents)
        self.insts = insts
        if is_train:
            # assert label2idx is None
            if deplabel2idx is not None:
                logger.warning(f"YOU ARE USING EXTERNAL deplabel2idx, WHICH IS NOT BUILT FROM TRAINING SET.")
                self.deplabel2idx = deplabel2idx
                self.pos2idx = pos2idx
                self.punctid = self.deplabel2idx[PUNCT_LABEL]
            else:
                logger.info(f"[Data Info] Using the training set to build deplabel index")
                self.deplabel2idx, self.root_dep_label_id = build_deplabel_idx(self.insts)
                _, self.pos2idx = build_pos_idx(self.insts)
                self.punctid = self.deplabel2idx[PUNCT_LABEL]
        else:
            assert deplabel2idx is not None ## for dev/test dataset we don't build label2idx
            self.deplabel2idx = deplabel2idx
            self.pos2idx = pos2idx
            self.punctid = self.deplabel2idx[PUNCT_LABEL]
        self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, self.deplabel2idx, self.pos2idx)
        self.tokenizer = tokenizer


    def read_from_sentences(self, sents: List[List[str]]):
        """
        sents = [['word_a', 'word_b'], ['word_aaa', 'word_bccc', 'word_ccc']]
        """
        insts = []
        for sent in sents:
            insts.append(Instance(words=sent, ori_words=sent))
        return insts


    def read_file(self, file: str, number: int = -1) -> List[Instance]:
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = ['<'+root_dep_label.lower()+'>']
            ori_words = ['<'+root_dep_label.lower()+'>']
            tags = [root_dep_label]
            depheads = [0]
            deplabels = [root_dep_label]
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(words=words, ori_words=ori_words, pos=tags, depheads=depheads, deplabels=deplabels))
                    words = ['<' + root_dep_label.lower() + '>']
                    ori_words = ['<' + root_dep_label.lower() + '>']
                    tags = [root_dep_label]
                    depheads =[0]
                    deplabels = [root_dep_label]
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                if ls[6] == '_':
                    word, tag, dephead, deplabel = ls[1], ls[3], -1, ls[7]
                else:
                    word, tag, dephead, deplabel = ls[1], ls[3], int(ls[6]), ls[7]
                ori_words.append(word)
                words.append(word)
                tags.append(tag)
                depheads.append(dephead)
                deplabels.append(deplabel)
        logger.info(f"{len(insts)}, number of sentences in file: {file}, we had added 'root' word at the beginning of each sentence")
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    # def collate_fn(self, batch:List[Dict]):
    #     word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch]
    #     max_seq_len = max(word_seq_len)
    #     max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])
    #
    #
    #     batch_size = len(batch)
    #     non_pad_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    #     punc_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.uint8)  # 初始化 punc_mask
    #     for i, feature in enumerate(batch):
    #         padding_length = max_wordpiece_length - len(feature["input_ids"])
    #         input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
    #         mask = feature["attention_mask"] + [0] * padding_length
    #         # type_ids = feature["token_type_ids"] + [self.tokenizer.pad_token_type_id] * padding_length
    #         padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
    #         orig_to_tok_index = feature["orig_to_tok_index"] + [0] * padding_word_len
    #         dephead_ids = feature["dephead_ids"] + [PAD_INDEX] * padding_word_len
    #         deplabel_ids = feature["deplabel_ids"] + [PAD_INDEX] * padding_word_len
    #         # 根据原始句子长度填充non_pad_mask，句子的非填充部分为1，填充部分为0
    #         seq_len = len(feature["orig_to_tok_index"])
    #         non_pad_mask[i, :seq_len] = 1
    #         # 使用 deplabel_ids 中的 'punct' 标签来构建 punc_mask
    #         punc_mask[i, :seq_len] = torch.tensor([deplabel_id == self.punctid for deplabel_id in feature['deplabel_ids']],
    #                                               dtype=torch.uint8)
    #         batch[i] = {"input_ids": input_ids,
    #                     "attention_mask": mask,
    #                     # "token_type_ids": type_ids,
    #                     "orig_to_tok_index": orig_to_tok_index,
    #                     "word_seq_len": feature["word_seq_len"],
    #                     "dephead_ids": dephead_ids,
    #                     "deplabel_ids": deplabel_ids}
    #     encoded_inputs = {key: [example[key] for example in batch] for key in batch[0].keys()}
    #     results = BatchEncoding(encoded_inputs, tensor_type='pt')
    #     # 将non_pad_mask 和 punc_mask 添加到结果中
    #     results['non_pad_mask'] = non_pad_mask
    #     results['punc_mask'] = punc_mask
    #     return results


## testing code to test the dataset
if __name__ == '__main__':
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    dataset = DepDataset(file="data/ptb/test.txt",tokenizer=tokenizer, is_train=True)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)
    print(len(train_dataloader))
    for batch in train_dataloader:
        # print(batch.input_ids.size())
        print(batch.input_ids)
        pass
