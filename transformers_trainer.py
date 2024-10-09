import argparse
from src.config import Config
import time
from src.model import DeepBiafine
import torch
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
from src.config.eval import calc_evalu_acc
from tqdm import tqdm
from src.data import DepDataset, batch_iter, batch_variable
from transformers import set_seed, AutoTokenizer
from logger import get_logger
logger = get_logger()

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:3", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="ptb")
    parser.add_argument('--optimizer', type=str, default="adamw", help="This would be useless if you are working with transformers package")
    parser.add_argument('--pretr_lr', type=float, default=0, help=" bert/roberta, 2e-5 to 5e-5, frozen is 0")
    parser.add_argument('--pretr_frozen', type=bool, default=True)
    parser.add_argument('--other_lr', type=float, default=2e-4, help=" LSTM/Transformer、MLP、Biaffine, 1e-3 to 1e-4, bert frozened set 1e-3 or 5e-4")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=48, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")

    parser.add_argument('--max_no_incre', type=int, default=80, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")

    ##model hyperparameter
    parser.add_argument('--embedder_type', type=str, default="bert-large-cased", help="you can use 'bert-base-cased' and bert-base-multilan-cased")
    parser.add_argument('--pos_embed_dim', type=int, default=96, help='pos_tag embedding size, heads | pos_embed_dim')
    parser.add_argument('--enc_type', type=str, default='adatrans', choices=['lstm', 'naivetrans', 'adatrans'], help='type of word encoder used')
    parser.add_argument('--enc_nlayers', type=int, default=3, help='number of encoder layers, 3 for LSTM or 6 for Transformer')
    parser.add_argument('--enc_dropout', type=float, default=0.33, help='dropout used in transformer or lstm')
    parser.add_argument('--enc_dim', type=int, default=400, help="hidden size of the encoder, usually we set to 400 for LSTM, 512 for transformer (d_model)")
    parser.add_argument('--heads', type=int, default=8, help='transformer heads')
    # parser.add_argument('--ff_dim', type=int, default=2048, help='transformer forward feed dim')

    parser.add_argument('--mlp_arc_dim', default=500, type=int, help='size of pos mlp hidden layer')
    parser.add_argument('--mlp_rel_dim', default=100, type=int, help='size of xpos mlp hidden layer')
    parser.add_argument('--biaf_dropout', default=0.33, type=float, help='dropout probability of biaffine')

    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args


# def train_model(config: Config, epoch: int, train_loader: DataLoader, dev_loader: DataLoader, test_loader: DataLoader):
def train_model(config, epoch, train_data, dev_data, test_data):
    ### Data Processing Info
    train_num = len(train_data)
    logger.info(f"[Data Info] number of training instances: {train_num}")
    logger.info(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}")
    logger.info(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.")
    logger.info(f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications.")
    model = DeepBiafine(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(model=model, pretr_lr=config.pretr_lr, other_lr=config.other_lr,
                                                                   num_training_steps=train_num // config.batch_size * epoch,
                                                                   weight_decay=1e-6, eps=1e-8, warmup_step=int(0.2 * train_num // config.batch_size * epoch))
    logger.info(f"[Optimizer Info] Modify the optimizer info as you need.")
    logger.info(optimizer)

    model.to(config.device)
    best_uas, best_las, test_best_uas, test_best_las = 0, 0, 0, 0

    no_incre_dev = 0
    logger.info(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs")
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        # for iter, batch in enumerate(train_loader, 1):
        for iter, batch_data in enumerate(batch_iter(train_data, config.batch_size, True)):
            batcher = batch_variable(batch_data, config)
            with torch.cuda.amp.autocast(enabled=False):
                loss = model(batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"], batcher["attention_mask"],
                             batcher["pos_ids"], batcher["dephead_ids"], batcher["deplabel_ids"])
            epoch_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
        end_time = time.time()
        logger.info(f"Epoch {i}: {epoch_loss:.5f}, Time is {(end_time - start_time):.2f}s")

        model.eval()
        dev_uas, dev_las = evaluate_model(config, model, dev_data, "dev")
        test_uas, test_las = evaluate_model(config, model, test_data, "test")
        if dev_uas > best_uas:
            no_incre_dev = 0
            best_uas = dev_uas
            if test_best_uas < test_uas:
                test_best_uas = test_uas
            if test_best_las < test_las:
                test_best_las = test_las
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            logger.info("early stop because there are %d epochs not increasing UAS on dev"%no_incre_dev)
            break

    logger.info("Finished archiving the models")
    logger.info("The best dev UAS: %4.2f, The corresponding test UAS %4.2f, LAS %4.2f" % (best_uas, test_best_uas, test_best_las))


def evaluate_model(config: Config, model: DeepBiafine, dataset: DepDataset, name: str):
    total_arc_acc, total_rel_acc, total_valid_arcs = 0, 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for i, batch_data in enumerate(batch_iter(dataset, config.batch_size, True)):
            batcher = batch_variable(batch_data, config)
            S_arc, S_rel = model(batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"],
                                 batcher["attention_mask"], batcher["pos_ids"], is_train=False)
            batch_size, sent_len = batcher["orig_to_tok_index"].size()
            maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=batcher["orig_to_tok_index"].device).view(1, sent_len).expand(batch_size, sent_len)
            non_pad_mask = torch.le(maskTemp, batcher["word_seq_lens"].view(batch_size, 1).expand(batch_size, sent_len))
            arc_acc, rel_acc, valid_arc = calc_evalu_acc(S_arc, S_rel, batcher["dephead_ids"], batcher["deplabel_ids"], non_pad_mask, batcher["punc_mask"])
            total_arc_acc += arc_acc
            total_rel_acc += rel_acc
            total_valid_arcs += valid_arc

    UAS = total_arc_acc * 100. / total_valid_arcs
    LAS = total_rel_acc * 100. / total_valid_arcs
    logger.info(f"[{name} set Total] UAS.: {UAS:4.2f}, LAS.: {LAS:4.2f}")
    return UAS, LAS


def main():
    parser = argparse.ArgumentParser(description="Roberta Deep Biaffine implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    logger.info(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer")
    # tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True, use_fast=True) # for roberta
    tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, use_fast=True) # for roberta
    logger.info(f"[Data Info] Reading dataset from: \t{conf.train_file}\t{conf.dev_file}\t{conf.test_file}")
    train_dataset = DepDataset(conf.train_file, tokenizer, is_train=True)
    conf.deplabel2idx = train_dataset.deplabel2idx
    conf.pos_size = len(train_dataset.pos2idx)
    conf.rel_size = len(train_dataset.deplabel2idx)
    conf.punctid = train_dataset.punctid

    dev_dataset = DepDataset(conf.dev_file, tokenizer, deplabel2idx=train_dataset.deplabel2idx, pos2idx=train_dataset.pos2idx, is_train=False)
    test_dataset = DepDataset(conf.test_file, tokenizer, deplabel2idx=train_dataset.deplabel2idx, pos2idx=train_dataset.pos2idx, is_train=False)
    num_workers = 0
    train_model(conf, conf.num_epochs, train_dataset, dev_dataset, test_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
    #                               collate_fn=train_dataset.collate_fn)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
    #                               collate_fn=dev_dataset.collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
    #                               collate_fn=test_dataset.collate_fn)
    #
    # train_model(conf, conf.num_epochs, train_dataloader, dev_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
