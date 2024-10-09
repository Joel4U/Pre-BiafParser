import torch
import torch.nn as nn
from typing import Tuple, Union
from src.model.embedder import TransformersEmbedder
from src.model.module.bilstm_enc import BiLSTMEncoder
# from src.model.module.trans_encoder import TransformerEncoder
from src.model.module.transformer_enc import TransformerEncoder




from src.model.module.mlp_biaf import NonlinearMLP, Biaffine

PAD_INDEX = 0

def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    # drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, time_step)).transpose(1, 2)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask

class DeepBiafine(nn.Module):

    def __init__(self, config):
        super(DeepBiafine, self).__init__()
        self.word_embedder = TransformersEmbedder(transformer_model_name=config.embedder_type)
        self.tag_embedding = nn.Embedding(num_embeddings=config.pos_size,
                                          embedding_dim=config.pos_embed_dim,
                                          padding_idx=0)
        self.enc_dropout = config.enc_dropout
        self.biaf_dropout = config.biaf_dropout
        input_dim = self.word_embedder.get_output_dim() + config.pos_embed_dim
        self.enc_type = config.enc_type
        if config.enc_type == 'lstm':
            self.encoder = BiLSTMEncoder(input_dim=input_dim,
                                         hidden_dim=config.enc_dim, drop_lstm=self.enc_dropout, num_lstm_layers=config.enc_nlayers)
        elif config.enc_type == 'adatrans':
            self.encoder = TransformerEncoder(d_model=input_dim, num_layers=config.enc_nlayers, n_head=config.heads,
                                              feedforward_dim=2 * input_dim, attn_type='adatrans', dropout=self.enc_dropout)
        elif config.enc_type == 'naivetrans':
            self.encoder = TransformerEncoder(d_model=input_dim, num_layers=config.enc_nlayers, n_head=config.heads,
                                              feedforward_dim=2 * input_dim, attn_type='naivetrans', dropout=self.enc_dropout)
        self._activation = nn.ReLU()
        self.mlp_arc = NonlinearMLP(in_feature=input_dim, out_feature=config.mlp_arc_dim * 2, activation=nn.ReLU())    # nn.LeakyReLU(0.1)
        self.mlp_rel = NonlinearMLP(in_feature=input_dim, out_feature=config.mlp_rel_dim * 2, activation=nn.ReLU())    # nn.LeakyReLU(0.1)
        self.arc_biaffine = Biaffine(config.mlp_arc_dim, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_dim, config.rel_size, bias=(True, True))

    def forward(self, subword_input_ids: torch.Tensor, word_seq_lens: torch.Tensor, orig_to_tok_index: torch.Tensor, attention_mask: torch.Tensor,
                    tags: torch.Tensor, trueheads: torch.Tensor = None, truerels: torch.Tensor = None,
                    is_train: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        word_emb = self.word_embedder(subword_input_ids, orig_to_tok_index, attention_mask)
        tags_emb = self.tag_embedding(tags)
        batch_size, sent_len = orig_to_tok_index.size()
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=word_emb.device).view(1, sent_len).expand(batch_size, sent_len)
        non_pad_mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        word_rep = torch.cat((word_emb, tags_emb), dim=-1).contiguous()
        if self.training:
            word_rep = timestep_dropout(word_rep, 0.2)
        if self.enc_type == 'lstm':
            enc_out = self.encoder(word_rep, word_seq_lens)
        else:
            enc_out = self.encoder(word_rep, non_pad_mask)

        if is_train:
            enc_out = timestep_dropout(enc_out, self.enc_dropout)

        arc_feat = self.mlp_arc(enc_out)
        rel_feat = self.mlp_rel(enc_out)
        arc_head, arc_dep = arc_feat.chunk(2, dim=-1)
        rel_head, rel_dep = rel_feat.chunk(2, dim=-1)

        if is_train:
            arc_head = timestep_dropout(arc_head, self.biaf_dropout)
            arc_dep = timestep_dropout(arc_dep, self.biaf_dropout)
            rel_head = timestep_dropout(rel_head, self.biaf_dropout)
            rel_dep = timestep_dropout(rel_dep, self.biaf_dropout)

        S_arc = self.arc_biaffine(arc_dep, arc_head).squeeze(-1)
        S_rel = self.rel_biaffine(rel_dep, rel_head)

        if is_train:
            loss = self.calc_loss(S_arc, S_rel, trueheads, truerels, non_pad_mask)
            return loss
        else:
            return S_arc, S_rel

    def calc_loss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
        '''
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len) 有效部分mask
        :return:
        '''
        # non_pad_mask[:, 0] = 0  # mask out <root>
        # non_pad_mask = non_pad_mask.byte()
        pad_mask = (non_pad_mask == 0)

        bz, seq_len, _ = pred_arcs.size()
        masked_true_heads = true_heads.masked_fill(pad_mask, -1)
        arc_loss = nn.functional.cross_entropy(pred_arcs.reshape(bz*seq_len, -1), masked_true_heads.reshape(-1), ignore_index=-1)

        bz, seq_len, seq_len, rel_size = pred_rels.size()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_heads].contiguous()

        masked_true_rels = true_rels.masked_fill(pad_mask, -1)
        # (bz*seq_len, rel_size)  (bz*seq_len, )
        rel_loss = nn.functional.cross_entropy(out_rels.reshape(-1, rel_size), masked_true_rels.reshape(-1), ignore_index=-1)
        return arc_loss + rel_loss