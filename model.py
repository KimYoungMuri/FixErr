from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import numpy as np
from utils import try_gpu, BOS_INDEX, EOS_INDEX, PAD_INDEX, UNK_INDEX, BOS, EOS, PAD, UNK
from utils import prepare_rnn_seq, recover_rnn_seq
from module.graphnet import GraphAttentionEncoderFlow
from module.decoder import Decoder, Attention
from module.copy_generator import CopyGenerator, CopyGeneratorLoss, collapse_copy_scores
from module.beam_search import BeamSearch


def make_one_hot(labels, C):
    labels = labels.unsqueeze(1)
    one_hot = torch.cuda.FloatTensor(labels.size(0), C).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


def cross_entropy_after_probsum(pred, soft_targets):
    softmax = nn.Softmax()
    probsum = torch.sum(soft_targets * softmax(pred), 1)
    return torch.mean(- torch.log(probsum))


class Model(nn.Module):
    def __init__(self, vocab, vocab_x):
        super().__init__()

    def initialize(self):
        pass

    def forward(self, batch):
        raise NotImplementedError

    def get_loss(self, logit, batch):
        raise NotImplementedError

    def get_pred(self, logit, batch):
        raise NotImplementedError


class CRmodel(Model):
    def __init__(self, vocab, vocab_x):
        super().__init__(vocab, vocab_x)

        self.vocab = vocab
        self.vocab_x = vocab_x

        ##Embedding
        self.tok_emb = nn.Embedding(len(vocab), 200)
        self.dropout = nn.Dropout(0.3)

        ##Combine outputs
        pos_enc = []
        pos_enc.append(nn.Dropout(0.3))
        pos_enc.append(nn.Linear(300, 200))
        pos_enc.append(nn.ReLU())
        self.pos_enc = nn.Sequential(*pos_enc)

        ##Graph attention
        self.graph_attention = GraphAttentionEncoderFlow(2, 200, 5, 200, 0.3, 0.3)

        ##LSTM
        self.txt_emb1 = nn.LSTM(300, 200, num_layers=3, bidirectional=True, dropout=0.3)
        self.txt_lin1 = nn.Linear(400, 200)
        self.txt_emb2 = nn.LSTM(200, 200, num_layers=1, bidirectional=True, dropout=0.3)

        self.cod_emb1 = nn.LSTM(300, 200, num_layers=3, bidirectional=True, dropout=0.3)
        self.cod_lin1 = nn.Linear(400, 200)
        self.cod_emb2 = nn.LSTM(200, 200, num_layers=1, bidirectional=True, dropout=0.3)

        self.msg_emb1 = nn.LSTM(300, 200, num_layers=3, bidirectional=True, dropout=0.3)
        self.msg_lin1 = nn.Linear(400, 200)
        self.msg_emb2 = nn.LSTM(200, 200, num_layers=1, bidirectional=True, dropout=0.3)

        ##Combine layer
        combine_layer = []
        combine_layer.append(nn.Dropout(0.3))
        combine_layer.append(nn.Linear(3200, 200))
        combine_layer.append(nn.ReLU())
        combine_layer.append(nn.Dropout(0.3))
        combine_layer.append(nn.Linear(200, 200))
        combine_layer.append(nn.ReLU())
        self.com_lay = nn.Sequential(*combine_layer)

        ##LSTM3
        self.lin_emb = nn.LSTM(200, 200, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)

        ##Localization
        final_layer = []
        final_layer.append(nn.Dropout(0.3))
        final_layer.append(nn.Linear(600, 200))
        final_layer.append(nn.ReLU())
        final_layer.append(nn.Dropout(0.3))
        final_layer.append(nn.Linear(200, 1))
        self.fin_lay = nn.Sequential(*final_layer)

        ##Gold code
        self.bridge_c, self.bridge_h = [nn.Sequential(nn.Dropout(0.3), nn.Linear(2000, 2000),nn.Tanh(),
                                                      nn.Linear(2000, 1600), nn.Tanh())] * 2
        self.decoder = Decoder(self.tok_emb, 600, 400, len(vocab), n_layers=4, dropout=0.3)
        self.cop_gen = CopyGenerator(input_size=400, output_size=len(vocab))

        ##loss
        self.cop_gen_loss = CopyGeneratorLoss(vocab_size=len(vocab))

    def initialize(self):
        pass

    def create_mask(self, input_sequence):
        return ((input_sequence != PAD_INDEX) & (input_sequence != BOS_INDEX) & (input_sequence != EOS_INDEX))

    def forward_encode(self, batch):
        text_stuff = None
        (cod_stuff, cod_slen), (msg_stuff, msg_slen), err_line, gold_code = self.get_stuffs_to_embed(batch)
        num_lines = len(cod_stuff[0])

        def prepare_for_removing_pad(mask):
            mask = mask.long()
            b_size, fat_len = mask.size()
            slim_lens = mask.sum(dim=1)
            max_slim_len = max(slim_lens.cpu().numpy())
            fat2slim = []
            slim2fat = []
            fat_count = 0
            slim_count = 0
            for b_idx in range(b_size):
                _mask = mask[b_idx]
                positive_idxs = (_mask == 1).nonzero().squeeze(1)

                _fat2slim = _mask.index_add_(0, positive_idxs, try_gpu(torch.arange(positive_idxs.size(0))) + slim_count)
                _fat2slim = _fat2slim - 1
                _fat2slim = _fat2slim.unsqueeze(0)

                _slim2fat = positive_idxs + fat_count

                slim_count += max_slim_len
                fat_count += fat_len

                fat2slim.append(_fat2slim)
                slim2fat.append(_slim2fat)

            fat2slim = torch.cat(fat2slim, dim=0)
            slim2fat = pad_sequence(slim2fat, batch_first=True, padding_value=-1)
            return fat2slim, slim2fat

        def prep_graph_mask(true_slen):
            _b_size_ = len(batch)
            if batch[0].graph_mask == None: return None
            _num_seqs = len(batch[0].graph_mask)
            graph_mask = try_gpu(torch.zeros(_b_size_, _num_seqs, true_slen))
            for b_id, ex in enumerate(batch):
                if ex.graph_mask == None: return None
                for seq_id, src_seq in enumerate(ex.graph_mask):
                    curr_len = len(src_seq)
                    graph_mask[b_id, seq_id, :curr_len] = torch.tensor(src_seq)
            graph_mask = graph_mask.view(_b_size_, -1)
            return graph_mask.byte()

        def prep_graph_A(slim_len):
            _b_size_ = len(batch)
            graph_A = try_gpu(torch.zeros(_b_size_, slim_len, slim_len))
            for b_id, ex in enumerate(batch):
                curr_nodes = len(ex.graph_A)
                assert (curr_nodes <= slim_len)
                graph_A[b_id, :curr_nodes, :curr_nodes] = torch.tensor(ex.graph_A)
            graph_A = 1 - graph_A
            return graph_A.byte()

        # Get word embedding
        _true_slen = max(cod_slen, msg_slen)
        code_indices, code_wembs = self.embed_stuff_for_wembs(cod_stuff, _true_slen)
        msg_indices, msg_wembs = self.embed_stuff_for_wembs(msg_stuff, _true_slen)
        _b_size, _num_lines, _cod_slen, _wembdim = code_wembs.size()
        _, _msg_slen, _ = msg_wembs.size()

        # Concat absolute positional emb
        code_wembs = code_wembs.view(_b_size, -1, _wembdim)
        pos_embs = self.positional_encoding([0] * _b_size, code_wembs.size(1))
        code_wembs = torch.cat([code_wembs, pos_embs], dim=2)
        _wembdim2 = code_wembs.size(2)

        msg_mask = self.create_mask(msg_indices).unsqueeze(1)
        pos_embs = self.positional_encoding([0] * _b_size, msg_indices.size(1))
        msg_wembs = torch.cat([msg_wembs, pos_embs], dim=2)

        # LSTM encoding1
        code_wembs = code_wembs.view(_b_size, _num_lines, _true_slen, _wembdim2)
        msg_embeds1, msg_embeds_c1, msg_wembs = self.embed_stuff_for_lstm(msg_indices, msg_wembs, self.msg_emb1)
        code_embeds1, code_embeds_c1, code_wembs = self.embed_stuff_for_lstm(code_indices, code_wembs, self.cod_emb1)
        msg_wembs = self.msg_lin1(msg_wembs)
        code_wembs = self.cod_lin1(code_wembs)

        _b_size, _num_lines, _true_slen, _wembdim = code_wembs.size()

        # Line-level Positional encoding with error index information
        pos_embeds = self.positional_encoding(err_line, num_lines)
        pos_embeds = pos_embeds.unsqueeze(2).repeat(1, 1, _true_slen, 1)
        code_wembs = self.pos_enc(torch.cat([code_wembs, pos_embeds], dim=3))

        msg_code_graph_mask = prep_graph_mask(_true_slen)
        if msg_code_graph_mask is None:
            pass
        else:
            # Reshape and make code_wembs slim (remove tokens not in graph)
            code_indices = code_indices.view(_b_size, -1)
            code_wembs = code_wembs.view(_b_size, -1, _wembdim)

            msg_code_wembs_orig = torch.cat([msg_wembs, code_wembs], dim=1)
            fat2slim, slim2fat = prepare_for_removing_pad(msg_code_graph_mask)
            msg_code_wembs_w_dummy = torch.cat([try_gpu(torch.zeros(1, _wembdim)).float(), msg_code_wembs_orig.view(-1, _wembdim)], dim=0)
            msg_code_wembs = F.embedding(slim2fat + 1, msg_code_wembs_w_dummy)
            max_slim_len = slim2fat.size(1)

            # Attention (graph)
            msg_code_graph_A = prep_graph_A(max_slim_len)
            msg_code_wembs = self.graph_attention(msg_code_wembs, msg_code_graph_A)
            _, _, _out_dim = msg_code_wembs.size()
            msg_code_wembs_w_dummy = torch.cat([try_gpu(torch.zeros(1, _out_dim)).float(), msg_code_wembs.view(-1, _out_dim)], dim=0)
            msg_code_wembs = F.embedding(fat2slim + 1, msg_code_wembs_w_dummy)
            msg_code_wembs = msg_code_wembs * msg_code_graph_mask.unsqueeze(2).float() + msg_code_wembs_orig * (
                        msg_code_graph_mask == 0).unsqueeze(2).float()
            msg_code_wembs = msg_code_wembs.view(_b_size, 1 + _num_lines, _true_slen, _out_dim)
            msg_wembs = msg_code_wembs[:, 0].contiguous()
            code_wembs = msg_code_wembs[:, 1:].contiguous()

        code_indices = code_indices.view(_b_size, _num_lines, _true_slen)

        # LSTM encoding2
        msg_embeds2, msg_embeds_c2, msg_embeds_output = self.embed_stuff_for_lstm(msg_indices, msg_wembs, self.msg_emb2)
        msg_embeds = torch.cat([msg_embeds1, msg_embeds2], dim=1)
        msg_embeds_c = torch.cat([msg_embeds_c1, msg_embeds_c2], dim=1)
        code_embeds2, code_embeds_c2, code_embeds_output = self.embed_stuff_for_lstm(code_indices, code_wembs, self.cod_emb2)
        code_embeds = torch.cat([code_embeds1, code_embeds2], dim=2)
        code_embeds_c = torch.cat([code_embeds_c1, code_embeds_c2], dim=2)

        # Concatenate everything
        combo = torch.cat([code_embeds, msg_embeds.unsqueeze(1).expand(-1, num_lines, -1)], dim=2)
        combo = self.com_lay(combo)

        # LSTM on top
        line_seq_hidden, _ = self.lin_emb(combo)

        all_enc_stuff = [combo, line_seq_hidden]
        all_enc_stuff += [gold_code]
        all_enc_stuff += [code_embeds, code_embeds_c, code_indices, code_embeds_output]
        all_enc_stuff += [msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output]

        return all_enc_stuff

    def forward_localize(self, batch, all_enc_stuff):
        combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output, msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output = all_enc_stuff

        # Compute logits
        final_input = line_seq_hidden
        final_input = torch.cat([combo, final_input], dim=2)
        final = self.fin_lay(final_input).squeeze(2)

        label = [ex.gold_linenos for ex in batch]
        label = try_gpu(torch.tensor(label))
        label = label.float() / label.sum(dim=1, keepdim=True).float()
        localization_label = label
        localization_out = final

        return localization_out, localization_label

    def forward_edit(self, batch, all_enc_stuff, train_mode=True, beam_size=1, edit_lineno_specified=None):
        self.beam_size = beam_size

        combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output, msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output = all_enc_stuff

        gold_linenos = []
        if edit_lineno_specified is None:
            for ex in batch:
                gold_linenos.append(ex.edit_linenos)
        else:
            gold_code_line_stuff = []

            for b_id, ex in enumerate(batch):
                lidx = edit_lineno_specified[b_id]
                tgt_seq = ex.gold_code_lines[lidx].code

                gold_code_line_stuff.append(ex.gold_code_lines[lidx].code_idxs)

                edit_linenos = [0] * len(ex.edit_linenos)
                edit_linenos[lidx] = 1
                gold_linenos.append(edit_linenos)
                ex.edit_linenos = edit_linenos

        gold_linenos = try_gpu(torch.tensor(gold_linenos))
        gold_linenos_onehot = gold_linenos.unsqueeze(2).float()

        def get_oneline_vecs(gold_linenos_onehot, embeds_h, embeds_c, embeds_output, indices):
            oneline_h = (embeds_h * gold_linenos_onehot).sum(dim=1, keepdim=False)
            oneline_c = (embeds_c * gold_linenos_onehot).sum(dim=1, keepdim=False)
            oneline_enc_output = (embeds_output * gold_linenos_onehot.unsqueeze(3)).sum(dim=1, keepdim=False)
            oneline_indices = (indices.float() * gold_linenos_onehot).sum(dim=1, keepdim=False)
            b_size = oneline_h.size(0)
            lstm_dim = 200
            oneline_h = oneline_h.view(b_size, -1, lstm_dim).transpose(0,1)
            oneline_c = oneline_c.view(b_size, -1, lstm_dim).transpose(0,1)
            oneline_enc_output = oneline_enc_output.transpose(0,1)
            oneline_indices = oneline_indices.transpose(0,1)
            return oneline_h, oneline_c, oneline_enc_output, oneline_indices

        def format_tensor_length(in_tensor, true_slen):
            sizes = list(in_tensor.size())
            orig_slen = sizes[2]
            if orig_slen > true_slen:
                ret = in_tensor[:, :, :true_slen]
            else:
                sizes[2] = true_slen
                ret = try_gpu(torch.zeros(*sizes).fill_(PAD_INDEX))
                ret[:, :, :orig_slen] = in_tensor
            return Variable(ret).float()

        def prep_src_map_all_lines(true_slen):
            src_vocabs = []
            src_map = []
            for ex in batch:
                src_vocabs.append(ex.src_vocab)
                __src_map = []
                for src_seq in ex.src_map:
                    curr_len = len(src_seq)
                    if curr_len > true_slen:
                        padded_src_seq = src_seq[:true_slen]
                    else:
                        padded_src_seq = src_seq + [0]*(true_slen-curr_len)
                    __src_map.append(padded_src_seq)
                src_map.append(__src_map)
            src_map = try_gpu(torch.tensor(src_map)).transpose(0,1).transpose(1,2).contiguous()
            _num_seqs, _slen, _b_size = src_map.size()
            max_id = torch.max(src_map)
            src_map = make_one_hot(src_map.view(-1), max_id+1).view(-1, _b_size, max_id+1)
            src_map = Variable(src_map)
            return src_vocabs, src_map

        _b_size = line_seq_hidden.size(0)
        for_dec_init_h = self.bridge_h(torch.cat([code_embeds, line_seq_hidden], dim=2))
        for_dec_init_c = self.bridge_c(torch.cat([code_embeds_c, line_seq_hidden], dim=2))
        dec_init_h, dec_init_c, code_oneline_enc_output, code_oneline_indices = get_oneline_vecs(gold_linenos_onehot, for_dec_init_h, for_dec_init_c, code_embeds_output, code_indices)

        true_slen = code_indices.size(2)
        src_vocabs, src_map = prep_src_map_all_lines(true_slen)
        _msg_indices = format_tensor_length(msg_indices.unsqueeze(1), true_slen)
        _msg_embeds_output = format_tensor_length(msg_embeds_output.unsqueeze(1), true_slen)
        all_src_indices = torch.cat([_msg_indices.long(), code_indices.long()], dim=1)
        all_enc_output  = torch.cat([_msg_embeds_output, code_embeds_output], dim=1)
        _, _, _, __dim = all_enc_output.size()
        all_src_indices = all_src_indices.view(_b_size, -1).transpose(0,1)
        all_enc_output  = all_enc_output.view(_b_size, -1, __dim).transpose(0,1)
        packed_dec_input = [dec_init_h, dec_init_c, all_enc_output, all_src_indices, gold_code_line_stuff]

        dec_output, padded_gold_code_line = self.forward_helper_decode(batch, packed_dec_input, src_vocabs, src_map, train_mode)

        edit_out = dec_output
        edit_label = padded_gold_code_line

        return edit_out, edit_label

    def forward_helper_decode(self, batch, packed_dec_input, src_vocabs, src_map, train_mode):
        enc_h, enc_c, enc_output, src_indices, gold_code_line_stuff = packed_dec_input
        batch_size = enc_output.size(1)
        gold_code_line = [torch.tensor([BOS_INDEX]+ seq +[EOS_INDEX]) for seq in gold_code_line_stuff]
        padded_gold_code_line = try_gpu(pad_sequence(gold_code_line))

        gold_max_seq_len = len(padded_gold_code_line)

        _, _, enc_dim = enc_h.size()
        enc_h = enc_h.transpose(0,1).view(batch_size, 4, -1)
        enc_h = enc_h.transpose(0,1).contiguous()
        enc_c = enc_c.transpose(0,1).view(batch_size, 4, -1)
        enc_c = enc_c.transpose(0,1).contiguous()
        hidden = (enc_h, enc_c)

        if train_mode:
            output_tokens = padded_gold_code_line
            teacher_forcing_ratio = float(train_mode)
        else:
            output_tokens = try_gpu(torch.zeros((max(100, gold_max_seq_len), batch_size)).long().fill_(BOS_INDEX))
            teacher_forcing_ratio = 0
        input_tokens = src_indices

        vocab_size = self.decoder.output_size

        max_seq_len = len(output_tokens)
        dynamic_vocab_size = vocab_size + src_map.size(2)
        outputs = try_gpu(torch.zeros(max_seq_len, batch_size, dynamic_vocab_size))

        output = output_tokens[0,:]

        # Create mask (used for attention)
        mask = self.create_mask(input_tokens).transpose(0,1)

        extra_feed = None
        context_vec = None

        if train_mode:
            for t in range(1, max_seq_len):
                output = output.unsqueeze(0)
                output, hidden, attn, context_vec = self._decode_and_generate_one_step(output, hidden, enc_output, mask, context_vec, extra_feed, src_vocabs, src_map, beam_size=1, collapse=False)

                outputs[t] = output
                teacher_force = (random.random() < teacher_forcing_ratio)
                top1 = output.max(1)[1]
                output = (output_tokens[t] if teacher_force else top1)

            return outputs, padded_gold_code_line

        else:
            if self.beam_size > 1:
                allHyp, allScores = self.beam_decode(hidden, enc_output, mask, extra_feed, src_vocabs, src_map)
                return (allHyp, allScores), None
            else:
                outputs = self.greedy_decode(hidden, enc_output, mask, extra_feed, src_vocabs, src_map)
                return outputs, None

    def _decode_and_generate_one_step(self, decoder_in, hidden_state, memory_bank, mask, context_vec, extra_feed, src_vocabs, src_map, beam_size=1, collapse=False, batch_offset=None):
        decoder_in = decoder_in.masked_fill(decoder_in.gt(len(self.vocab) - 1), UNK_INDEX)
        dec_out, hidden, dec_attn, context_vec = self.decoder(decoder_in, hidden_state, memory_bank, mask, context_vec, extra_feed)
        attn = dec_attn["copy"]
        scores = self.cop_gen(dec_out, attn.view(-1, attn.size(2)), src_map)
        scores = scores.view(beam_size, -1, scores.size(-1)).transpose(0,1)
        if collapse:
            scores = collapse_copy_scores(scores, None, self.vocab_x, src_vocabs, batch_dim=0, batch_offset=batch_offset)
        scores = scores.view(-1, scores.size(-1))

        return scores, hidden, attn, context_vec

    def greedy_decode(self, enc_hidden, enc_output, mask, extra_feed, src_vocabs, src_map):
        batch_size = enc_output.size(1)
        hidden = enc_hidden
        output_tokens = try_gpu(torch.zeros(100, batch_size)).long().fill_(BOS_INDEX)
        output = output_tokens[0,:]
        vocab_size = self.decoder.output_size
        dynamic_vocab_size = vocab_size + src_map.size(2)
        outputs = try_gpu(torch.zeros(100, batch_size, dynamic_vocab_size))
        context_vec = None
        for t in range(1, 100):
            output = output.unsqueeze(0)
            output, hidden, attn, context_vec = self._decode_and_generate_one_step(output, hidden, enc_output, mask, context_vec, extra_feed, src_vocabs, src_map, beam_size=1, collapse=True)
            outputs[t] = output
            teacher_force = 0
            top1 = output.max(1)[1]
            output = (output_tokens[t] if teacher_force else top1)
            if (output.item() == EOS_INDEX):
                return outputs
        return outputs

    def beam_decode(self, enc_hidden, enc_output, mask, extra_feed, src_vocabs, src_map):
        beam_size = self.beam_size
        batch_size = enc_output.size(1)

        src_map = Variable(src_map.data.repeat(1, beam_size, 1))
        enc_h_t, enc_c_t = enc_hidden
        dec_states = [Variable(enc_h_t.data.repeat(1, beam_size, 1)), Variable(enc_c_t.data.repeat(1, beam_size, 1))]
        memory_bank = Variable(enc_output.data.repeat(1, beam_size, 1))
        memory_mask = Variable(mask.data.repeat(beam_size, 1))
        extra_feed_ = Variable(extra_feed.data.repeat(1, beam_size, 1)) if extra_feed is not None else None

        max_length = 100
        beam = BeamSearch(beam_size, n_best=beam_size, batch_size=batch_size, global_scorer=None,
                          pad=PAD_INDEX, eos=EOS_INDEX, bos=BOS_INDEX, min_length=0, ratio=0, max_length=max_length,
                          mb_device="cuda", return_attention=False, stepwise_penalty=False, block_ngram_repeat=0,
                          exclusion_tokens=[], memory_lengths=None)
        context_vec = None
        for step in range(max_length):
            input = beam.current_predictions.view(1, -1)
            scores, (trg_h_t, trg_c_t), attn, context_vec = self._decode_and_generate_one_step(input, (dec_states[0], dec_states[1]), memory_bank, memory_mask, context_vec, extra_feed_, src_vocabs, src_map, beam_size, collapse=True, batch_offset=beam._batch_offset)
            dec_states = (trg_h_t, trg_c_t)
            log_probs = scores.log()
            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break
            select_indices = beam.current_origin
            if any_beam_is_finished:
                memory_bank = memory_bank.index_select(1, select_indices)
                memory_mask = memory_mask.index_select(0, select_indices)
                src_map = src_map.index_select(1, select_indices)
                extra_feed_ = extra_feed_.index_select(1, select_indices) if extra_feed_ is not None else None
                dec_states = (dec_states[0].index_select(1, select_indices), dec_states[1].index_select(1, select_indices))
        allHyp, allScores = beam.predictions, beam.scores
        return allHyp, allScores

    def get_stuffs_to_embed(self, batch):
        code_stuff = []
        gold_code_line_stuff = []
        _code_slen = 0
        _msg_slen = 0
        msg_stuff = []
        err_linenos = []
        for ex in batch:
            code_stuff_sub = []
            for line in ex.code_lines:
                code_stuff_sub.append(line.code_idxs)
                _cur_code_slen = len(line.code_idxs) + 2 #BOS, EOS
                _code_slen = _cur_code_slen if _cur_code_slen > _code_slen else _code_slen
            code_stuff.append(code_stuff_sub)
            lidx = np.argmax(np.array(ex.edit_linenos))
            gold_code_line_stuff.append(ex.gold_code_lines[lidx].code_idxs)

            msg_stuff.append(ex.err_line.msg_idxs)
            err_linenos.append(ex.err_line.lineno)
            _cur_msg_slen = len(ex.err_line.msg_idxs) + 2 #BOS, EOS
            _msg_slen = _cur_msg_slen if _cur_msg_slen > _msg_slen else _msg_slen
        return (code_stuff, _code_slen), (msg_stuff, _msg_slen), err_linenos, gold_code_line_stuff

    def embed_stuff_for_wembs(self, stuff, true_slen):

        def pad_sequence_with_length(sequences, true_slen, batch_first=False, padding_value=0):
            max_size = sequences[0].size()
            trailing_dims = max_size[1:]
            max_len = max([s.size(0) for s in sequences])
            max_len = max_len if true_slen==0 else true_slen
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims
            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        _b_size = len(stuff)
        if isinstance(stuff[0][0], list):
            all_seq_indices = []
            for batch in stuff:
                for seq in batch:
                    token_indices = [BOS_INDEX] + seq + [EOS_INDEX]
                    token_indices = torch.tensor(token_indices)
                    all_seq_indices.append(token_indices)
            padded_token_indices = try_gpu(pad_sequence_with_length(all_seq_indices, true_slen))
        else:
            all_seq_indices = []
            for seq in stuff:
                token_indices = [BOS_INDEX] + seq + [EOS_INDEX]
                token_indices = torch.tensor(token_indices)
                all_seq_indices.append(token_indices)
            padded_token_indices = try_gpu(pad_sequence_with_length(all_seq_indices, true_slen))

        embedded_tokens = self.tok_emb(padded_token_indices)
        embedded_tokens = self.dropout(embedded_tokens)

        _seqlen_, _b_, _dim_ = embedded_tokens.size()
        padded_token_indices = padded_token_indices.transpose(0,1)
        padded_token_indices = padded_token_indices.view(_b_size, -1, _seqlen_).squeeze(1)

        embedded_tokens = embedded_tokens.transpose(0,1) #(`batch`, seqlen, embed_dim)
        embedded_tokens = embedded_tokens.view(_b_size, -1, _seqlen_, _dim_).squeeze(1) #(batch, seqlen, dim) or (batch, num_lines, seqlen, dim)

        return padded_token_indices.contiguous(), embedded_tokens.contiguous()

    def embed_stuff_for_lstm(self, inp_indices, inp_wembs, seq_embedder):
        if len(inp_wembs.size()) == 4:
            _2d_flag = True
            _b_size, _num_lines, true_slen, _wembdim = inp_wembs.size()
            inp_wembs = inp_wembs.view(-1, true_slen, _wembdim)
            inp_indices = inp_indices.view(-1, true_slen)
        else:
            _2d_flag = False
            _b_size, true_slen, _wembdim = inp_wembs.size()
            _num_lines = None

        inp_mask = (inp_indices != PAD_INDEX)
        inp_length = inp_mask.sum(dim=1)
        inp_wembs = inp_wembs.transpose(0,1)

        #LSTM
        seq_input, hx, rev_order, mask = prepare_rnn_seq(inp_wembs, inp_length, hx=None, masks=None, batch_first=False)
        seq_output, hn = seq_embedder(seq_input)
        lstm_output, (h_n, c_n) = recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=False)
        _num_seqs = lstm_output.size(1)
        lstm_output_fullypadded = try_gpu(torch.zeros((true_slen, _num_seqs, lstm_output.size(2)))).float()
        lstm_output_fullypadded[:lstm_output.size(0)] = lstm_output

        # Arrange dimenstions
        lstm_out_h = h_n.transpose(0, 1).reshape(_num_seqs, -1)
        lstm_out_c = c_n.transpose(0, 1).reshape(_num_seqs, -1)
        lstm_output = lstm_output_fullypadded.transpose(0, 1).reshape(_num_seqs, true_slen, -1)

        if _2d_flag:
            lstm_out_h = lstm_out_h.view(_b_size, _num_lines, -1)
            lstm_out_c = lstm_out_c.view(_b_size, _num_lines, -1)
            lstm_output = lstm_output.view(_b_size, _num_lines, true_slen, -1)

        return lstm_out_h, lstm_out_c, lstm_output

    def positional_encoding(self, err_linenos, num_lines):
        err_linenos = torch.tensor(err_linenos)
        offsets = torch.arange(num_lines).unsqueeze(0) - err_linenos.unsqueeze(1)
        offsets = try_gpu(offsets.float())

        # Build arguments for sine and cosine
        coeffs = try_gpu(torch.arange(100 / 2.))
        coeffs = torch.pow(10000., - coeffs / 100)
        arguments = offsets.unsqueeze(2) * coeffs
        result = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=2)
        return result

    def get_loss_localization(self, logit, label, batch):
        loss = cross_entropy_after_probsum(logit, label.float())
        return loss

    def get_loss_edit(self, dec_output, padded_gold_code_line, batch, force_copy_loss=None):
        true_tlen, b_size, dynamic_vocab_size = dec_output.size()
        scores = dec_output.view(-1, dynamic_vocab_size)
        target = padded_gold_code_line.view(-1)

        align = []
        for ex in batch:
            align.append(torch.tensor(ex.align).long())

        _align = try_gpu(pad_sequence(align, padding_value=UNK_INDEX))
        _tlen = _align.size(0)
        align = try_gpu(torch.zeros((true_tlen, b_size)).long().fill_(UNK_INDEX))
        align[:_tlen] = _align
        align = align.view(-1)
        align = Variable(align)

        loss = self.cop_gen_loss(scores, align, target, b_size, force_copy_loss)

        return loss

    def get_pred_localization(self, logit, batch, train_mode=True):
        return torch.argmax(logit, dim=1)

    def get_pred_edit(self, dec_output, batch, train_mode=True, retAllHyp=False):
        if train_mode:
            return torch.argmax(dec_output, dim=2, keepdim=False).transpose(0,1)
        else:
            if isinstance(dec_output, tuple):
                allHyp, allScores = dec_output
                if not retAllHyp:
                    return [torch.tensor(hyps[0]) for hyps in allHyp]
                else:
                    return [allHyp, allScores]
            else:
                return torch.argmax(dec_output, dim=2, keepdim=False).transpose(0,1)
