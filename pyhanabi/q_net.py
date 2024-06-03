import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np
from transformers import BertConfig, BertLMHeadModel, BertForPreTraining
from transformers import BertTokenizer, DistilBertModel, BertModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer



@torch.jit.script
def duel(v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor) -> torch.Tensor:
    assert a.size() == legal_move.size()
    assert legal_move.dim() == 3  # seq, batch, dim
    legal_a = a * legal_move
    # NOTE: this may cause instability
    # avg_legal_a = legal_a.sum(2, keepdim=True) / legal_move.sum(2, keepdim=True).clamp(min=0.1)
    # q = v + legal_a - avg_legal_a
    # NOTE: this is fine
    # q = v + legal_a - legal_a.mean(2, keepdim=True)
    # NOTE: this fake dueling is also fine
    q = v + legal_a
    return q


def cross_entropy(net, lstm_o, target_p, hand_slot_mask, seq_len):
    # target_p: [seq_len, batch, num_player, 5, 3]
    # hand_slot_mask: [seq_len, batch, num_player, 5]
    logit = net(lstm_o).view(target_p.size())
    q = nn.functional.softmax(logit, -1)
    logq = nn.functional.log_softmax(logit, -1)
    plogq = (target_p * logq).sum(-1)
    xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(min=1e-6)

    if xent.dim() == 3:
        # [seq, batch, num_player]
        xent = xent.mean(2)

    # save before sum out
    seq_xent = xent
    xent = xent.sum(0)
    assert xent.size() == seq_len.size()
    avg_xent = (xent / seq_len).mean().item()
    return xent, avg_xent, q, seq_xent.detach()


class LSTMNet(torch.jit.ScriptModule):
    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        ff_layers = [nn.Linear(self.priv_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2
        # print(priv_s.shape)
        priv_s = priv_s.unsqueeze(0)
        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.net(priv_s)
        if len(hid) == 0:
            o, _ = self.lstm(x)
        else:
            o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class MultiHeadAttention(torch.jit.ScriptModule):
    def __init__(self, d_model, num_head):
        super().__init__()
        assert d_model % num_head == 0

        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = np.sqrt(self.d_head)

    @torch.jit.script_method
    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        seq_len, batch, _ = x.size()
        qkv = self.qkv_proj(x).view(seq_len, batch, 3, self.num_head, self.d_head)
        q, k, v = qkv.permute((1, 2, 3, 0, 4)).unbind(1)
        # q, k, v = einops.rearrange(
        #     qkv, "t b (k h d) -> b k h t d", k=3, h=self.num_head
        # ).unbind(1)
        # attn_v = torch.nn.functional.scaled_dot_product_attention(
        #     q, k, v, dropout_p=0.0, is_causal=False
        # )
        score = torch.einsum("bhtd,bhsd->bhts", q, k) / self.scale
        # prod = (q * k).sum(3) / self.scale  # prod: [batch, num_head, seq]
        attn = torch.nn.functional.softmax(score, -1)
        attn_v = torch.einsum("bhts,bhsd->bhtd", attn, v)
        attn_v = attn_v.permute((2, 0, 1, 3)).flatten(start_dim=2)

        # attn_v = einops.rearrange(attn_v, "b h t d -> t b (h d)")
        return self.out_proj(attn_v)


class SelfAttention(torch.jit.ScriptModule):
    def __init__(self, input_dim, num_head):
        super().__init__()

        self.input_dim = input_dim
        self.num_head = num_head

        self.k_proj = nn.Linear(self.input_dim, self.input_dim)
        self.multi_proj = nn.Linear(self.input_dim, self.num_head * self.input_dim)
        self.mha = MultiHeadAttention(self.input_dim, 1)  # use 1 head for MHA for now
        self.scale = np.sqrt(input_dim)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        # x: [seq, batch, input_dim]
        seq, batch, input_dim = x.size()
        multi_proj = self.multi_proj(x).view(seq * batch, self.num_head, input_dim)
        # multi_proj: [seq * batch, num_head, input_dim] -> [num_head, seq * batch, input_dim]
        multi_proj = multi_proj.permute((1, 0, 2))
        attn_v = self.mha(multi_proj)  # attn_v: [num_head, seq * batch, input_dim]
        attn_v = attn_v.permute((1, 0, 2)).view(seq, batch, self.num_head, input_dim)

        # compute q_weights
        k_proj = self.k_proj(x).unsqueeze(2)
        prod = (k_proj * attn_v).sum(3) / self.scale
        q_weight = nn.functional.softmax(prod, 2) # q_weight: [seq, batch, num_head]
        return attn_v, q_weight


class MHANet(torch.jit.ScriptModule):
    def __init__(self, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        ff_layers = [nn.Linear(self.priv_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.sa_module = SelfAttention(self.hid_dim, 4)

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # # for aux task
        # self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, 1)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2

        priv_s = priv_s.unsqueeze(0)
        x = self.net(priv_s)
        o, q_weight = self.sa_module(x)  # o: [seq, batch, num_head, dim]
        a = self.fc_a(o)  # q: [seq, batch, num_head, num_action]
        a = (a * q_weight.unsqueeze(3)).sum(2)
        a = a.squeeze(0)
        return a, hid

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.net(priv_s)
        o, q_weight = self.sa_module(x)  # o: [seq, batch, num_head, dim]
        q = self.fc_v(o) + self.fc_a(o)  # q: [seq, batch, num_head, num_action]
        q = (q * q_weight.unsqueeze(3)).sum(2)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    # def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
    #     return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class PublicLSTMNet(torch.jit.ScriptModule):
    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        self.in_dim = in_dim
        self.priv_in_dim = in_dim[1]
        self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2
        priv_s = priv_s.unsqueeze(0)
        publ_s = publ_s.unsqueeze(0)

        priv_o = self.priv_net(priv_s)
        x = self.publ_net(publ_s)
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))

        o = priv_o * publ_o
        a = self.fc_a(o)
        a = a.squeeze(0)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class TextLSTMNet(torch.jit.ScriptModule):
    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility

        self.in_dim = in_dim
        self.device = device
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer
        self.path = '/home/mila/n/nekoeiha/scratch/llm_hanabi_hive/2p_action_ids.json'
        self.act_tok = self.load_json(self.path)
        # self.act_tok = torch.tensor(self.act_tokens['input_ids'], device=device)

        if self.out_dim == 1:

            # self.net = BertForPreTraining.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2',
            #                                               output_hidden_states=True)
            vocab_size = 30522
            self.embedding = nn.Embedding(vocab_size, hid_dim)
            # self.net = MultiHeadAttention(self.hid_dim, 1)
            # self.mlp = nn.Linear(self.hid_dim)
            self.net = SelfAttentionEmbedding(hid_dim, 1)
            self.action_lstm = nn.LSTM(
                self.hid_dim,
                self.hid_dim,
                num_layers=self.num_lstm_layer
            ).to(device)
            self.action_lstm.flatten_parameters()

        else:
            ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
            for i in range(1, self.num_ff_layer):
                ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
                ff_layers.append(nn.ReLU())
            self.net = nn.Sequential(*ff_layers)

        self.state_lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.state_lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)
        self.call_transformer = self.load_transformers()
        # self.compiled_transformers = torch.jit.load('/home/mila/a/arjun.vaithilingam-sudhakar/scratch/hanabi_may24/Zeroshot_hanabi_instructrl/pyhanabi/bert_traced.pt')
        # self.compiled_transformers_device = self.compiled_transformers.to_device('cuda:1')
        self.example = torch.tensor([[ 101, 7592, 1010, 2129, 2024, 2017, 1029,  102]])

    def load_transformers(self):

        model = BertModel.from_pretrained('bert-base-uncased')
        model.to('cuda:1')
        model.eval()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dummy_input = tokenizer("Hello, this is a TorchScript test", return_tensors='pt')
        input_ids = dummy_input['input_ids'].to('cuda:1')

        # Trace the model with strict=False
        traced_model = torch.jit.trace(model, input_ids, strict=False)
        return traced_model
        # Save the traced model
        # traced_model.save("bert_traced.pt")

    def load_json(self, path):
        with open(path) as f:
            d = json.load(f)
        # temp = torch.tensor(d['input_ids'],device=self.device)
        return torch.tensor(d['input_ids'],device=self.device).transpose(1,0)
    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2
        # priv_s = torch.randint(low=0, high=1718, size=(128, hid["h0"].shape[1]), device=priv_s.device)
        print('priv_s q_net',priv_s.shape)
        with torch.no_grad():
            outputs = self.call_transformer(priv_s)

        print(outputs['last_hidden_state'].shape)
        priv_s = torch.transpose(priv_s, 0, 1).to(self.device)


        embed = self.embedding(priv_s)
        x = self.net(embed)
        # print('priv - x', x.shape)
        # print('X.shape', x.shape)

        # hid = self.get_h0(1)
        o, (h, c) = self.state_lstm(x, (hid["h0"].to(priv_s.device), hid["c0"].to(priv_s.device)))
        o = o[-1, :, :]
        if self.out_dim == 1:
            # priv_s = torch.transpose(priv_s, 1, 0)
            # with open('/home/mila/a/arjun.vaithilingam-sudhakar/scratch/hanabi_may_15/Zeroshot_hanabi_instructrl/pyhanabi/action_tokens/2p_action_ids.json') as f:
            #     d = json.load(f)
            # act_tok = torch.Tensor(d['input_ids'], device=priv_s.device)
            # act_tok = torch.randint(low=0, high=1718, size=(16, 21), device=priv_s.device)
            # print('q_net forward 522', self.act_tok.shape)
            act_embed = self.embedding(self.act_tok)
            x = self.net(act_embed)
            hid_action = self.get_h0(21)
            o_action, (h_act, c_act) = self.action_lstm(x, (hid_action["h0"].to(priv_s.device), hid_action["c0"].to(priv_s.device)))
            o_action = o_action[-1, :, :]
            o = o_action.repeat(hid["h0"].shape[1], 1) * o.repeat(21, 1)
            o = o.view(hid["h0"].shape[1]*21, -1)

        a = self.fc_a(o)
        a = a.view(hid["h0"].shape[1], -1)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        # if priv_s.dim() == 2:
        #     priv_s = priv_s.unsqueeze(0)
        #     publ_s = publ_s.unsqueeze(0)
        #     legal_move = legal_move.unsqueeze(0)
        #     action = action.unsqueeze(0)
        #     one_step = True
        # print("priv s shape before embed", priv_s.shape)
        priv_s = priv_s.to(self.device)
        embed = self.embedding(priv_s.to(self.device))
        embed = embed.reshape(-1, embed.shape[-2], embed.shape[-1])
        # print("priv s shape after embed", embed.shape)

        x = self.net(embed)
        # print(x.shape)
        if len(hid) == 0:
            o, _ = self.state_lstm(x)
        else:
            o, _ = self.state_lstm(x, (hid["h0"], hid["c0"]))

        # print(o.shape)
        o = o[::128, :, :]
        # print(o.shape)
        # print('***')
        if self.out_dim == 1:
            # print('q_net forward 573',priv_s.shape)

            # act_tok = torch.randint(low=0, high=1718, size=(16, 21), device=priv_s.device)
            # print('act_tok shape in 589' ,self.act_tok.shape)
            act_embed = self.embedding(self.act_tok)
            x = self.net(act_embed)
            hid_action = self.get_h0(21)
            o_action, (h_act, c_act) = self.action_lstm(x, (
            hid_action["h0"].to(priv_s.device), hid_action["c0"].to(priv_s.device)))
            o_action = o_action[-1, :, :].unsqueeze(1)
            # print(o_action.shape)
            o_repeated = o_action.repeat(o.shape[0], o.shape[1], 1) * o.repeat(21, 1, 1)
            o_repeated = o_repeated.view(-1, o.shape[-1])
        else:
            o_repeated = o

        # print(o.shape)
        a = self.fc_a(o_repeated).view(legal_move.size())
        v = self.fc_v(o).view(legal_move.shape[0], legal_move.shape[1], 1)
        # print(a.shape)
        # print(legal_move.shape)
        # print(v.shape)
        q = duel(v, a, legal_move)

        # print(q.shape)
        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class SelfAttentionEmbedding(torch.jit.ScriptModule):
    def __init__(self, input_dim, num_head):
        super().__init__()

        self.input_dim = input_dim
        self.num_head = num_head

        self.k_proj = nn.Linear(self.input_dim, self.input_dim)
        self.multi_proj = nn.Linear(self.input_dim, self.num_head * self.input_dim)
        self.mha = MultiHeadAttention(self.input_dim, 1)  # use 1 head for MHA for now
        self.scale = np.sqrt(input_dim)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        # x: [seq, batch, input_dim]
        seq, batch, input_dim = x.size()
        multi_proj = self.multi_proj(x).view(seq * batch, self.num_head, input_dim)
        # multi_proj: [seq * batch, num_head, input_dim] -> [num_head, seq * batch, input_dim]
        multi_proj = multi_proj.permute((1, 0, 2))
        attn_v = self.mha(multi_proj)  # attn_v: [num_head, seq * batch, input_dim]
        attn_v = attn_v.permute((1, 0, 2)).view(seq, batch, self.num_head, input_dim)

        # compute q_weights
        k_proj = self.k_proj(x).unsqueeze(2)
        prod = (k_proj * attn_v).sum(3) / self.scale
        q_weight = nn.functional.softmax(prod, 2) # q_weight: [seq, batch, num_head]

        # compute weighted sum of attention values across all heads
        attn_v_weighted = (q_weight.unsqueeze(3) * attn_v).sum(2)  # [seq, batch, input_dim]

        return attn_v_weighted
