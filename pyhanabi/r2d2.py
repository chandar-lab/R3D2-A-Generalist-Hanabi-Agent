import torch
import json
import torch.nn as nn
from typing import Tuple, Dict
from net import PublicLSTMNet, LSTMNet, MHANet, TextLSTMNet


class R2D2Agent(torch.jit.ScriptModule):
    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        device,
        in_dim,
        hid_dim,
        out_dim,
        net,
        num_lstm_layer,
        lm_weights="random",
        num_of_player=2,
        num_of_additional_layer=0,
        num_lm_layer=1,
        lora_dim=128,
        off_belief=False,
    ):
        super().__init__()
        self.out_dim = out_dim
        if net == "publ-lstm":
            self.online_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
            self.target_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        elif net == "lstm":
            self.online_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
            self.target_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        elif net == "drrn-lstm":
            self.online_net = TextLSTMNet(
                device,
                in_dim,
                hid_dim,
                1,
                num_lstm_layer,
                lm_weights,
                num_of_player,
                num_lm_layer,
                lora_dim,
            ).to(device)
            self.target_net = TextLSTMNet(
                device,
                in_dim,
                hid_dim,
                1,
                num_lstm_layer,
                lm_weights,
                num_of_player,
                num_lm_layer,
                lora_dim,
            ).to(device)
        elif net == "text-input-lstm":
            self.online_net = TextLSTMNet(
                device,
                in_dim,
                hid_dim,
                out_dim,
                num_lstm_layer,
                lm_weights,
                num_of_player,
                num_lm_layer,
                lora_dim,
            ).to(device)
            self.target_net = TextLSTMNet(
                device,
                in_dim,
                hid_dim,
                out_dim,
                num_lstm_layer,
                lm_weights,
                num_of_player,
                num_lm_layer,
                lora_dim,
            ).to(device)
        else:
            assert False, f"{net} not implemented"

        for p in self.target_net.parameters():
            p.requires_grad = False

        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.net = net
        self.num_lstm_layer = num_lstm_layer
        self.off_belief = off_belief
        self.device = device
        self.lm_weights = lm_weights
        self.num_of_player = num_of_player
        self.lora_dim = lora_dim
        self.num_of_additional_layer = num_of_additional_layer
        self.num_lm_layer = num_lm_layer

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}

        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.net,
            self.num_lstm_layer,
            self.lm_weights,
            self.num_of_player,
            self.num_of_additional_layer,
            self.num_lm_layer,
            self.lora_dim,
            self.off_belief,
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move[:, : adv.shape[1]]
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid, {"adv": adv, "legal_move": legal_move}

    @torch.jit.script_method
    def pikl_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
        pikl_lambda: torch.Tensor,
        bp_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pikl_lambda = pikl_lambda.unsqueeze(1)
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        assert adv.size() == bp_logits.size()
        pikl_adv = adv + pikl_lambda * bp_logits
        legal_adv = pikl_adv - (1 - legal_move) * 1e10

        extra = {
            "adv": adv,
            "bp_logits": bp_logits,
            "pikl_lambda": pikl_lambda,
            "legal_adv": legal_adv,
            "legal_move": legal_move,
        }
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid, extra

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        if self.net == "publ-lstm" or self.net == "lstm":
            priv_s = obs["priv_s"].to(self.device)
            publ_s = priv_s[:, 125:]  # obs["publ_s"]

        else:
            priv_s = obs["priv_s_text"].to(self.device)
            publ_s = priv_s[:, 125:]  # obs["publ_s"]
        legal_move = obs["legal_move"].to(self.device)
        if "eps" in obs:
            eps = obs["eps"].flatten(0, 1).to(self.device)
        else:
            eps = torch.zeros((priv_s.size(0),), device=self.device)

        # converge it hid to from batch first to batch second
        # hid size: [batch, num_layer, num_player, dim] -> [num_layer, batch x num_player, dim]
        # if len(obs) > 0:
        batch, num_layer, num_player, rnn_dim = obs["h0"].size()
        hid = {
            "h0": obs["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": obs["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }
        if "llm_prior" in obs:
            pikl_lambda = obs["pikl_lambda"]
            llm_prior = obs["llm_prior"]
            greedy_action, new_hid, extra = self.pikl_act(
                priv_s, publ_s, legal_move, hid, pikl_lambda, llm_prior
            )
        else:
            greedy_action, new_hid, extra = self.greedy_act(
                priv_s, publ_s, legal_move.to(self.device), hid
            )

        reply = {}
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).float()
        action = greedy_action.detach().long()

        reply["a"] = action.detach().cpu()

        for k, v in new_hid.items():
            v = v.transpose(0, 1).view(batch, num_layer, num_player, rnn_dim)
            reply[k] = v.detach().cpu()

        return reply

    @torch.jit.script_method
    def td_error(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        reply: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_seq_len = obs["priv_s"].size(0)
        if self.net == "publ-lstm" or self.net == "lstm":
            priv_s = obs["priv_s"].to(self.device)
        else:
            priv_s = obs["priv_s_text"].to(self.device)

        legal_move = obs["legal_move"].to(self.device)
        action = reply["a"]

        if self.net == "lstm":
            if self.vdn:
                num_player = priv_s.size(2)
                priv_s = priv_s.flatten(1, 2)
                legal_move = legal_move.flatten(1, 2)
                action = action.flatten(1, 2)
            publ_s = priv_s[:, :, 125:]
        else:
            priv_s = torch.transpose(priv_s, 1, 2)
            publ_s = priv_s

        bsize, num_player = priv_s.size(1), 1
        for k, v in hid.items():
            hid[k] = v.flatten(1, 2).contiguous()

        online_qa, greedy_a, online_q, lstm_o = self.online_net(
            priv_s, publ_s, legal_move, action, hid
        )

        if self.off_belief:
            target = obs["target"]
        else:
            target_qa, _, _, _ = self.target_net(
                priv_s, publ_s, legal_move, greedy_a, hid
            )
            if self.vdn:
                target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)

            target_qa = torch.cat(
                [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
            )
            target_qa[-self.multi_step :] = 0
            assert target_qa.size() == reward.size()
            target = reward + bootstrap * (self.gamma**self.multi_step) * target_qa

        if self.vdn:
            online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
            lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        if self.off_belief:
            err = err * obs["valid_fict"]
        return err, lstm_o, online_q

    def aux_task(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        if self.vdn:
            seq_size, bsize, num_player, _ = hand.size()
            own_hand = hand.view(seq_size, bsize, num_player, 5, 3)
        else:
            seq_size, bsize, _ = hand.size()
            own_hand = hand.view(seq_size, bsize, 5, 3)
        own_hand_slot_mask = own_hand.sum(-1)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size
        if stat is not None:
            stat["aux"].feed(avg_xent1)
        return pred_loss1

    def loss(self, batch, aux_weight, stat):
        err, lstm_o, _ = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.bootstrap,
            batch.seq_len,
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        if stat is not None:
            stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())
            stat["game_len"].feed(batch.seq_len.mean())

        loss = rl_loss
        if aux_weight <= 0:
            return loss

        pred1 = self.aux_task(
            lstm_o,
            batch.obs["own_hand"],
            batch.seq_len,
            rl_loss.size(),
            stat,
        )
        loss = rl_loss + aux_weight * pred1
        return loss
