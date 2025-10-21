import abc
from typing import Any, Tuple

import torch
from habitat_baselines.rl.ppo.policy import Policy
from torch import Size, Tensor
from thop import profile

from sim2sim_vlnce.Sim2Sim.vln_action_mapping import (
    create_candidate_features,
    idx_to_action,
)


class CustomFixedCategorical(torch.distributions.Categorical):
    """Same as the CustomFixedCategorical in hab-lab, but renames log_probs
    to log_prob. All the torch distributions use log_prob.
    """

    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class VLNPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.action_distribution = None
        self.critic = None

    def forward(self, *x):
        raise NotImplementedError

    def encode_instruction(self, tokens, mask) -> Tuple[Tensor, Tensor]:
        """Generates the first hidden state vector h_t and encodes each
        instruction token. Call once for episode initialization.

        Returns:
            h_t (Tensor): [B x hidden_size]
            instruction_features (Tensor): [B x max_len x hidden_size]
        """
        return self.net.vln_bert("language", tokens, lang_mask=mask)

    def encode_instruction_efficient(self, tokens, mask, episode_ids, gflops_tracker) -> Tuple[Tensor, Tensor]:
        """Generates the first hidden state vector h_t and encodes each
        instruction token. Call once for episode initialization. Also
        tracks the GFLOPs for each episode.

        Returns:
            h_t (Tensor): [B x hidden_size]
            instruction_features (Tensor): [B x max_len x hidden_size]
        """
        h_t, instruction_features = None, None
        for i in range(len(episode_ids)):
            _tokens = tokens[i].unsqueeze(0)
            _mask = mask[i].unsqueeze(0)
            
            (_h_t, _instruction_features), flops, _ = profile(
                self.net.vln_bert, 
                inputs=("language", _tokens, None, None, _mask, None, None, None),
                verbose=False
            ) # no keyword arguments for thop :(

            if h_t is None:
                h_t = torch.zeros((tokens.size(0), *_h_t[0].shape)).to(_h_t.device)
                instruction_features = torch.zeros((tokens.size(0), *_instruction_features[0].shape)).to(_instruction_features.device)

            h_t[i] = _h_t[0].detach()
            instruction_features[i] = _instruction_features[0].detach()

            gflops_tracker[episode_ids[i]]["instruction_encoder"] += flops / 1e9

        return (h_t, instruction_features)

    def act(
        self,
        observations,
        h_t,
        instruction_features,
        instruction_mask,
        deterministic=False,
    ):
        instruction_features = torch.cat(
            (h_t.unsqueeze(1), instruction_features[:, 1:, :]), dim=1
        )
        (
            vis_features,
            coordinates,
            vis_mask,
        ) = create_candidate_features(observations)

        h_t, action_logit = self.net(
            instruction_features=instruction_features,
            attention_mask=torch.cat((instruction_mask, vis_mask), dim=1),
            lang_mask=instruction_mask,
            vis_mask=vis_mask,
            cand_feats=vis_features,
            action_feats=observations["mp3d_action_angle_feature"],
        )

        # Mask candidate logits that have no associated action
        action_logit.masked_fill(vis_mask, -float("inf"))
        distribution = CustomFixedCategorical(logits=action_logit)

        if deterministic:
            action_idx = distribution.mode()
        else:
            action_idx = distribution.sample()

        return h_t, idx_to_action(action_idx, coordinates)

    def act_efficient(
        self,
        observations,
        h_t,
        instruction_features,
        instruction_mask,
        deterministic=False,
        epsiode_ids=None,
        gflops_tracker=None,
    ):
        instruction_features = torch.cat(
            (h_t.unsqueeze(1), instruction_features[:, 1:, :]), dim=1
        )
        (
            vis_features,
            coordinates,
            vis_mask,
        ) = create_candidate_features(observations)

        attention_mask = torch.cat((instruction_mask, vis_mask), dim=1)
        
        h_t, action_logit = None, None
        for i in range(len(epsiode_ids)):
            _instruction_features = instruction_features[i].unsqueeze(0)
            _attention_mask = attention_mask[i].unsqueeze(0)
            _lang_mask = instruction_mask[i].unsqueeze(0)
            _vis_mask = vis_mask[i].unsqueeze(0)
            _cand_feats = vis_features[i].unsqueeze(0)
            _action_feats = observations["mp3d_action_angle_feature"][i].unsqueeze(0)

            (_h_t, _action_logit), flops, _ = profile(
                self.net, 
                inputs=(_instruction_features, _attention_mask, _lang_mask, _vis_mask, _cand_feats, _action_feats),
                verbose=False
            )
            
            if h_t is None:
                h_t = torch.zeros((len(epsiode_ids), *_h_t[0].shape)).to(_h_t.device)
                action_logit = torch.zeros((len(epsiode_ids), *_action_logit[0].shape)).to(_action_logit.device)
            
            h_t[i] = _h_t[0].detach()
            action_logit[i] = _action_logit[0].detach()

            gflops_tracker[epsiode_ids[i]]["policy"] += flops / 1e9

        # Mask candidate logits that have no associated action
        action_logit.masked_fill(vis_mask, -float("inf"))
        distribution = CustomFixedCategorical(logits=action_logit)

        if deterministic:
            action_idx = distribution.mode()
        else:
            action_idx = distribution.sample()

        return h_t, idx_to_action(action_idx, coordinates)

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self,
        observations,
        h_t,
        instruction_features,
        instruction_mask,
    ) -> CustomFixedCategorical:
        instruction_features = torch.cat(
            (h_t.unsqueeze(1), instruction_features[:, 1:, :]), dim=1
        )
        vis_features, _, vis_mask = create_candidate_features(observations)

        h_t, action_logit = self.net(
            instruction_features=instruction_features,
            attention_mask=torch.cat((instruction_mask, vis_mask), dim=1),
            lang_mask=instruction_mask,
            vis_mask=vis_mask,
            cand_feats=vis_features,
            action_feats=observations["mp3d_action_angle_feature"],
        )

        # Mask candidate logits that have no associated action
        action_logit.masked_fill(vis_mask, -float("inf"))
        return h_t, CustomFixedCategorical(logits=action_logit)
