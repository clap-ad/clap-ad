import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lead.tfv6.speed_score_utils import lateral_shift_paths


class HydraSpeedHead(nn.Module):
    def __init__(
            self, 
            embed_dims=256,
            num_speed_bins=8,
            num_modes=1,
            planning_steps=6,
            path_aug_cnt=None,
            combined_attention=True,
            fuser_dim=2,
            common_context_encoder=True,
            flags_shapes_dict={'ttc': 0, 'tl': 0, 'st': 0},
            common_score_mlp=True,
        ):
        """Initializes the HydraSpeedHead module.
        This head is responsible for predicting scores for future speeds
        based on encoded agent, environment, and context information.
        Args:
            embed_dims (int): The dimensionality of the input embeddings.
            Defaults to 256.
            num_speed_bins (int): The number of discrete bins to represent the
            speed distribution. Defaults to 8.
            num_modes (int): The number of different future modes to predict.
            Defaults to 1.
            planning_steps (int): The number of future time steps in the planned
            path, used to configure the context encoder. Defaults to 6.
            path_aug_cnt (int, optional): The count for path augmentation.
            Defaults to None.
            combined_attention (bool): Flag for using combined attention.
            If false, traffic light and ttc scores are predicted separately with necessary inputs only.
            Defaults to True.
            fuser_dim (int): The dimensionality of the fusion layer. 2 for UniAD and 3 for VAD
            common_context_encoder (bool): Whether to use a common context encoder for both ttc and traffic light branches when combined_attention is False.
            flags_shapes (dict): The shape of flags passed to the context encoder, if any. E.g., {'tl': 4, 'st': 2}.
        """
        super(HydraSpeedHead, self).__init__()
        self.embed_dims = embed_dims
        self.combined_attention = combined_attention
        self.num_modes = num_modes
        self.common_context_encoder = common_context_encoder
        self.common_score_mlp = common_score_mlp
        self.flags_shapes_dict = flags_shapes_dict
        if self.flags_shapes_dict is None:
            self.flags_shapes_dict = {'ttc': 0, 'tl': 0, 'st': 0, 'ot': 0}

        # self.agent_fuser = AgentFuser(embed_dims=self.embed_dims, fuser_dim=fuser_dim)
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True)
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True)
        )

        if self.combined_attention:
            # context: path output (6 * 2), ego speed
            self.context_encoder = ContextEncoder(num_path=planning_steps * 2, num_flags=0, flag_emb_dim=8, hidden_dim=self.embed_dims, out_dim=self.embed_dims)     
            self.speed_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
            
            self.speed_queries = nn.Parameter(torch.randn(num_modes, num_speed_bins, self.embed_dims))

            self.prob_mlp = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims // 2),
                nn.LayerNorm(self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, 1)
            )

        else:
            assert num_modes > 1
            if self.common_context_encoder:
                # context: path output (6 * 2) only
                self.context_encoder = ContextEncoder(num_path=planning_steps * 2, num_flags=0, flag_emb_dim=8, hidden_dim=self.embed_dims, out_dim=self.embed_dims, use_speed=False)
            else:
                self.ttc_context_encoder = ContextEncoder(num_path=planning_steps * 2, num_flags=self.flags_shapes_dict.get("ttc"), flag_emb_dim=8, hidden_dim=self.embed_dims, out_dim=self.embed_dims, use_speed=False)
                self.tl_context_encoder = ContextEncoder(num_path=planning_steps * 2, num_flags=self.flags_shapes_dict.get("tl"), flag_emb_dim=8, hidden_dim=self.embed_dims, out_dim=self.embed_dims, use_speed=False)
                if num_modes == 3:
                    self.st_context_encoder = ContextEncoder(num_path=planning_steps * 2, num_flags=self.flags_shapes_dict.get("st"), flag_emb_dim=8, hidden_dim=self.embed_dims, out_dim=self.embed_dims, use_speed=False)
                if num_modes == 4:
                    self.st_context_encoder = ContextEncoder(num_path=planning_steps * 2, num_flags=self.flags_shapes_dict.get("st"), flag_emb_dim=8, hidden_dim=self.embed_dims, out_dim=self.embed_dims, use_speed=False)
                    self.ot_context_encoder = ContextEncoder(num_path=planning_steps * 2, num_flags=self.flags_shapes_dict.get("ot"), flag_emb_dim=8, hidden_dim=self.embed_dims, out_dim=self.embed_dims, use_speed=False)
                
            
            if self.common_score_mlp:
                self.prob_mlp = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims // 2),
                    nn.LayerNorm(self.embed_dims // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims // 2, 1)
                )
            else:
                self.ttc_prob_mlp = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims // 2),
                    nn.LayerNorm(self.embed_dims // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims // 2, 1)
                )
                self.tl_prob_mlp = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims // 2),
                    nn.LayerNorm(self.embed_dims // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims // 2, 1)
                )
                if num_modes == 3:
                    self.st_prob_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims, self.embed_dims // 2),
                        nn.LayerNorm(self.embed_dims // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_dims // 2, 1)
                    )
                if num_modes == 4:
                    self.st_prob_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims, self.embed_dims // 2),
                        nn.LayerNorm(self.embed_dims // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_dims // 2, 1)
                    )
                    self.ot_prob_mlp = nn.Sequential(
                        nn.Linear(self.embed_dims, self.embed_dims // 2),
                        nn.LayerNorm(self.embed_dims // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.embed_dims // 2, 1)
                    )
            
            
            self.ttc_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
            self.tl_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)

            self.ttc_queries = nn.Parameter(torch.randn(1, num_speed_bins, self.embed_dims))
            self.tl_queries = nn.Parameter(torch.randn(1, num_speed_bins, self.embed_dims))
            if num_modes == 3:
                # stop sign branch
                self.st_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
                self.st_queries = nn.Parameter(torch.randn(1, num_speed_bins, self.embed_dims))
            if num_modes == 4:
                # stop sign branch
                self.st_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
                self.st_queries = nn.Parameter(torch.randn(1, num_speed_bins, self.embed_dims))
                # overtake branch
                self.ot_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
                self.ot_queries = nn.Parameter(torch.randn(1, num_speed_bins, self.embed_dims))

        self.path_aug_cnt = path_aug_cnt

    def get_context_token(self, context_type, *args, **kwargs):
        """
        Wrapper to select the correct context encoder.
        Args:
            context_type (str): 'ttc', 'tl', or 'st'
            *args, **kwargs: Arguments to pass to the encoder
        Returns:
            torch.Tensor: Context token
        """
        if self.common_context_encoder:
            return self.context_encoder(*args, **kwargs)
        elif context_type == 'ttc':
            return self.ttc_context_encoder(*args, **kwargs)
        elif context_type == 'tl':
            return self.tl_context_encoder(*args, **kwargs)
        elif context_type == 'st':
            return self.st_context_encoder(*args, **kwargs)
        elif context_type == 'ot':
            return self.ot_context_encoder(*args, **kwargs)
        else:
            raise ValueError(f"Unknown context_type: {context_type}")
        
    def get_scores_from_query(self, score_type, *args, **kwargs):
        """
        Wrapper to select the correct score MLP.
        Args:
            score_type (str): 'ttc', 'tl', or 'st'
            *args, **kwargs: Arguments to pass to the MLP
        Returns:
            torch.Tensor: Score tensor
        """
        if self.common_score_mlp:
            return self.prob_mlp(*args, **kwargs)
        elif score_type == 'ttc':
            return self.ttc_prob_mlp(*args, **kwargs)
        elif score_type == 'tl':
            return self.tl_prob_mlp(*args, **kwargs)
        elif score_type == 'st':
            return self.st_prob_mlp(*args, **kwargs)
        else:
            raise ValueError(f"Unknown score_type: {score_type}")


    def forward(self, plan_query, path_pred, radar_feat=None):
        path_pred = path_pred.detach()
        
        speed_logits_ttc, path_pred_augmented = self._forward_time_to_collision_tfv6(plan_query, path_pred)  # [B, NUM_PATHS, MODES=1, S]
        speed_logits_tl = self._forward_traffic_light_tfv6(plan_query, path_pred)  # [B, NUM_PATHS=1, MODES=1, S]

        NUM_PATHS = speed_logits_ttc.shape[1]
        speed_logits = torch.cat([speed_logits_ttc, speed_logits_tl.expand(-1, NUM_PATHS, -1, -1)], dim=2)  # [B, NUM_PATHS, 2, S]

        if self.num_modes == 3:
            speed_logits_st = self._forward_stop_sign_tfv6(plan_query, path_pred)  # [B, NUM_PATHS=1, MODES=1, S]
            speed_logits = torch.cat([speed_logits, speed_logits_st.expand(-1, NUM_PATHS, -1, -1)], dim=2)  # [B, NUM_PATHS, 3, S]

        if self.num_modes == 4:
            speed_logits_st = self._forward_stop_sign_tfv6(plan_query, path_pred)  # [B, NUM_PATHS=1, MODES=1, S]
            speed_logits = torch.cat([speed_logits, speed_logits_st.expand(-1, NUM_PATHS, -1, -1)], dim=2)  # [B, NUM_PATHS, 4, S]

            speed_logits_ot = self._forward_overtake_tfv6(plan_query, path_pred)  # [B, NUM_PATHS=1, MODES=1, S]
            speed_logits = torch.cat([speed_logits, speed_logits_ot.expand(-1, NUM_PATHS, -1, -1)], dim=2)  # [B, NUM_PATHS, 4, S]

        return speed_logits, path_pred_augmented

    
    def _forward_time_to_collision_tfv6(self, plan_query, path_pred):
        """
        Speed prediction forward pass.
        Accepts agent queries and context inputs for ttc score.

        Args:
            plan_query (torch.Tensor): Plan_query output (B, HW, C) C is the embed dim
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
        Note:
            In encoding part, basically we view each pixel as an object with its own features.
            Previously in self.static_encoder(plan_query) it was [B, Ns, C],
            but now it is [B, HW, C].
        Returns:
            torch.Tensor: Speed logits [B, NUM_PATHS, 1, S].
        """
        # NOTE: Do not call directly because detaching is handled in forward()
        attn_input = []

        if self.path_aug_cnt is not None:  # If we want to aug the predicted path
            path_pred = lateral_shift_paths(path_pred)  # [B, Aug_cnt, T, 2]
            context_token_per_path = []
            for i in range(self.path_aug_cnt):
                # context_token = self.context_encoder(path_pred[:, i], speed, flags)  # [B, C]
                context_token = self.get_context_token('ttc', path_pred[:, i])  # [B, C]
                context_token_per_path.append(context_token.unsqueeze(1))
            context_token = torch.cat(context_token_per_path, dim=1)  # [B, Aug_cnt, C]
        else:
            # context_token = self.context_encoder(path_pred, speed, flags).unsqueeze(1)  # [B, 1, C]
            context_token = self.get_context_token('ttc', path_pred).unsqueeze(1)  # [B, 1, C]

        B, NUM_PATHS = context_token.shape[0], context_token.shape[1]
        fused_feat = plan_query  # [B, HW, C]
        agent_tokens = self.agent_encoder(fused_feat)  # [B, HW, C]

        # Append agent tokens for attention input
        attn_input.append(agent_tokens)

        # only use ttc queries
        speed_queries = self.ttc_queries  # [1, S, C]
        _, S, C = speed_queries.shape

        if NUM_PATHS > 1:
            # reshape attention input so we can process all paths in parallel
            if len(attn_input) > 0:
                attn_input = torch.cat(attn_input, dim=1)
                attn_input = attn_input.unsqueeze(1).expand(-1, NUM_PATHS, -1, -1)  # [B, NUM_PATHS, N, C]
                context_tokens = context_token.unsqueeze(2)  # [B, NUM_PATHS, 1, C]
                attn_input = torch.cat([attn_input, context_tokens], dim=2)  # [B, NUM_PATHS, N+1, C]
            else:
                attn_input = context_token.unsqueeze(2)  # [B, NUM_PATHS, 1, C]
            attn_input = attn_input.reshape(B * NUM_PATHS, -1, C)  # [B*NUM_PATHS, N+1, C]
            speed_queries = speed_queries.expand(B * NUM_PATHS, -1, -1)  # [B*NUM_PATHS, S, C]
            speed_tokens, _ = self.ttc_attn(speed_queries, attn_input, attn_input)  # [B*NUM_PATHS, S, C]
            speed_logits = self.get_scores_from_query('ttc', speed_tokens)  # [B*NUM_PATHS, S, 1]
            speed_logits = speed_logits.reshape(B, NUM_PATHS, S, 1)  # [B, NUM_PATHS, S, 1]
        else:
            if context_token is not None:
                attn_input.append(context_token)
            attn_input = torch.cat(attn_input, dim=1)  # [B, N+1, C]
            speed_queries = speed_queries.expand(B, -1, -1)  # [B, S, C]
            speed_tokens, _ = self.ttc_attn(speed_queries, attn_input, attn_input)  # [B, S, C]
            speed_logits = self.get_scores_from_query('ttc', speed_tokens)  # [B, S, 1]

        return speed_logits.reshape(B, NUM_PATHS, 1, S), path_pred  # [B, NUM_PATHS, MODES=1, S]
    

    def _forward_traffic_light_tfv6(self, plan_query, path_pred):
        """
        Speed prediction forward pass.
        Accepts traffic light queries and context inputs for tl score.

        Args:
            plan_query (torch.Tensor): Plan query output (B, HW, C) 
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
        Note:
            In encoding part, basically we view each pixel as an object with its own features.
            Previously in self.static_encoder(plan_query) it was [B, Ns, C],
            but now it is [B, HW, C].
        Returns:
            torch.Tensor: Speed logits [B, 1, 1, S].
        """
        # NOTE: Do not call directly because detaching is handled in forward()
        B = path_pred.shape[0]
        attn_input = []

        context_token = self.get_context_token('tl', path_pred).unsqueeze(1)  # [B, 1, C]
        attn_input.append(context_token)

        # object path (traffic light score)
        B, Ns, C = plan_query.shape
        obj_tokens = self.static_encoder(plan_query)  # [B, Ns, C]
        attn_input.append(obj_tokens)
        
        # only use tl queries
        speed_queries = self.tl_queries # [1, S, C]
        _, S, C = speed_queries.shape

        attn_input = torch.cat(attn_input, dim=1)  # [B, Ns+1, C]
        speed_queries = speed_queries.expand(B, -1, -1)  # [B, S, C]
        speed_tokens, _ = self.tl_attn(speed_queries, attn_input, attn_input)  # [B, S, C]
        speed_logits = self.get_scores_from_query('tl', speed_tokens)  # [B, S, 1]

        return speed_logits.reshape(B, 1, 1, S)  # [B, NUM_PATHS=1, MODES=1, S]
        
        
    def _forward_stop_sign_tfv6(self, plan_query, path_pred):
        """
        Speed prediction forward pass.
        Accepts stop sign and context inputs for stop sign score.

        Args:
            plan_query (torch.Tensor): Plan query output (B, HW, C)
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
        Note:
            In encoding part, basically we view each pixel as an object with its own features.
            Previously in self.static_encoder(plan_query) it was [B, Ns, C],
            but now it is [B, HW, C].
        Returns:
            torch.Tensor: Speed logits [B, 1, 1, S].
        """
        # NOTE: Do not call directly because detaching is handled in forward()
        B = path_pred.shape[0]
        attn_input = []

        context_token = self.get_context_token('st', path_pred).unsqueeze(1)  # [B, 1, C]
        attn_input.append(context_token)

        # object path (stop sign score)
        B, Ns, C = plan_query.shape
        obj_tokens = self.static_encoder(plan_query)  # [B, Ns, C]
        attn_input.append(obj_tokens)
        
        # only use st queries
        speed_queries = self.st_queries # [1, S, C]
        _, S, C = speed_queries.shape

        attn_input = torch.cat(attn_input, dim=1)  # [B, Ns+1, C]
        speed_queries = speed_queries.expand(B, -1, -1)  # [B, S, C]
        speed_tokens, _ = self.st_attn(speed_queries, attn_input, attn_input)  # [B, S, C]
        speed_logits = self.get_scores_from_query('st', speed_tokens)  # [B, S, 1]

        return speed_logits.reshape(B, 1, 1, S)  # [B, NUM_PATHS=1, MODES=1, S]

    def _forward_overtake_tfv6(self, plan_query, path_pred):
        """
        Speed prediction forward pass.
        Accepts overtake and context inputs for overtake score.

        Args:
            plan_query (torch.Tensor): Plan query output (B, HW, C)
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
        Note:
            In encoding part, basically we view each pixel as an object with its own features.
            Previously in self.static_encoder(plan_query) it was [B, Ns, C],
            but now it is [B, HW, C].
        Returns:
            torch.Tensor: Speed logits [B, 1, 1, S].
        """
        # NOTE: Do not call directly because detaching is handled in forward()
        B = path_pred.shape[0]
        attn_input = []

        context_token = self.get_context_token('ot', path_pred).unsqueeze(1)  # [B, 1, C]
        attn_input.append(context_token)

        # object path (overtake score)
        B, Ns, C = plan_query.shape
        obj_tokens = self.static_encoder(plan_query)  # [B, Ns, C]
        attn_input.append(obj_tokens)

        # only use ot queries
        speed_queries = self.ot_queries # [1, S, C]
        _, S, C = speed_queries.shape

        attn_input = torch.cat(attn_input, dim=1)  # [B, Ns+1, C]
        speed_queries = speed_queries.expand(B, -1, -1)  # [B, S, C]
        speed_tokens, _ = self.ot_attn(speed_queries, attn_input, attn_input)  # [B, S, C]
        speed_logits = self.get_scores_from_query('ot', speed_tokens)  # [B, S, 1]

        return speed_logits.reshape(B, 1, 1, S)  # [B, NUM_PATHS=1, MODES=1, S]

    def _forward_time_to_collision(self, track_query, traj_query, traj_scores, path_pred, speed=None, flags=None):
        """
        Speed prediction forward pass.
        Accepts agent queries and context inputs for ttc score.

        Args:
            track_query (torch.Tensor): TrackFormer output [B, N, C_TR]
            traj_query (torch.Tensor): MotionFormer output [B, N, P, C_TJ]
            traj_scores (torch.Tensor): Confidence scores in the trajectory of this modality. [B, N, P]
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
            speed (torch.Tensor): DISABLED Ego speed from can bus [B, 1]
            flags (torch.Tensor): Traffic light and stop sign flags [B, 2]
        Returns:
            torch.Tensor: Speed logits [B, NUM_PATHS, 1, S].
        """
        # NOTE: Do not call directly because detaching is handled in forward()
        attn_input = []
        
        if self.path_aug_cnt is not None: # If we want to aug the predicted path
            path_pred = lateral_shift_paths(path_pred) # [B, Aug_cnt, T, 2]
            context_token_per_path = []
            for i in range(self.path_aug_cnt):
                # context_token = self.context_encoder(path_pred[:, i], speed, flags)  # [B, C]
                context_token = self.get_context_token('ttc', path_pred[:, i], speed, flags)  # [B, C]
                context_token_per_path.append(context_token.unsqueeze(1))
            context_token = torch.cat(context_token_per_path, dim=1)  # [B, Aug_cnt, C]
        else:
            # context_token = self.context_encoder(path_pred, speed, flags).unsqueeze(1)  # [B, 1, C]
            context_token = self.get_context_token('ttc', path_pred, speed, flags).unsqueeze(1)  # [B, 1, C]

        NUM_PATHS = context_token.shape[1]
        _, N, _ = traj_scores.shape # traj_scores correctly represents when there are no agents in scene
        if N == 0:
            # If there are no agents, we can skip prediction and output 10 for all speed bins (logits)
            B = path_pred.shape[0]
            S = self.ttc_queries.shape[1]
            return torch.ones(B, NUM_PATHS, 1, S, device=path_pred.device) * 10, path_pred
        
        
        B, N, P, C_TJ = traj_query.shape
        B, _, C_TR = track_query.shape
        C = C_TR # The target shape

        # Flatten agents into batch dimension so we can encode all agents at once
        track_flat = track_query.reshape(B * N, C_TR)          # [B*N, C_TR]
        traj_flat = traj_query.reshape(B * N, P, C_TJ)         # [B*N, P, C_TJ]
        scores_flat = traj_scores.reshape(B * N, P)  # [B*N, P]

        fused_feat = self.agent_fuser(track_flat, traj_flat, scores_flat)  # [B*N, C]
        fused_feat = fused_feat.view(B, N, C)                              # [B, N, C]

        agent_tokens = self.agent_encoder(fused_feat)  # [B, N, C]

        # Append agent tokens for attention input
        attn_input.append(agent_tokens)

        # only use ttc queries
        speed_queries = self.ttc_queries  # [1, S, C]
        _, S, C = speed_queries.shape

        if NUM_PATHS > 1:
            # reshape attention input so we can process all paths in parallel
            if len(attn_input) > 0:
                attn_input = torch.cat(attn_input, dim=1)
                attn_input = attn_input.unsqueeze(1).expand(-1, NUM_PATHS, -1, -1)  # [B, NUM_PATHS, N, C]
                context_tokens = context_token.unsqueeze(2)  # [B, NUM_PATHS, 1, C]
                attn_input = torch.cat([attn_input, context_tokens], dim=2)  # [B, NUM_PATHS, N+1, C]
            else:
                attn_input = context_token.unsqueeze(2)  # [B, NUM_PATHS, 1, C]
            attn_input = attn_input.reshape(B * NUM_PATHS, -1, C)  # [B*NUM_PATHS, N+1, C]
            speed_queries = speed_queries.expand(B * NUM_PATHS, -1, -1)  # [B*NUM_PATHS, S, C]
            speed_tokens, _ = self.ttc_attn(speed_queries, attn_input, attn_input)  # [B*NUM_PATHS, S, C]
            speed_logits = self.get_scores_from_query('ttc', speed_tokens)  # [B*NUM_PATHS, S, 1]
            speed_logits = speed_logits.reshape(B, NUM_PATHS, S, 1)  # [B, NUM_PATHS, S, 1]
        else:
            if context_token is not None:
                attn_input.append(context_token)
            attn_input = torch.cat(attn_input, dim=1)  # [B, N+1, C]
            speed_queries = speed_queries.expand(B, -1, -1)  # [B, S, C]
            speed_tokens, _ = self.ttc_attn(speed_queries, attn_input, attn_input)  # [B, S, C]
            speed_logits = self.get_scores_from_query('ttc', speed_tokens)  # [B, S, 1]

        return speed_logits.reshape(B, NUM_PATHS, 1, S), path_pred  # [B, NUM_PATHS, MODES=1, S]
    
    def _forward_traffic_light(self, obj_track_query, path_pred, flags=None):
        """
        Speed prediction forward pass.
        Accepts traffic light queries and context inputs for tl score.

        Args:
            obj_track_query (torch.Tensor): TrackFormer queries for static objects (traffic light, stop sign, speed sign) [B, N_s, C]
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
            flags (torch.Tensor): Traffic light and stop sign flags [B, 2]
        Returns:
            torch.Tensor: Speed logits [B, 1, 1, S].
        """
        # NOTE: Do not call directly because detaching is handled in forward()
        B = path_pred.shape[0]
        attn_input = []
        if obj_track_query is None or obj_track_query.shape[1] == 0:
            # If there are no traffic lights, we can skip prediction and output 10 for all speed bins (logits)
            S = self.tl_queries.shape[1]
            return torch.ones(B, 1, 1, S, device=path_pred.device) * 10

        # context_token = self.context_encoder(path_pred, speed=None, flags=flags).unsqueeze(1)  # [B, 1, C]
        context_token = self.get_context_token('tl', path_pred, speed=None, flags=flags).unsqueeze(1)  # [B, 1, C]
        attn_input.append(context_token)

        # object path (traffic light score)
        B, Ns, C = obj_track_query.shape
        obj_tokens = self.static_encoder(obj_track_query)  # [B, Ns, C]
        attn_input.append(obj_tokens)
        
        # only use tl queries
        speed_queries = self.tl_queries # [1, S, C]
        _, S, C = speed_queries.shape

        attn_input = torch.cat(attn_input, dim=1)  # [B, Ns+1, C]
        speed_queries = speed_queries.expand(B, -1, -1)  # [B, S, C]
        speed_tokens, _ = self.tl_attn(speed_queries, attn_input, attn_input)  # [B, S, C]
        speed_logits = self.get_scores_from_query('tl', speed_tokens)  # [B, S, 1]

        return speed_logits.reshape(B, 1, 1, S)  # [B, NUM_PATHS=1, MODES=1, S]
        
    def _forward_stop_sign(self, obj_track_query, path_pred, flags=None):
        """
        Speed prediction forward pass.
        Accepts stop sign and context inputs for stop sign score.

        Args:
            obj_track_query (torch.Tensor): TrackFormer queries for static objects (traffic light, stop sign, speed sign) [B, N_s, C]
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
            flags (torch.Tensor): Traffic light and stop sign flags [B, 2]
        Returns:
            torch.Tensor: Speed logits [B, 1, 1, S].
        """
        # NOTE: Do not call directly because detaching is handled in forward()
        B = path_pred.shape[0]
        attn_input = []

        if obj_track_query is None or obj_track_query.shape[1] == 0:
            # If there are no stop signs, we can skip prediction and output 10 for all speed bins (logits)
            S = self.st_queries.shape[1]
            return torch.ones(B, 1, 1, S, device=path_pred.device) * 10

        # context_token = self.context_encoder(path_pred, speed=None, flags=flags).unsqueeze(1)  # [B, 1, C]
        context_token = self.get_context_token('st', path_pred, speed=None, flags=flags).unsqueeze(1)  # [B, 1, C]
        attn_input.append(context_token)

        # object path (stop sign score)
        B, Ns, C = obj_track_query.shape
        obj_tokens = self.static_encoder(obj_track_query)  # [B, Ns, C]
        attn_input.append(obj_tokens)
        
        # only use st queries
        speed_queries = self.st_queries # [1, S, C]
        _, S, C = speed_queries.shape

        attn_input = torch.cat(attn_input, dim=1)  # [B, Ns+1, C]
        speed_queries = speed_queries.expand(B, -1, -1)  # [B, S, C]
        speed_tokens, _ = self.st_attn(speed_queries, attn_input, attn_input)  # [B, S, C]
        speed_logits = self.get_scores_from_query('st', speed_tokens)  # [B, S, 1]

        return speed_logits.reshape(B, 1, 1, S)  # [B, NUM_PATHS=1, MODES=1, S]
        

    def _forward_combined(self, track_query, traj_query, traj_scores, obj_track_query, path_pred, speed=None, flags=None):
        """
        Combined speed prediction forward pass.
        Accepts either agent queries and context inputs for ttc score or
        only static object queries for traffic light score.

        Args:
            track_query (torch.Tensor): TrackFormer output [B, N, C]
            traj_query (torch.Tensor): MotionFormer output [B, N, P, C]
            traj_scores (torch.Tensor): Confidence scores in the trajectory of this modality. [B, N, P]
            speed (torch.Tensor): Ego speed from can bus [B, 1]
            path_pred (torch.Tensor): Path output offsets [B, 2 * planning_steps]
            flags (torch.Tensor): Traffic light and stop sign flags [B, 2]
            obj_track_query (torch.Tensor): TrackFormer queries for static objects (traffic light, stop sign, speed sign) [B, N_s, C]
        Returns:
            torch.Tensor: Speed logits [B, NUM_PATHS, M, S].
        """
        # [MotionFormer levels, batch size, number of agents, number of modalities, embedding dimension]
        # [B, N, P, C]
        # [1, N, 6, 256]
        # NOTE: Do not call directly because detaching is handled in forward()
        attn_input = []
        
        B = path_pred.shape[0]

        if self.path_aug_cnt is not None: # If we want to aug the predicted path
            path_pred = self.lateral_shift_paths(path_pred) # [B, Aug_cnt, T, 2]
            context_token_per_path = []
            for i in range(self.path_aug_cnt):
                # context_token = self.context_encoder(path_pred[:, i], speed, flags)  # [B, C]
                context_token = self.get_context_token('ttc', path_pred[:, i], speed, flags)  # [B, C]
                context_token_per_path.append(context_token.unsqueeze(1))
            context_token = torch.cat(context_token_per_path, dim=1)  # [B, Aug_cnt, C]
        else:
            # context_token = self.context_encoder(path_pred, speed, flags).unsqueeze(1)  # [B, 1, C]
            context_token = self.get_context_token('ttc', path_pred, speed, flags).unsqueeze(1)  # [B, 1, C]

        NUM_PATHS = context_token.shape[1]

        B, N, P, C_TJ = traj_query.shape
        _, N, _ = traj_scores.shape # traj_scores correctly represents when there are no agents in scene
        B, N, C_TR = track_query.shape
        C = C_TR # The target shape

        if N > 0:
            # Flatten agents into batch dimension so we can encode all agents at once
            track_flat = track_query.reshape(B * N, C_TR)          # [B*N, C_TR]
            traj_flat = traj_query.reshape(B * N, P, C_TJ)         # [B*N, P, C_TJ]
            scores_flat = traj_scores.reshape(B * N, P)  # [B*N, P]

            fused_feat = self.agent_fuser(track_flat, traj_flat, scores_flat)  # [B*N, C]
            fused_feat = fused_feat.view(B, N, C)                              # [B, N, C]

            agent_tokens = self.agent_encoder(fused_feat)  # [B, N, C]

            # Append agent tokens for attention input
            attn_input.append(agent_tokens)

        # object path (traffic light score)
        if obj_track_query is not None and obj_track_query.shape[1] != 0:
            B, Ns, C = obj_track_query.shape

            obj_tokens = self.static_encoder(obj_track_query)  # [B, Ns, C]

            attn_input.append(obj_tokens)
        
        # num_modes, speed_bins, embedding dimension
        M, S, C = self.speed_queries.shape
        speed_queries = self.speed_queries.reshape(M*S, C)  # [M*S, C]

        if NUM_PATHS > 1:
            # reshape attention input so we can process all paths in parallel
            if len(attn_input) > 0:
                attn_input = torch.cat(attn_input, dim=1)
                attn_input = attn_input.unsqueeze(1).expand(-1, NUM_PATHS, -1, -1)  # [B, NUM_PATHS, N+Ns, C]
                context_tokens = context_token.unsqueeze(2)  # [B, NUM_PATHS, 1, C]
                attn_input = torch.cat([attn_input, context_tokens], dim=2)  # [B, NUM_PATHS, N+Ns+1, C]
            else:
                attn_input = context_token.unsqueeze(2)  # [B, NUM_PATHS, 1, C]
            attn_input = attn_input.reshape(B * NUM_PATHS, -1, C)  # [B*NUM_PATHS, N+Ns+1, C]
            speed_queries = speed_queries.unsqueeze(0).expand(B * NUM_PATHS, -1, -1)  # [B*NUM_PATHS, M*S, C]
            speed_tokens, _ = self.speed_attn(speed_queries, attn_input, attn_input)  # [B*NUM_PATHS, M*S, C]
            speed_logits = self.prob_mlp(speed_tokens)  # [B*NUM_PATHS, M*S, 1]
            speed_logits = speed_logits.reshape(B, NUM_PATHS, M*S, 1)  # [B, NUM_PATHS, M*S, 1]
        else:
            if context_token is not None:
                attn_input.append(context_token)
            attn_input = torch.cat(attn_input, dim=1)  # [B, N+Ns+1, C]
            speed_queries = speed_queries.unsqueeze(0).expand(B, -1, -1)  # [B, M*S, C]
            speed_tokens, _ = self.speed_attn(speed_queries, attn_input, attn_input)  # [B, M*S, C]
            speed_logits = self.prob_mlp(speed_tokens)  # [B, M*S, 1]

        return speed_logits.reshape(B, NUM_PATHS, M, S), path_pred  # [B, NUM_PATHS, M, S]



class AgentFuser(nn.Module):
    def __init__(self, embed_dims, fuser_dim):
        """Initialize agent feature fuser for speed prediction."""
        super().__init__()
        self.embed_dims = embed_dims

        self.fuser_mlp = nn.Sequential(
            nn.Linear(self.embed_dims * fuser_dim, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.fuser_attn = ScoreBiasedAnchorAttention(self.embed_dims)

        self.fuser_query = nn.Parameter(torch.randn(1, self.embed_dims))

    def forward(self, track_query, traj_query, traj_scores):
        """
        Fuse agent features for speed prediction.

        Args:
            track_query (torch.Tensor): Single agent track query. [B, embed_dims]
            traj_query (torch.Tensor): Single agent traj query. [B, modalities, embed_dims]
            traj_scores (torch.Tensor): Confidence scores in the trajectory of this modality. [B, modalities]
        Returns:
            torch.Tensor: Fused agent features. [B, embed_dims]
        """
        
        B, P, C = traj_query.shape
        track_query = track_query[:, None].expand(-1,P,-1) # [B, C] -> [B, P, C]
        
        agent_feat = torch.cat([traj_query, track_query], dim=-1) # [B, P, 2C]

        agent_feat = self.fuser_mlp(agent_feat) # [B, P, 2C] -> [B, P, C]
        # agent_feat = rearrange(agent_feat, 'b m c -> m b c')

        agent_feat = self.fuser_attn(self.fuser_query.expand(B, -1), agent_feat, traj_scores)

        return agent_feat  # [B, C]


class ScoreBiasedAnchorAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.score_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, anchor_embeds, anchor_scores):
        """
        query:         [B, D]
        anchor_embeds: [B, K, D]
        anchor_scores: [B, K]  (logits or confidence scores)
        """

        # Project
        q = self.q_proj(query).unsqueeze(1)      # [B, 1, D]
        k = self.k_proj(anchor_embeds)           # [B, K, D]
        v = self.v_proj(anchor_embeds)           # [B, K, D]

        # Scaled dot-product attention
        attn_logits = torch.matmul(q, k.transpose(-1, -2))  # [B, 1, K]
        attn_logits = attn_logits / math.sqrt(k.size(-1))

        # Add score bias (belief injection)
        attn_logits = attn_logits + self.score_scale * anchor_scores.unsqueeze(1)

        # Normalize
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 1, K]

        # Fuse anchors
        fused = torch.matmul(attn_weights, v)  # [B, 1, D]

        return fused.squeeze(1)  # [B, D]


class ContextEncoder(nn.Module):
    def __init__(self, num_path=12, num_flags=0, flag_emb_dim=8, hidden_dim=256, out_dim=256, use_speed=True):
        """
        num_path: number of continuous x,y offsets (e.g., 2*6=12)
        num_flags: number of categorical/binary flags
        flag_emb_dim: embedding dimension for each flag (if categorical)
        hidden_dim: hidden size of intermediate MLP
        out_dim: output dimension of context token
        """
        super().__init__()
        self.use_speed = use_speed

        # Optional embedding for flags (if num_flags>0)
        # self.flag_embs = nn.ModuleList([nn.Embedding(2, flag_emb_dim) for _ in range(num_flags)]) if num_flags > 0 else None
        self.flag_embs = nn.Linear(num_flags, flag_emb_dim) if num_flags > 0 else None
        # Compute total input dimension: speed + path_pred + flags
        total_input_dim = num_path + (1 if use_speed else 0)
        if num_flags > 0:
            total_input_dim += flag_emb_dim

        # Simple MLP to encode context
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, path_pred, speed=None, flags=None):
        """
        Encodes context information for speed prediction.

        Args:
            speed (torch.Tensor): Ego speed, shape [B].
            path_pred (torch.Tensor): Predicted path offsets, shape [B, pred_horizon, 2].
            flags (torch.Tensor): Traffic light and stop sign flags, shape [B, num_flags] (optional).

        Returns:
            torch.Tensor: Context token of shape [B, out_dim].
        """
        flat_path = path_pred.reshape(path_pred.shape[0], -1)

        if self.use_speed:
            # Normalize speed
            speed_norm = speed.float() / 12.0
            # speed_norm = (speed - 5.0) / 10.0

            # Concatenate path and speed
            x = torch.cat([speed_norm.unsqueeze(1), flat_path], dim=-1)
        else:
            assert speed is None
            x = flat_path
        
        # Embed flags if present
        if self.flag_embs is not None and flags is not None:
            # flags is [B, num_flags], unbind splits it into a tuple of num_flags tensors of shape [B]
            # flag_tensors = flags.long().unbind(dim=1)
            # flag_embs = [emb(f) for emb, f in zip(self.flag_embs, flag_tensors)]
            flags_embs = self.flag_embs(flags)  # [B, num_flags * flag_emb_dim]
            x = torch.cat([x, flags_embs], dim=-1)

        # Encode through MLP
        out = self.mlp(x)
        return out  # [B, out_dim]