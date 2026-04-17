

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

class PlanningHeadClap(nn.Module):
    def __init__(
            self,
            bev_h=200,
            bev_w=200,
            embed_dims=256,
            planning_steps=6,
            command_dim=3,
            loss_planning=None,
            loss_collision=None,
            loss_speed=None,
            loss_steering=None,
            loss_accel=None,
            planning_eval=False,
            use_col_optim=False,
            col_optim_args=dict(
                occ_filter_range=5.0,
                sigma=1.0, 
                alpha_collision=5.0,
            ),
            with_adapter=False,
            predict_control=False,
            control_classification=False,
            use_local_tp=True,
            use_bev=False,
            num_speed_bins=7,
            num_modes=2,
            path_aug_cnt=5,
            speed_combined_attention=True,
            common_context_encoder=False,
            flags_shapes_dict=None,
            fuser_dim=3, 
            # 3 for UniAD (1x detection, 1x motion, 1x navigation)
            # 4 for VAD (1x detection, 2x motion, 1x navigation)
            # 2 for HipAD (1x detection & motion, 1x navigation)
            common_score_mlp=True,
            use_gru=True,
            old_config=True, # old is until VAD-v4
            **kwargs
        ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadClap, self).__init__()
        self.use_can_bus = True
        self.speed_combined_attention = speed_combined_attention

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = nn.Embedding(command_dim, embed_dims)
        
        # Multi-head outputs: path (GRU), speed (hydra), steering (Beta), acceleration (Beta)
        self.predict_control = predict_control
        self.control_classification = control_classification
        
        self.use_bev = use_bev
        
        self.old_config = old_config
        # + 1 for the target speed token
        self.use_local_tp = use_local_tp
        self.use_gru = use_gru

        mlp_input_size = embed_dims
        if self.use_local_tp:
            self.target_point_fc = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
            )
            mlp_input_size += 16 # for target point
        self.path_fc = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_input_size * 2),
            nn.ReLU(),
            nn.Linear(mlp_input_size * 2, mlp_input_size),
            nn.ReLU(),
            nn.Linear(mlp_input_size, planning_steps * 2),
        )
        
        assert num_modes > 1
        # use same speed head for time-to-collision (TTC) and traffic light (TL) branches
        # speed_fuser_dim = fuser_dim - 1 # In speed, navigation does not get fused
        speed_fuser_dim = fuser_dim # Needed for HipAD
        self.path_aug_cnt = path_aug_cnt
        # self.speed_branch = HydraSpeedHead(embed_dims=embed_dims, num_speed_bins=num_speed_bins, num_modes=3, planning_steps=planning_steps, path_aug_cnt=path_aug_cnt, combined_attention=speed_combined_attention, common_context_encoder=common_context_encoder, flags_shapes_dict=flags_shapes_dict, fuser_dim=speed_fuser_dim, common_score_mlp=common_score_mlp)
        # For compatibility with previous versions
        self.use_speed_wp = False
        self.speed_is_regression = False
        self.speed_is_hurdle = False
        self.speed_is_scores = True
        
        # Placeholder for path_branch (backward compatibility)
        self.path_branch = None
        
        # self.loss_planning = build_loss(loss_planning)
        # self.loss_speed = build_loss(loss_speed) if loss_speed is not None else None
        # self.loss_steering = build_loss(loss_steering) if loss_steering is not None else None
        # self.loss_accel = build_loss(loss_accel) if loss_accel is not None else None
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval
        

        self.mlp_fuser = nn.Sequential(
            nn.Linear(embed_dims * fuser_dim, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.use_bev:
            self.pos_embed = nn.Embedding(1, embed_dims)

        # one learnable query per output
        # we no longer use speed queries
        splitter_dim = self.planning_steps  # planning steps
        if self.predict_control:
            splitter_dim += 2  #  + steer accel
                
        # For RL fine-tuning step:
        self.traj_or_action = 'path_speed'  # 'traj' or 'action' or 'path_speed' or 'path_speed_wp'


    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      **kwargs
                      ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (torch.Tensor): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
            command (torch.Tensor, optional): The driving command issued to the self-driving car.
            gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        sdc_traj_scores = outs_motion['sdc_traj_scores']
        other_traj_queries = outs_motion['traj_query']
        other_track_queries = outs_motion['track_query']
        other_traj_scores = outs_motion['traj_scores']
        bev_pos = outs_motion['bev_pos']

        occ_mask = None
        
        outs_planning = self(
            bev_embed=bev_embed, 
            occ_mask=occ_mask, 
            bev_pos=bev_pos, 
            sdc_traj_query=sdc_traj_query, 
            sdc_track_query=sdc_track_query,
            sdc_traj_scores=sdc_traj_scores,
            command=command,
            other_traj_queries=other_traj_queries,
            other_track_queries=other_track_queries,
            other_traj_scores=other_traj_scores,
            **kwargs
        )
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs, **kwargs)
        # Pop 'raw_scores' from losses and put it in outs_planning dict
        if 'raw_scores' in losses:
            outs_planning['raw_scores'] = losses.pop('raw_scores')
        ret_dict = dict(losses=losses, outs_motion=outs_planning) # Why outs_motion?
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None, **kwargs):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        sdc_traj_scores = outs_motion['sdc_traj_scores']
        other_traj_queries = outs_motion['traj_query']
        other_track_queries = outs_motion['track_query']
        other_traj_scores = outs_motion['traj_scores']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']
        
        outs_planning = self(
            bev_embed=bev_embed,
            occ_mask=occ_mask,
            bev_pos=bev_pos,
            sdc_traj_query=sdc_traj_query,
            sdc_track_query=sdc_track_query,
            sdc_traj_scores=sdc_traj_scores,
            command=command,
            other_traj_queries=other_traj_queries,
            other_track_queries=other_track_queries,
            other_traj_scores=other_traj_scores,
            **kwargs
        )
        return outs_planning

    def forward_original_tf(self, 
                sdc_traj_query, 
                sdc_track_query,
                sdc_traj_scores,
                command,
                other_traj_queries,
                other_track_queries,
                other_traj_scores,
                tf_track_queries,
                st_track_queries,
                can_bus=None,
                crt_flags_dict=None,
                local_command_xy=None,
                **kwargs):
        """
        Re-implementation of forward_original for HiPAD, which has slightly different input structure.
        Args:
            - sdc_traj_query [B, 1, 256]
            - sdc_track_query [B, 1, 256] (In HipAD, sdc_track and sdc_traj are identical)
            - sdc_traj_scores: For now it is None
            - command: [B]
            - other_traj_queries: [B, 900, 6, 256] (900 proposals, 6 motion mode, 256 dim)
            - other_track_queries: [B, 900, 256]
            - other_traj_scores: [B, 900, 6]
            - tf_track_queries: list of size B, each element is [num_tf_tracks, 256]
            - st_track_queries: list of size B, each element is [num_st_tracks, 256]
            - local_command_xy: [B, 2]
        """
        if sdc_track_query.ndim == 1:
            sdc_track_query = sdc_track_query.unsqueeze(0).detach()  # [256] -> [1, 256]
        elif sdc_track_query.ndim == 3:
            sdc_track_query = sdc_track_query.squeeze(1).detach()
        navi_embed = self.navi_embed.weight[command]
        plan_query = torch.cat([sdc_track_query, navi_embed], dim=-1) # [B, 512]

        plan_query = self.mlp_fuser(plan_query)   # expand, then fuse  # [B, 512] -> [B, 256]
        
        B = plan_query.shape[0]  # Batch size
        path_input = plan_query  # [B, 256]

        # Path: GRU autoregressively predicts deltas
        if self.use_local_tp:
            if isinstance(local_command_xy, list):
                local_command_xy = local_command_xy[0]
            local_command_xy = local_command_xy.to(path_input.dtype)

        if self.use_gru:
            if path_input.dim() == 2:
                path_input = path_input.unsqueeze(0)  # [1, 6, 256]
            if local_command_xy is not None and local_command_xy.dim() == 2:
                local_command_xy = local_command_xy.unsqueeze(0)  # [1, 6, 2]
            path_pred = self.path_gru(path_input, target_point=local_command_xy)  # [B, T, 2]
        else:
            if self.use_local_tp:
                local_feat = self.target_point_fc(local_command_xy)  # [B, 16]
                path_input = torch.cat([path_input, local_feat], dim=-1)
            path_pred = self.path_fc(path_input).view(B, self.planning_steps, 2)  # [B, T, 2]
            path_pred = torch.cumsum(path_pred, dim=1)
        # Apply bivariate Gaussian activation for PPO
        # path_pred[0] = bivariate_gaussian_activation(path_pred[0])
        
        # Speed: hydra classification logits
        # speed: [B, 1] tensor
        # path_pred: [B, pred_horizon, 2] tensor
        # flags: list of [B, X] LongTensors, one per flag (optional)
        
        if self.speed_combined_attention:
            speed = can_bus[7].unsqueeze(0) # TODO Make input batch
        else:
            speed = None

        # Hydra speed head
        hydra_speed_inputs = dict(
            track_query=other_track_queries, 
            traj_query=other_traj_queries, 
            traj_scores=other_traj_scores, 
            tf_track_query=tf_track_queries, 
            st_track_query=st_track_queries, 
            speed=speed, 
            path_pred=path_pred,
            flags=crt_flags_dict
        )
        speed_logits, path_pred_augmented = self.speed_branch(**hydra_speed_inputs)  # [B, NUM_PATHS, 2, S], [B, T, 2]

        # speed_logits = torch.cat([speed_logits_ttc, speed_logits_tl.expand(-1, speed_logits_ttc.shape[1], -1, -1)], dim=2)  # [B, NUM_PATHS, 2*S, 1]

        outs_planning = dict(
            sdc_traj=path_pred,
            sdc_traj_all=path_pred,
            speed_logits=speed_logits,
            # speed_logits_ttc=speed_logits_ttc,
            # speed_logits_tl=speed_logits_tl,
            traj_or_action=self.traj_or_action,
            path_pred_augmented=path_pred_augmented if self.path_aug_cnt is not None else None,
        )

        return outs_planning

    def forward(self, **kwargs):
        
        occ_mask = kwargs.get("occ_mask", None)
        bev_pos = kwargs.get("bev_pos", None)
        sdc_traj_query = kwargs["sdc_traj_query"]
        sdc_track_query = kwargs["sdc_track_query"]
        sdc_traj_scores = kwargs.get("sdc_traj_scores", None)
        command = kwargs["command"]
        other_traj_queries = kwargs["other_traj_queries"]
        other_track_queries = kwargs["other_track_queries"]
        other_traj_scores = kwargs["other_traj_scores"]
        tf_track_queries = kwargs.get("tf_track_queries", None)
        st_track_queries = kwargs.get("st_track_queries", None)
        crt_flags_dict = kwargs.get("crt_flags_dict", None)
        can_bus = kwargs["can_bus"]
        local_command_xy = kwargs.get("local_command_xy", None)
        model_name = kwargs.get("model_name", "unknown_model")
        return self.forward_original_tf(
            sdc_traj_query=sdc_traj_query,
            sdc_track_query=sdc_track_query,
            sdc_traj_scores=sdc_traj_scores,
            command=command,
            other_traj_queries=other_traj_queries,
            other_track_queries=other_track_queries,
            other_traj_scores=other_traj_scores,
            tf_track_queries=tf_track_queries,
            st_track_queries=st_track_queries,
            can_bus=can_bus,
            crt_flags_dict=crt_flags_dict,
            local_command_xy=local_command_xy,
            **kwargs
        )

    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, **kwargs):
        """
        Compute all losses: path, collision, speed, steering, acceleration.
        This loss was updated for to account for higher batch sizes
        Args:
            sdc_planning: [B, 1, T, 3] with (x, y, heading)
            sdc_planning_mask: [B, 1, T, 3] mask
            outs_planning: dict with predictions
        Note:
            In case of usage of speed_wp, kwargs contains:
            {
                sdc_planning: [1, 1, T, 3]
                sdc_planning_mask: [1, 1, T, 3] mask
            }
        """
        sdc_traj_all = outs_planning['sdc_traj_all'] # [B, T, 2]
        loss_dict = dict()
        
        # 1. Collision losses
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](
                sdc_traj_all, 
                sdc_planning[0, :, :self.planning_steps, :3], 
                torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), 
                future_gt_bbox[0][1:self.planning_steps+1]
            )
            loss_dict[f'loss_collision_{i}'] = loss_collision
        
        # 2. Path ADE loss
        loss_ade = self.loss_planning(
            sdc_traj_all,
            sdc_planning[:, :, :self.planning_steps, :2].squeeze(), 
            torch.any(sdc_planning_mask[:, :, :self.planning_steps].squeeze(), dim=-1)
        )
        loss_dict.update(dict(loss_ade=loss_ade))
        
        # 3. Speed loss (classification or regression)
        if self.loss_speed is not None:
            # current_speed: Tensor of shape [B] - current speed in m/s
            # path_pred: Tensor of shape [B, M, 2] - predicted future path. M is number of future steps
            # scores_pred: Tensor of shape [B, , num_classes] - model's predicted score logits for each speed bin
            # scene_score_info: Tensor of shape [B] - scene information for the current step used to compute ground truth scores

            # skip for now
            # return loss_dict
            if "curr_speed" in kwargs:
                curr_speed = kwargs["curr_speed"]
            elif "can_bus" in kwargs:
                curr_speed = kwargs["can_bus"][7]
            else:
                curr_speed = None
                print("Warning: current_speed is not available, for the speed loss calculation")
            path_pred = outs_planning["sdc_traj"]
            scores_pred = outs_planning["speed_logits"]
            scene_score_info = kwargs["scene_score_info"]
            path_pred_augmented = outs_planning.get("path_pred_augmented", None) # [B, K, T, 2]

            # loss_speed = self.loss_speed(curr_speed, path_pred, scores_pred, scene_score_info):
            loss_speed = self.loss_speed(curr_speed, path_pred, scores_pred, scene_score_info, path_pred_augmented)
            loss_dict.update(loss_speed)
        
        if self.predict_control:
            if self.control_classification:
                # 4. Steering loss
                if self.loss_steering is not None and 'steering_logits' in outs_planning:
                    steering_logits = outs_planning['steering_logits']  # [B, 8]
                    loss_steering = self.loss_steering(steering_logits, gt_fut_steer)
                    loss_dict['loss_steering'] = loss_steering

                # 5. Acceleration loss (throttle and brake mutually exclusive)
                if self.loss_accel is not None and 'accel_logits' in outs_planning and gt_fut_throttle is not None and gt_fut_brake is not None:
                    gt_accel = gt_fut_throttle - gt_fut_brake
                    accel_logits = outs_planning['accel_logits']  # [B, 6]
                    loss_accel = self.loss_accel(accel_logits, gt_accel)
                    loss_dict['loss_accel'] = loss_accel
            else:  # beta dist
                # 4. Steering loss
                if self.loss_steering is not None and 'steering_alpha' in outs_planning:
                    steering_alpha = outs_planning['steering_alpha']  # [B]
                    steering_beta = outs_planning['steering_beta']  # [B]
                    loss_steering = self.loss_steering(steering_alpha, steering_beta, gt_fut_steer)
                    loss_dict['loss_steering'] = loss_steering
                
                # 5. Acceleration loss (throttle and brake mutually exclusive)
                if self.loss_accel is not None and 'accel_alpha' in outs_planning and gt_fut_throttle is not None and gt_fut_brake is not None:
                    gt_accel = gt_fut_throttle - gt_fut_brake
                    accel_alpha = outs_planning['accel_alpha']  # [B]
                    accel_beta = outs_planning['accel_beta']  # [B]
                    loss_accel = self.loss_accel(accel_alpha, accel_beta, gt_accel)
                    loss_dict['loss_accel'] = loss_accel
        
        return loss_dict