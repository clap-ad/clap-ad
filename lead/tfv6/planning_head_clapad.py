

from lead.tfv6.speed_score_utils import compute_overtake_adherence, compute_stop_sign_adherence, compute_traffic_light_adherence, compute_ttc
from lead.tfv6.planning_decoder import PlanningContextEncoder, PositionEmbeddingSine
from lead.tfv6.hydra_speed_head import HydraSpeedHead
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

class PlanningHeadClap(nn.Module):
    def __init__(
            self,
            use_local_tp=True,
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
            config=None,
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
        # For TFv6
        self.config = config
        embed_dims = self.config.transfuser_token_dim
        self.target_points_normalization_constants = torch.tensor(
            self.config.target_points_normalization_constants,
            dtype=self.config.torch_float_type,
        )
        self.planning_steps = self.config.num_route_points_prediction # since we predict 12 waypoints in future
        self.path_query = nn.Parameter(
            torch.zeros(
                1,
                self.planning_steps,
                self.config.transfuser_token_dim,
            )
        )
        self.dimension_adapter = nn.Conv2d(
            512, self.config.transfuser_token_dim, kernel_size=1
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                self.config.transfuser_token_dim,
                self.config.transfuser_num_bev_cross_attention_heads,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=self.config.transfuser_num_bev_cross_attention_layers,
            norm=nn.LayerNorm(self.config.transfuser_token_dim),
        )
        # Single linear head, matching PlanningDecoder.route_decoder.
        self.path_fc = nn.Linear(embed_dims, 2)
        num_modes = 4 # TTC - TF-light - Stop sign - Overtake
        assert num_modes > 1
        # use same speed head for time-to-collision (TTC) and traffic light (TL) branches
        # speed_fuser_dim = fuser_dim - 1 # In speed, navigation does not get fused
        speed_fuser_dim = fuser_dim # Needed for HipAD
        self.path_aug_cnt = path_aug_cnt
        speed_combined_attention = False
        common_context_encoder = False
        common_score_mlp = True
        self.speed_branch = HydraSpeedHead(
            embed_dims=embed_dims,
            num_speed_bins=num_speed_bins, 
            num_modes=num_modes, 
            planning_steps=self.planning_steps, 
            path_aug_cnt=path_aug_cnt, 
            combined_attention=speed_combined_attention, 
            common_context_encoder=common_context_encoder, 
            flags_shapes_dict=flags_shapes_dict, 
            fuser_dim=speed_fuser_dim, 
            common_score_mlp=common_score_mlp
        )
        # For compatibility with previous versions
        self.use_speed_wp = False
        self.speed_is_regression = False
        self.speed_is_hurdle = False
        self.speed_is_scores = True

        # Loss funcations
        speed_bins = self.config.speed_bins
        ttc_time_horizon = self.config.ttc_time_horizon # in seconds
        self.loss_speed = SpeedScoringLoss(
            speed_bins=speed_bins,
            crt_cfg_dict={
                "ttc_cfg":{
                    "time_horizon": ttc_time_horizon,
                    "coll_radius": 0.1,
                    "safe_distance": 0.5,
                    "use_speed_dependent_horizon": self.config.use_speed_dependent_ttc_horizon,
                    "time_horizon_slow": self.config.ttc_time_horizon_slow,
                },
                "traffic_rule_cfg":{
                    "speed_thresh":0.1,
                    "ss_stop_distance": 2.0
                },
                "overtake_cfg":{
                    "ot_stop_distance": 20.0
                }
            },
        )
        
        self.traj_or_action = 'path_speed'  # 'traj' or 'action' or 'path_speed' or 'path_speed_wp'
        self.use_tf_context_encoder = self.config.use_tf_context_encoder
        if self.use_tf_context_encoder:
            self.planning_context_encoder = PlanningContextEncoder(
                config=self.config,
                input_bev_channels=self.config.transfuser_token_dim,
                device=config.device,
            )
        else:
            self.target_points_normalization_constants = torch.tensor(
                self.config.target_points_normalization_constants,
                dtype=self.config.torch_float_type,
            )
            self.tp_encoder = nn.Linear(2, config.transfuser_token_dim)
            
            self.num_navi_info = 1 # Number of different embeddings representing future navigation info
            command_dim = self.config.discrete_command_dim

            if self.config.use_tp:
                self.num_navi_info += 1
                self.tp_encoder = nn.Linear(2, config.transfuser_token_dim)

            if self.config.use_previous_tp:
                self.num_navi_info += 1

            if self.config.use_next_tp:
                self.num_navi_info += 1
            
            self.cosine_pos_embeding = PositionEmbeddingSine(
                config, self.config.transfuser_token_dim // 2, normalize=True
            )
            self.navi_embed = nn.Embedding(command_dim, embed_dims)

        # Break query symmetry: with zero-initialized queries and no positional
        # encoding on the path query, all route slots are identical, collapsing
        # the predicted route to a straight line. Matches PlanningDecoder.
        nn.init.uniform_(self.path_query)

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
        bev_embed = kwargs.get("bev_embed") # [B, 512, 10, 12]
        planner_radar_features = kwargs.get("planner_radar_features") # [B, 20, 256]
        planner_radar_predictions = kwargs.get("planner_radar_predictions") # [B, 20, 4] x,y,vel,score
        device = bev_embed.device
        dtype=self.config.torch_float_type
        data = kwargs.get("data", {})
        self.target_points_normalization_constants = self.target_points_normalization_constants.to(device, dtype=dtype)
        
        if not self.use_tf_context_encoder:
            navi_embed = []
            if self.config.use_discrete_command:
                command = data["command"].to(
                    device
                )
                # command is one-hot encoded [B, 6], need to get index
                command = torch.argmax(command, dim=-1)
                navi_embed.append(
                    self.navi_embed.weight[command].unsqueeze(1)
                )
            # Encode target point
            if self.config.use_tp:
                target_point = data["target_point"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
                target_point = target_point / self.target_points_normalization_constants
                tp_token = self.tp_encoder(target_point).reshape(
                    -1, 1, self.config.transfuser_token_dim
                )  # (bs, 1, transfuser_token_dim)
                navi_embed.append(tp_token)

            if self.config.use_previous_tp:
                previous_tp = data["target_point_previous"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
                previous_tp = previous_tp / self.target_points_normalization_constants
                previous_tp_token = self.tp_encoder(previous_tp).reshape(
                    -1, 1, self.config.transfuser_token_dim
                )  # (bs, 1, transfuser_token_dim)
                navi_embed.append(previous_tp_token)

            if self.config.use_next_tp:
                next_tp = data["target_point_next"].to(
                    device, dtype=torch.float32, non_blocking=True
                )
                next_tp = next_tp / self.target_points_normalization_constants
                next_tp_token = self.tp_encoder(next_tp).reshape(
                    -1, 1, self.config.transfuser_token_dim
                )  # (bs, 1, transfuser_token_dim)
                navi_embed.append(next_tp_token)


            plan_query = self.dimension_adapter(bev_embed) # [B, transfuser_token_dim, H, W]
            plan_query = plan_query + self.cosine_pos_embeding(plan_query)  # (B, transfuser_token_dim, H, W)
            plan_query = torch.flatten(plan_query, start_dim=2)  # (B, transfuser_token_dim, H * W)
            plan_query = plan_query.permute(0, 2, 1)  # (B, H * W, transfuser_token_dim)

            navi_embed = torch.cat(navi_embed, dim=1)  # [B, (1+1+1+1), transfuser_token_dim]

            plan_nav_query = torch.cat([plan_query, navi_embed], dim=1) # [B, (1+1+1+1) + H * W, transfuser_token_dim]

            B = plan_nav_query.shape[0]  # Batch size
            path_query = self.transformer_decoder(self.path_query.repeat(B, 1, 1), plan_nav_query) # [B, T, transfuser_token_dim]
        else:
            self.kv = context_tokens = self.planning_context_encoder(
                bev_features=self.dimension_adapter(bev_embed),
                radar_logits=planner_radar_features,
                radar_predictions=planner_radar_predictions,
                data=data,
                log=kwargs.get("log", {})
            )
            bs = context_tokens.shape[0]
            path_query = self.transformer_decoder(self.path_query.repeat(bs, 1, 1), context_tokens)
            plan_query = context_tokens

        path_pred = self.path_fc(path_query) # [B, T, 2]
        path_pred = torch.cumsum(path_pred, dim=1)
        
        # Speed: hydra classification logits
        # speed: [B, 1] tensor
        # path_pred: [B, pred_horizon, 2] tensor

        # Hydra speed head
        hydra_speed_inputs = dict(
            path_pred=path_pred,
            plan_query=plan_query
        )
        speed_logits, path_pred_augmented = self.speed_branch(**hydra_speed_inputs)  # [B, NUM_PATHS, 3, S], [B, AUG_CNT, T, 2]


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
        
        return self.forward_original_tf(
            **kwargs
        )

    def compute_loss(self, data, predictions, loss, log):
        """
        Args:
            - data: dict: contains ground truth dict
            - predictions: Perception: contains pred_clapad attribute
            - loss: dict: holds all losses, where new loss should be appended to it
        """
        first_loss = next(iter(loss.values()))
        device = first_loss.device if hasattr(first_loss, "device") else None
        # predict spatial path
        route_label = data["route"].to(
            device, dtype=self.config.torch_float_type, non_blocking=True
        )
        route_mask = data.get("route_mask", None)
        if route_mask is None:
            route_mask = torch.ones_like(route_label[..., 0], dtype=torch.bool)
        else:
            route_mask = route_mask.to(device)
            
        pred_clapad = predictions.pred_clapad
        pred_route = pred_clapad["sdc_traj"]

        # Apply mask to route, route_mask: [B, T], pred_route: [B, T, 2], route_label: [B, T, 2]
        route_mask_expanded = route_mask.unsqueeze(-1).expand_as(route_label)  # [B, T, 2]

        # ADE: L1 loss over all valid points
        ade_pred = pred_route[route_mask_expanded]
        ade_label = route_label[route_mask_expanded]
        loss["loss_spatial_route"] = F.l1_loss(
            ade_pred, ade_label.float()
        )  # ADE

        # FDE: L1 loss at the last valid time step for each batch element
        # Find the last valid index for each batch
        last_indices = route_mask.long().sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(route_mask.shape[0], device=route_mask.device)
        fde_pred = pred_route[batch_indices, last_indices, :]  # [B, 2]
        fde_label = route_label[batch_indices, last_indices, :]  # [B, 2]
        loss["loss_spatial_route"] += F.l1_loss(
            fde_pred.float(),
            fde_label.float(),
        )  # FDE

        curr_speed = data["speed"]
        path_pred = pred_route
        scores_pred = pred_clapad["speed_logits"]
        scene_score_info = data["scene_score_info"]
        path_pred_augmented = pred_clapad.get("path_pred_augmented", None) # [B, K, T, 2]

        loss_speed = self.loss_speed(curr_speed, path_pred, scores_pred, scene_score_info, path_pred_augmented)
        loss.update(loss_speed)

        return

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
        
        return loss_dict
    
class SpeedScoringLoss(nn.Module):
    """
    Provides scores that take into consideration variety of metrics given scene context.
    These scores are relevant for speed prediction objectives.
    Model should learn the scoring function.
    """
    def __init__(self, speed_bins, crt_cfg_dict):
        super(SpeedScoringLoss, self).__init__()
        self.speed_bins = speed_bins # list of speed bin edges to be scored
        self.num_classes = len(speed_bins)
        self.crt_cfg_dict = crt_cfg_dict
        # TTC Config
        self.ttc_cfg = crt_cfg_dict.get('ttc_cfg', {})
        self.time_horizon = self.ttc_cfg.get('time_horizon', 0.5)
        self.coll_radius = self.ttc_cfg.get('coll_radius', 0.1) # Below which we consider collision
        self.safe_distance = self.ttc_cfg.get('safe_distance', 0.5) # Above which we consider safe
        # If enabled, use a per-speed-bin time horizon for the TTC ground truth: the
        # highest speed bin uses time_horizon, the lowest uses time_horizon_slow, and
        # intermediate speeds are linearly interpolated.
        self.use_speed_dependent_horizon = self.ttc_cfg.get('use_speed_dependent_horizon', False)
        self.time_horizon_slow = self.ttc_cfg.get('time_horizon_slow', 1.0)
        # CARLA values multiplied by 2 because CARLA considers half-size
        self.ego_wl = [0.9183566570281982 * 2, 2.44619083404541 * 2]
        # Traffic Rule Adherence Config
        self.traffic_rule_cfg = crt_cfg_dict.get('traffic_rule_cfg', {})
        self.tl_stop_distance = self.traffic_rule_cfg.get('tl_stop_distance', 6.0) # At this distance all tl speed scores (except 0) should be 0
        self.ss_stop_distance = self.traffic_rule_cfg.get('ss_stop_distance', 3.0) # At this distance all ss speed scores (except 0) should be 0
        self.traffic_rule_decel_min = self.traffic_rule_cfg.get('decel_min', 2.5)  # score=1 when required decel <= this
        self.traffic_rule_decel_max = self.traffic_rule_cfg.get('decel_max', 4.0)  # score=0 when required decel >= this
        # Overtake config
        self.ot_cfg = crt_cfg_dict.get('overtake_cfg', {})
        self.use_ot = True if len(self.ot_cfg) > 0 else False
        self.ot_stop_distance = self.ot_cfg.get('ot_stop_distance', 20.0) # At this distance all ot speed scores (except 0) should be 0
        self.ot_decel_min = self.ot_cfg.get('ot_decel_min', 2.5)  # score=1 when required decel <= this
        self.ot_decel_max = self.ot_cfg.get('ot_decel_max', 4.0)  # score=0 when required decel >= this
        
        # Scores: NOTE: Do not change order of scores (without changing HydraSpeedHead output order)
        self.score_cfg = {
            "TTC_Score": 1.0,
            "traffic_light": 0.2,
            "stop_sign": 0.2,
            "overtake": 0.2,
            # "comfort_scores": 0.5,
            # "feasibility_scores": 1.0,
            # "speed_limit": 0.3,
        }

    def forward(self, curr_speed, path_pred, scores_pred, scene_score_info, path_pred_augmented=None, debug=False):
        """
        Args:
            current_speed: Tensor of shape [B] - current speed in m/s
            path_pred: Tensor of shape [B, T, 2] - predicted future path. T is number of future steps
            scores_pred: Tensor of shape [B, K, H, S_CLS] - model's predicted score logits for each speed bin
                - B: Batch size, K: Number of augmented paths, H: Number of heads, S_CLS: Number of speed classes
            scene_score_info: Tensor of shape [B] - scene information for the current step used to compute ground truth scores
            path_pred_augmented: Tensor of shape [B, K, T, 2] - augmented predicted future path
            debug: If True, returns raw scores for debugging purposes next to loss
        Returns:
            loss: Scalar tensor
        """
        B = path_pred.shape[0]
        cuda_device = path_pred.device
        # path_pred = path_pred.cpu().detach()
        PATH_NUM = scores_pred.shape[1]

        # Compute TTC scores for all paths at once
        if path_pred_augmented is not None:
            # skip last path_pred_augmented
            path_pred_augmented = path_pred_augmented[:, :PATH_NUM, :, :]  # [B, K, T, 2]
            ttc_scores = compute_ttc(
                path_pred=path_pred_augmented,  # [B, K, T, 2]
                scene_score_info=scene_score_info,
                speed_bins=self.speed_bins,
                time_horizon=self.time_horizon,
                safe_distance=self.safe_distance,
                coll_radius=self.coll_radius,
                ego_wl=self.ego_wl,
                cuda_device=cuda_device,
                time_horizon_slow=self.time_horizon_slow if self.use_speed_dependent_horizon else None
            )  # [B, K, S_CLS]
        else:
            # If no augmentation, expand path_pred to [B, 1, T, 2] for consistent indexing
            path_pred_exp = path_pred.unsqueeze(1)
            ttc_scores = compute_ttc(
                path_pred=path_pred_exp,  # [B, 1, T, 2]
                scene_score_info=scene_score_info,
                speed_bins=self.speed_bins,
                time_horizon=self.time_horizon,
                safe_distance=self.safe_distance,
                coll_radius=self.coll_radius,
                ego_wl=self.ego_wl,
                cuda_device=cuda_device,
                time_horizon_slow=self.time_horizon_slow if self.use_speed_dependent_horizon else None
            )  # [B, 1, S_CLS]

        # Compute other scores once (not per path)
        traffic_light_scores = compute_traffic_light_adherence(
            speed_bins=self.speed_bins,
            scene_score_info=scene_score_info,
            stop_distance_from_trigger=self.tl_stop_distance,
            decel_min=self.traffic_rule_decel_min,
            decel_max=self.traffic_rule_decel_max,
        )  # [B, S_CLS]
        stop_sign_scores = compute_stop_sign_adherence(
            speed_bins=self.speed_bins,
            scene_score_info=scene_score_info,
            stop_distance_from_trigger=self.ss_stop_distance,
            decel_min=self.traffic_rule_decel_min,
            decel_max=self.traffic_rule_decel_max,
        )  # [B, S_CLS]
        if self.use_ot:
            overtake_scores = compute_overtake_adherence(
                speed_bins=self.speed_bins,
                scene_score_info=scene_score_info,
                stop_distance_from_trigger=self.ot_stop_distance,
                decel_min=self.ot_decel_min,
                decel_max=self.ot_decel_max,
            )  # [B, S_CLS]

        # Stack all scores for each path
        combined_scores = []
        for i in range(ttc_scores.shape[1]):  # K
            ttc_score_i = ttc_scores[:, i, :]  # [B, S_CLS]
            # Expand other scores to match [B, 1, S_CLS] if needed
            tl_score = traffic_light_scores
            ss_score = stop_sign_scores
            if tl_score.dim() == 2:
                tl_score = tl_score.unsqueeze(1)
            if ss_score.dim() == 2:
                ss_score = ss_score.unsqueeze(1)
            if self.use_ot:
                ot_score = overtake_scores
                if ot_score.dim() == 2:
                    ot_score = ot_score.unsqueeze(1)
                combined = torch.stack([ttc_score_i, tl_score.squeeze(1), ss_score.squeeze(1), ot_score.squeeze(1)], dim=1)  # [B, H, S_CLS]
            else:
                combined = torch.stack([ttc_score_i, tl_score.squeeze(1), ss_score.squeeze(1)], dim=1)  # [B, H, S_CLS]
            combined_scores.append(combined)


        # Stack over K (paths): [K, B, H, S_CLS]
        combined_scores = torch.stack(combined_scores, dim=1).to(scores_pred.device)  # [B, K, H, S_CLS]

        # For debugging:
        raw_scores = {}
        for idx, score in enumerate(self.score_cfg.keys()):
            raw_scores[score] = combined_scores[:,:, idx, :]  # Shape: [B, K, S_CLS]

        # log_pred = torch.log_softmax(scores_pred, dim=-1)   # [B, K, S]
        # pred = torch.sigmoid(scores_pred)
        losses = {}
        total_loss = 0.0
        B, K, H, S_CLS = combined_scores.shape
        for i, (key, base_weight) in enumerate(self.score_cfg.items()):
            gt = combined_scores[:, :, i, :]  # [B, K, S_CLS]
            pred = scores_pred[:, :, i, :]    # [B, K, S_CLS]

            is_safe_scenario = torch.all(gt >= 0.99, dim=-1)  # [B, K]
            # Dynamic weight per path
            dynamic_sample_weight = torch.where(
                is_safe_scenario,
                torch.ones_like(is_safe_scenario, dtype=gt.dtype),
                torch.ones_like(is_safe_scenario, dtype=gt.dtype) * 10
            )  # [B, K]

            # BCE loss per sample and path, summed over S_CLS
            kl = F.binary_cross_entropy_with_logits(pred, gt, reduction="none").sum(dim=-1)  # [B, K]
            kl = kl * dynamic_sample_weight  # [B, K]
            loss_i = kl.mean()  # scalar: mean over batch and path
            weighted_loss = base_weight * loss_i
            losses[key] = weighted_loss # These are added to the dict to be tracked but only loss_speed_scores will be backpropagated not these separate losses
            total_loss += weighted_loss

        losses["loss_speed_scores"] = total_loss # "loss" should be in the name if intended to be backpropagated
        if debug:
            losses["raw_scores"] = raw_scores

        return losses