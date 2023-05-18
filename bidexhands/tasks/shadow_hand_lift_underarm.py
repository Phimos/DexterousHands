# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from PIL import Image as Im
from typing import Optional, Sequence

import numpy as np
import os
import random
import torch

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class ShadowHandLiftUnderarm(BaseTask):
    """
    This class corresponds to the LiftUnderarm task. This environment requires grasping the pot handle 
    with two hands and lifting the pot to the designated position. This environment is designed to 
    simulate the scene of lift in daily life and is a practical skill.

    Args:
        cfg (dict): The configuration file of the environment, which is the parameter defined in the
            dexteroushandenvs/cfg folder

        sim_params (isaacgym._bindings.linux-x86_64.gym_37.SimParams): Isaacgym simulation parameters 
            which contains the parameter settings of the isaacgym physics engine. Also defined in the 
            dexteroushandenvs/cfg folder

        physics_engine (isaacgym._bindings.linux-x86_64.gym_37.SimType): Isaacgym simulation backend
            type, which only contains two members: PhysX and Flex. Our environment use the PhysX backend

        device_type (str): Specify the computing device for isaacgym simulation calculation, there are 
            two options: 'cuda' and 'cpu'. The default is 'cuda'

        device_id (int): Specifies the number of the computing device used when simulating. It is only 
            useful when device_type is cuda. For example, when device_id is 1, the device used 
            is 'cuda:1'

        headless (bool): Specifies whether to visualize during training

        agent_index (list): Specifies how to divide the agents of the hands, useful only when using a 
            multi-agent algorithm. It contains two lists, representing the left hand and the right hand. 
            Each list has six numbers from 0 to 5, representing the palm, middle finger, ring finger, 
            tail finger, index finger, and thumb. Each part can be combined arbitrarily, and if placed 
            in the same list, it means that it is divided into the same agent. The default setting is
            [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], which means that the two whole hands are 
            regarded as one agent respectively.

        is_multi_agent (bool): Specifies whether it is a multi-agent environment
    """
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        # assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            # "pot": "mjcf/pot.xml",
            "pot": "mjcf/pot/mobility.urdf"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)
        
        self._parse_observation_and_action_space()

        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = 'z'

        self.fingertips = ["ffdistal", "lfdistal", "mfdistal", "rfdistal", "thdistal"]
        self.hand_center = "palm"

        self.num_fingertips = len(self.fingertips) * 2

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        if self.obs_type in ["point_cloud"]:
            from PIL import Image as Im
            from bidexhands.utils import o3dviewer
            # from pointnet2_ops import pointnet2_utils

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Retrieve generic tensor descriptors for the simulation
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)            # [num_envs * num_actors, 13]
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)                    # [num_envs * num_dofs, 2]
        _dof_forces = self.gym.acquire_dof_force_tensor(self.sim)                    # [num_envs * num_dofs]
        _rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)      # [num_envs * num_rigid_bodies, 13]
        _force_sensor_states = self.gym.acquire_force_sensor_tensor(self.sim)        # [num_envs * num_force_sensors, 6]
        _jacobian_matrix_left = self.gym.acquire_jacobian_tensor(self.sim, "left")   # [num_envs, num_dofs, 6, num_rigid_bodies]
        _jacobian_matrix_right = self.gym.acquire_jacobian_tensor(self.sim, "right") # [num_envs, num_dofs, 6, num_rigid_bodies]
        
        print("root_states.shape: ", _root_states.shape)
        print("dof_states.shape: ", _dof_states.shape)
        print("rigid_body_states.shape: ", _rigid_body_states.shape)
        print("force_sensor_states.shape: ", _force_sensor_states.shape)
        print("dof_forces.shape: ", _dof_forces.shape)
        print("jacobian_matrix_left.shape: ", _jacobian_matrix_left.shape)
        print("jacobian_matrix_right.shape: ", _jacobian_matrix_right.shape)
        
        # Calculate number of envs, actors, dofs, rigid bodies, and force sensors
        self.num_actors = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.num_force_sensors = self.gym.get_sim_force_sensor_count(self.sim) // self.num_envs
        self.num_rigid_bodies = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs
        
        print("num_actors: ", self.num_actors)
        print("num_dofs: ", self.num_dofs)
        print("num_force_sensors: ", self.num_force_sensors)
        print("num_rigid_bodies: ", self.num_rigid_bodies)
        
        # Wrap tensors with gymtorch
        self.root_states = gymtorch.wrap_tensor(_root_states)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_forces = gymtorch.wrap_tensor(_dof_forces)
        self.rigid_body_states = gymtorch.wrap_tensor(_rigid_body_states)
        self.force_sensor_states = gymtorch.wrap_tensor(_force_sensor_states)
        # TODO: use the jacobian matrix
        
        # Refresh the tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.cached_root_states = self.root_states.clone()
        
        # Create some wrapper tensors for different slices
        self.root_positions = self.root_states[:, :3]
        self.root_orientations = self.root_states[:, 3:7]
        self.root_linear_velocities = self.root_states[:, 7:10]
        self.root_angular_velocities = self.root_states[:, 10:13]
        
        root_states = self.root_states.view(self.num_envs, self.num_actors, 13)
        self.shadow_hand_left_root_states = root_states[:, self.shadow_hand_left_index, :]
        self.shadow_hand_left_root_positions = self.shadow_hand_left_root_states[:, :3]
        self.shadow_hand_left_root_orientations = self.shadow_hand_left_root_states[:, 3:7]
        self.shadow_hand_left_root_linear_velocities = self.shadow_hand_left_root_states[:, 7:10]
        self.shadow_hand_left_root_angular_velocities = self.shadow_hand_left_root_states[:, 10:13]
        
        self.shadow_hand_right_root_states = root_states[:, self.shadow_hand_right_index, :]
        self.shadow_hand_right_root_positions = self.shadow_hand_right_root_states[:, :3]
        self.shadow_hand_right_root_orientations = self.shadow_hand_right_root_states[:, 3:7]
        self.shadow_hand_right_root_linear_velocities = self.shadow_hand_right_root_states[:, 7:10]
        self.shadow_hand_right_root_angular_velocities = self.shadow_hand_right_root_states[:, 10:13]
        
        self.object_root_states = root_states[:, self.object_index, :]
        self.object_root_positions = self.object_root_states[:, :3]
        self.object_root_orientations = self.object_root_states[:, 3:7]
        self.object_root_linear_velocities = self.object_root_states[:, 7:10]
        self.object_root_angular_velocities = self.object_root_states[:, 10:13]
        
        self.goal_root_states = root_states[:, self.goal_index, :]
        self.goal_root_positions = self.goal_root_states[:, :3]
        self.goal_root_orientations = self.goal_root_states[:, 3:7]
        
        dof_states = self.dof_states.view(self.num_envs, self.num_dofs, 2)
        self.shadow_hand_left_dof_positions = dof_states[:, self.shadow_hand_left_dof_start:self.shadow_hand_left_dof_end, 0]
        self.shadow_hand_left_dof_velocities = dof_states[:, self.shadow_hand_left_dof_start:self.shadow_hand_left_dof_end, 1]
        self.shadow_hand_right_dof_positions = dof_states[:, self.shadow_hand_right_dof_start:self.shadow_hand_right_dof_end, 0]
        self.shadow_hand_right_dof_velocities = dof_states[:, self.shadow_hand_right_dof_start:self.shadow_hand_right_dof_end, 1]
        
        dof_forces = self.dof_forces.view(self.num_envs, self.num_dofs)
        self.shadow_hand_left_dof_forces = dof_forces[:, self.shadow_hand_left_dof_start:self.shadow_hand_left_dof_end]
        self.shadow_hand_right_dof_forces = dof_forces[:, self.shadow_hand_right_dof_start:self.shadow_hand_right_dof_end]
        
        rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_rigid_bodies, 13)
        
        force_sensor_states = self.force_sensor_states.view(self.num_envs, 2, self.num_force_sensors // 2, 6)
        self.fingertip_left_force_sensor_states = force_sensor_states[:, 0, :, :]
        self.fingertip_right_force_sensor_states = force_sensor_states[:, 1, :, :]
        
        # Allocate tensors and variables
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_rigid_bodies, 3), device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

    def _parse_observation_and_action_space(self):
        num_shadow_hand_dofs = 30
        num_shadow_hand_actuated_dofs = 26
        num_fingertips = 5

        observation_mapping = {
            "shadow_hand_left_dof_position": (num_shadow_hand_dofs, "shadow_hand_left_dof_positions", ("dof", "position")),
            "shadow_hand_left_dof_velocity": (num_shadow_hand_dofs, "shadow_hand_left_dof_velocities", ("dof", "velocity")),
            "shadow_hand_left_dof_force": (num_shadow_hand_dofs, "shadow_hand_left_dof_forces", ("dof", "force")),
            "shadow_hand_left_fingertip_position": (num_fingertips * 3, "fingertip_left_positions", ("fingertip", "position")),
            "shadow_hand_left_fingertip_orientation": (num_fingertips * 4, "fingertip_left_orientations", ("fingertip", "orientation")),
            "shadow_hand_left_fingertip_linear_velocity": (num_fingertips * 3, "fingertip_left_linear_velocities", ("fingertip", "linear_velocity")),
            "shadow_hand_left_fingertip_angular_velocity": (num_fingertips * 3, "fingertip_left_angular_velocities", ("fingertip", "angular_velocity")),
            "shadow_hand_left_fingertip_force": (num_fingertips * 6, "fingertip_left_force_sensor_states", ("fingertip", "force")),
            "shadow_hand_left_position": (3, "shadow_hand_left_positions", ("position",)),
            "shadow_hand_left_orientation": (4, "shadow_hand_left_orientations", ("orientation",)),
            "shadow_hand_left_action": (num_shadow_hand_actuated_dofs, "shadow_hand_left_actions", ("action",)),
            "shadow_hand_right_dof_position": (num_shadow_hand_dofs, "shadow_hand_right_dof_positions", ("dof", "position")),
            "shadow_hand_right_dof_velocity": (num_shadow_hand_dofs, "shadow_hand_right_dof_velocities", ("dof", "velocity")),
            "shadow_hand_right_dof_force": (num_shadow_hand_dofs, "shadow_hand_right_dof_forces", ("dof", "force")),
            "shadow_hand_right_fingertip_position": (num_fingertips * 3, "fingertip_right_positions", ("fingertip", "position")),
            "shadow_hand_right_fingertip_orientation": (num_fingertips * 4, "fingertip_right_orientations", ("fingertip", "orientation")),
            "shadow_hand_right_fingertip_linear_velocity": (num_fingertips * 3, "fingertip_right_linear_velocities", ("fingertip", "linear_velocity")),
            "shadow_hand_right_fingertip_angular_velocity": (num_fingertips * 3, "fingertip_right_angular_velocities", ("fingertip", "angular_velocity")),
            "shadow_hand_right_fingertip_force": (num_fingertips * 6, "fingertip_right_force_sensor_states", ("fingertip", "force")),
            "shadow_hand_right_position": (3, "shadow_hand_right_positions", ("position",)),
            "shadow_hand_right_orientation": (4, "shadow_hand_right_orientations", ("orientation",)),
            "shadow_hand_right_action": (num_shadow_hand_actuated_dofs, "shadow_hand_right_actions", ("action",)),
            "object_position": (3, "object_root_positions", ("position",)),
            "object_orientation": (4, "object_root_orientations", ("orientation",)),
            "object_linear_velocity": (3, "object_root_linear_velocities", ("linear_velocity",)),
            "object_angular_velocity": (3, "object_root_angular_velocities", ("angular_velocity",)),
            "goal_position": (3, "goal_root_positions", ("position",)),
            "goal_orientation": (4, "goal_root_orientations", ("orientation",)),
            "object_to_goal_orientation": (4, "object_to_goal_orientation", ("orientation",)),
            "object_left_handle_position": (3, "pot_left_handle_positions", ("position",)),
            "object_right_handle_position": (3, "pot_right_handle_positions", ("position",)),
        }

        observation_space = self.cfg["env"]["observationSpace"]
        num_observations = sum([observation_mapping[observation][0] for observation in observation_space])
        
        self.observation_space = observation_space
        self.observation_mapping = observation_mapping
        
        self.enable_pointcloud_observation = any(["pointcloud" in tags for _, _, tags in observation_mapping.values()])
                
        self._display_observation_space(observation_space, observation_mapping)
        
        # Set number of observations & actions        
        self.cfg["env"]["numObservations"] = num_observations
        self.cfg["env"]["numStates"] = num_observations
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = num_shadow_hand_actuated_dofs
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = num_shadow_hand_actuated_dofs * 2

    def _pack_observations(self):
        observations = []
        for name in self.observation_space:
            num_dim, attr_name, tags = self.observation_mapping[name]
            observation = getattr(self, attr_name)
            
            if "dof" in tags and "position" in tags:
                observation = unscale(observation, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            if "dof" in tags and "force" in tags:
                observation = observation * self.force_torque_obs_scale
            if "dof" in tags and "velocity" in tags:
                observation = observation * self.vel_obs_scale
            if "angular_velocity" in tags:
                observation = observation * self.vel_obs_scale
            
            observations.append(observation.reshape(self.num_envs, num_dim))
        
        return torch.cat(observations, dim=-1)

    def _display_observation_space(self, observation_space, observation_mapping):
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", title="Observation Space")
        table.add_column("Observation")
        table.add_column("Size")
        table.add_column("Start")
        table.add_column("End")
        table.add_column("Tags")
        
        num_observations = 0
        for observation in observation_space:
            num_dim, _, tags = observation_mapping[observation]
            table.add_row(
                observation, 
                str(num_dim), 
                str(num_observations), 
                str(num_observations + num_dim),
                ", ".join(tags) if tags else ""
            )
            num_observations += num_dim
            
        console.print(table)

    def create_sim(self):
        """
        Allocates which device will simulate and which device will render the scene. Defines the simulation type to be used
        """

        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _display_dof_properties(self, asset: gymapi.Asset, properties: np.ndarray):
        """Displays the DoF properties of the asset in a rich table

        Args:
            asset (gymapi.Asset): IssacGym asset
            properties (np.ndarray): DoF properties, stored in a structured array
        """
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", title="DoF Properties")
        table.add_column("Name")

        columns = properties.dtype.names
        
        for column in columns:
            table.add_column(column.capitalize())
            
        for i in range(properties.shape[0]):
            name = self.gym.get_asset_dof_name(asset, i)
            item = [name] + [str(properties[column][i]) for column in columns]
            table.add_row(*item)

        console.print(table)

    def _random_colorize(self, env: gymapi.Env, shadow_hand_actor: int, shadow_hand_side: str, part_level: bool = False):
        """Randomly colorize the shadow hand.

        Args:
            env (gymapi.Env): IsaacGym environment handle.
            shadow_hand_actor (int): Shadow hand actor handle.
            shadow_hand_side (str): Shadow hand side ("left" or "right").
            part_level (bool, optional): Whether to colorize at part level. Defaults to False.
        """
        assert shadow_hand_side in ["left", "right"]
        prefix = "lh_" if shadow_hand_side == "left" else "rh_"
        
        colorization_groups = [
            ["base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"],
            ["wrist", "palm"],
            ["ffknuckle", "ffproximal", "ffmiddle", "ffdistal"],
            ["lfmetacarpal", "lfknuckle", "lfproximal", "lfmiddle", "lfdistal"],
            ["mfknuckle", "mfproximal", "mfmiddle", "mfdistal"],
            ["rfknuckle", "rfproximal", "rfmiddle", "rfdistal"],
            ["thbase", "thproximal", "thhub", "thmiddle", "thdistal"]
        ]
        
        for i, group in enumerate(colorization_groups):
            if i == 0:
                continue
            for j in range(len(group)):
                group[j] = prefix + group[j]
        
        color = gymapi.Vec3(random.random(), random.random(), random.random())
        for i, group in enumerate(colorization_groups):
            if part_level:
                color = gymapi.Vec3(random.random(), random.random(), random.random())
            for link in group:
                self.gym.set_rigid_body_color(
                    env, 
                    shadow_hand_actor, 
                    self.gym.find_actor_rigid_body_index(env, shadow_hand_actor, link, gymapi.DOMAIN_ACTOR),
                    gymapi.MESH_VISUAL, 
                    color
                )

    def _transform_to_tensor(self, transform: gymapi.Transform, with_quaternion: bool = True, with_velocity: bool = True, device: Optional[torch.device] = None) -> torch.Tensor:
        """Transform `gymapi.Transform` to `torch.Tensor`
        """
        device = self.device if device is None else device
        
        position = [transform.p.x, transform.p.y, transform.p.z]
        quaternion = [transform.r.x, transform.r.y, transform.r.z, transform.r.w]
        velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        if with_quaternion and with_velocity:
            return torch.tensor(position + quaternion + velocity, dtype=torch.float32, device=device)
        elif with_quaternion and not with_velocity:
            return torch.tensor(position + quaternion, dtype=torch.float32, device=device)
        elif not with_quaternion and not with_velocity:
            return torch.tensor(position, dtype=torch.float32, device=device)
        else:
            raise NotImplementedError

    """
    Functions - create assets
    """
    
    def _create_ground_plane(self):
        """Creates the ground plane for the simulation
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_robots(self):
        asset_root = "../assets"
        
        shadow_hand_left_asset_file = "urdf/shadow_robot/ur10e_shadow_hand_left.urdf"
        shadow_hand_right_asset_file = "urdf/shadow_robot/ur10e_shadow_hand_right.urdf"
        
        # Load assets
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.1
        asset_options.linear_damping = 0.1

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        shadow_hand_left_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_left_asset_file, asset_options)
        shadow_hand_right_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_right_asset_file, asset_options)
        
        self.shadow_hand_left_asset = shadow_hand_left_asset
        self.shadow_hand_right_asset = shadow_hand_right_asset
        
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_right_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_right_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_right_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_right_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_right_asset)
        
        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)
        
        # Setup tendon properties
        # TODO: num_shadow_hand_tendons = 0, so this block of code does nothing
        limit_stiffness = 30.0
        damping = 0.1
        relevant_tendon_dofs = ["FFJ1", "MFJ1", "RFJ1", "LFJ1"]
        tendon_props_left = self.gym.get_asset_tendon_properties(shadow_hand_left_asset)
        tendon_props_right = self.gym.get_asset_tendon_properties(shadow_hand_right_asset)
        
        for i in range(self.num_shadow_hand_tendons):
            name = self.gym.get_asset_tendon_name(shadow_hand_left_asset, i)
            if name.split("_")[-1] in relevant_tendon_dofs:
                tendon_props_left[i].limit_stiffness = limit_stiffness
                tendon_props_left[i].damping = damping
            name = self.gym.get_asset_tendon_name(shadow_hand_right_asset, i)
            if name.split("_")[-1] in relevant_tendon_dofs:
                tendon_props_right[i].limit_stiffness = limit_stiffness
                tendon_props_right[i].damping = damping
        self.gym.set_asset_tendon_properties(shadow_hand_left_asset, tendon_props_left)
        self.gym.set_asset_tendon_properties(shadow_hand_right_asset, tendon_props_right)
        
        # Setup actuator properties
        actuated_arm_dof_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", 
        ]
        actuated_hand_dof_names = [
            "rh_WRJ2", "rh_WRJ1", 
            "rh_FFJ4", "rh_FFJ3", "rh_FFJ2",
            "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2",
            "rh_MFJ4", "rh_MFJ3", "rh_MFJ2",
            "rh_RFJ4", "rh_RFJ3", "rh_RFJ2",
            "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"
        ]
        
        self.actuated_arm_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_right_asset, name) for name in actuated_arm_dof_names]
        self.actuated_hand_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_right_asset, name) for name in actuated_hand_dof_names]
        self.actuated_dof_indices = self.actuated_arm_dof_indices + self.actuated_hand_dof_indices

        # TODO: Set initial positions from config file
        init_arm_dof_positions_left = {
            "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -np.pi + 1.25, "elbow_joint": -2.0,
            "wrist_1_joint": -np.pi * 3 / 4, "wrist_2_joint": -np.pi / 2, "wrist_3_joint": 0.0,
        }
        init_arm_dof_positions_right = {
            "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.25, "elbow_joint": 2.0,
            "wrist_1_joint": -np.pi / 4, "wrist_2_joint": np.pi / 2, "wrist_3_joint": 0.0,
        }

        dof_props_left = self.gym.get_asset_dof_properties(shadow_hand_left_asset)
        dof_props_right = self.gym.get_asset_dof_properties(shadow_hand_right_asset)

        # effort, lower, upper, velocity, damping (hand) load from urdf
        # damping (arm), stiffness (arm) set to the values in IsaacSim
        # stiffness (hand) set to 3.0 (approximate value)
        # TODO: check the stiffness values with UK team
        default_dof_props_left = {
            "shoulder_pan_joint": {'effort': 330.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 2.0943951023931953, 'damping': 34.90659, 'stiffness': 349.06589},
            "shoulder_lift_joint": {'effort': 330.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 2.0943951023931953, 'damping': 34.90659, 'stiffness': 349.06589},
            "elbow_joint": {'effort': 150.0, 'lower': -3.141592653589793, 'upper': 3.141592653589793, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "wrist_1_joint": {'effort': 56.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "wrist_2_joint": {'effort': 56.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "wrist_3_joint": {'effort': 56.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "lh_WRJ2": {'effort': 10.0, 'lower': -0.5235987755982988, 'upper': 0.17453292519943295, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_WRJ1": {'effort': 30.0, 'lower': -0.6981317007977318, 'upper': 0.4886921905584123, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_FFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_FFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_FFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_FFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_MFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_MFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_MFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_MFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_RFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_RFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_RFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_RFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_LFJ5": {'effort': 2.0, 'lower': 0.0, 'upper': 0.7853981633974483, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_LFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_LFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_LFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_LFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_THJ5": {'effort': 5.0, 'lower': -1.0471975511965976, 'upper': 1.0471975511965976, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
            "lh_THJ4": {'effort': 3.0, 'lower': 0.0, 'upper': 1.2217304763960306, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
            "lh_THJ3": {'effort': 2.0, 'lower': -0.20943951023931953, 'upper': 0.20943951023931953, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
            "lh_THJ2": {'effort': 2.0, 'lower': -0.6981317007977318, 'upper': 0.6981317007977318, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "lh_THJ1": {'effort': 1.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
        }
        
        default_dof_props_right = {
            "shoulder_pan_joint": {'effort': 330.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 2.0943951023931953, 'damping': 34.90659, 'stiffness': 349.06589},
            "shoulder_lift_joint": {'effort': 330.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 2.0943951023931953, 'damping': 34.90659, 'stiffness': 349.06589},
            "elbow_joint": {'effort': 150.0, 'lower': -3.141592653589793, 'upper': 3.141592653589793, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "wrist_1_joint": {'effort': 56.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "wrist_2_joint": {'effort': 56.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "wrist_3_joint": {'effort': 56.0, 'lower': -6.283185307179586, 'upper': 6.283185307179586, 'velocity': 3.141592653589793, 'damping': 34.90659, 'stiffness': 349.06589},
            "rh_WRJ2": {'effort': 10.0, 'lower': -0.5235987755982988, 'upper': 0.17453292519943295, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_WRJ1": {'effort': 30.0, 'lower': -0.6981317007977318, 'upper': 0.4886921905584123, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_FFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_FFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_FFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_FFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_MFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_MFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_MFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_MFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_RFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_RFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_RFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_RFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_LFJ5": {'effort': 2.0, 'lower': 0.0, 'upper': 0.7853981633974483, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_LFJ4": {'effort': 2.0, 'lower': -0.3490658503988659, 'upper': 0.3490658503988659, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_LFJ3": {'effort': 2.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_LFJ2": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_LFJ1": {'effort': 2.0, 'lower': 0.0, 'upper': 1.5707963267948966, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_THJ5": {'effort': 5.0, 'lower': -1.0471975511965976, 'upper': 1.0471975511965976, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
            "rh_THJ4": {'effort': 3.0, 'lower': 0.0, 'upper': 1.2217304763960306, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
            "rh_THJ3": {'effort': 2.0, 'lower': -0.20943951023931953, 'upper': 0.20943951023931953, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
            "rh_THJ2": {'effort': 2.0, 'lower': -0.6981317007977318, 'upper': 0.6981317007977318, 'velocity': 2.0, 'damping': 0.1, 'stiffness': 3.0},
            "rh_THJ1": {'effort': 1.0, 'lower': -0.2617993877991494, 'upper': 1.5707963267948966, 'velocity': 4.0, 'damping': 0.2, 'stiffness': 3.0},
        }
        
        for name, props in default_dof_props_left.items():
            index = self.gym.find_asset_dof_index(shadow_hand_left_asset, name)
            dof_props_left["driveMode"] = gymapi.DofDriveMode.DOF_MODE_POS
            for key, value in props.items():
                dof_props_left[key][index] = value
                
        for name, props in default_dof_props_right.items():
            index = self.gym.find_asset_dof_index(shadow_hand_right_asset, name)
            dof_props_right["driveMode"] = gymapi.DofDriveMode.DOF_MODE_POS
            for key, value in props.items():
                dof_props_right[key][index] = value
        
        self._display_dof_properties(shadow_hand_left_asset, dof_props_left)
        
        # # WARNING: the following properties are estimated
        # # TODO: replace those properties with the real ones
        # for i in range(self.num_shadow_hand_dofs):
        #     dof_props_left["stiffness"][i] = 3
        #     dof_props_left["driveMode"][i] = gymapi.DofDriveMode.DOF_MODE_POS
        #     dof_props_left["velocity"][i] = 10
        #     dof_props_left["damping"][i] = 0.1
        #     dof_props_left['effort'][i] = 0.5
        #     dof_props_right["stiffness"][i] = 3
        #     dof_props_right["driveMode"][i] = gymapi.DofDriveMode.DOF_MODE_POS
        #     dof_props_right["velocity"][i] = 10
        #     dof_props_right["damping"][i] = 0.1
        #     dof_props_right['effort'][i] = 0.5
        
        # for i in range(0, 6):
        #     dof_props_left['stiffness'][i] = 400
        #     dof_props_left['damping'][i] = 80
        #     dof_props_left['effort'][i] = 200
        #     dof_props_right['stiffness'][i] = 400
        #     dof_props_right['damping'][i] = 80
        #     dof_props_right['effort'][i] = 200

        self.shadow_hand_dof_props_left = dof_props_left
        self.shadow_hand_dof_props_right = dof_props_right
        
        self.shadow_hand_dof_lower_limits = [dof_props_right["lower"][i] for i in range(self.num_shadow_hand_dofs)]
        self.shadow_hand_dof_upper_limits = [dof_props_right["upper"][i] for i in range(self.num_shadow_hand_dofs)]
        
        self.shadow_hand_left_dof_init_positions = [0.0 for _ in range(self.num_shadow_hand_dofs)]
        self.shadow_hand_left_dof_init_velocities = [0.0 for _ in range(self.num_shadow_hand_dofs)]
        self.shadow_hand_right_dof_init_positions = [0.0 for _ in range(self.num_shadow_hand_dofs)]
        self.shadow_hand_right_dof_init_velocities = [0.0 for _ in range(self.num_shadow_hand_dofs)]

        # Set initial dof positions
        for name, value in init_arm_dof_positions_left.items():
            index = self.gym.find_asset_dof_index(shadow_hand_left_asset, name)
            self.shadow_hand_left_dof_init_positions[index] = value
        
        for name, value in init_arm_dof_positions_right.items():
            index = self.gym.find_asset_dof_index(shadow_hand_right_asset, name)
            self.shadow_hand_right_dof_init_positions[index] = value
            
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.actuated_arm_dof_indices = to_torch(self.actuated_arm_dof_indices, dtype=torch.long, device=self.device)
        self.actuated_hand_dof_indices = to_torch(self.actuated_hand_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, dtype=torch.float, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, dtype=torch.float, device=self.device)
        self.shadow_hand_left_dof_init_positions = to_torch(self.shadow_hand_left_dof_init_positions, dtype=torch.float, device=self.device)
        self.shadow_hand_left_dof_init_velocities = to_torch(self.shadow_hand_left_dof_init_velocities, dtype=torch.float, device=self.device)
        self.shadow_hand_right_dof_init_positions = to_torch(self.shadow_hand_right_dof_init_positions, dtype=torch.float, device=self.device)
        self.shadow_hand_right_dof_init_velocities = to_torch(self.shadow_hand_right_dof_init_velocities, dtype=torch.float, device=self.device)
            
        # Set initial root pose
        shadow_hand_left_init_root_pose = gymapi.Transform()
        shadow_hand_left_init_root_pose.p = gymapi.Vec3(-0.20, -2.00, 0.20)
        shadow_hand_left_init_root_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, -np.pi / 2)
        
        shadow_hand_right_init_root_pose = gymapi.Transform()
        shadow_hand_right_init_root_pose.p = gymapi.Vec3(-0.20, 0.80, 0.20)
        shadow_hand_right_init_root_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, -np.pi / 2)
        
        self.shadow_hand_left_init_root_pose = shadow_hand_left_init_root_pose
        self.shadow_hand_right_init_root_pose = shadow_hand_right_init_root_pose
        
        # Get fingertip asset indices
        self.fingertip_left_asset_indices = [self.gym.find_asset_rigid_body_index(shadow_hand_left_asset, f"lh_{fingertip}") for fingertip in self.fingertips]
        self.fingertip_right_asset_indices = [self.gym.find_asset_rigid_body_index(shadow_hand_right_asset, f"rh_{fingertip}") for fingertip in self.fingertips]
        
        # Create force sensors
        for handle in self.fingertip_left_asset_indices:
            self.gym.create_asset_force_sensor(self.shadow_hand_left_asset, handle, gymapi.Transform())
        for handle in self.fingertip_right_asset_indices:
            self.gym.create_asset_force_sensor(self.shadow_hand_right_asset, handle, gymapi.Transform())
            
        # Get hand center asset indices
        self.shadow_hand_left_center_asset_index = self.gym.find_asset_rigid_body_index(shadow_hand_left_asset, f"lh_{self.hand_center}")
        self.shadow_hand_right_center_asset_index = self.gym.find_asset_rigid_body_index(shadow_hand_right_asset, f"rh_{self.hand_center}")

    def _create_table(self):
        table_texture_filepath = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture = self.gym.create_texture_from_file(self.sim, table_texture_filepath)
        
        # Create table asset
        table_dims = gymapi.Vec3(0.3, 0.3, 0.4)
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        
        table_init_pose = gymapi.Transform()
        table_init_pose.p = gymapi.Vec3(0.0, -0.6, 0.5 * table_dims.z)
        table_init_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)
        
        self.table_texture = table_texture
        self.table_asset = table_asset
        self.table_init_pose = table_init_pose

    def _create_object(self):
        asset_root = "../assets"
        object_asset_file = "mjcf/pot/mobility.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000
        
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)
        
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        object_init_pose = gymapi.Transform()
        object_init_pose.p = gymapi.Vec3(0, -0.6, 0.45)
        
        self.object_asset = object_asset
        self.object_init_pose = object_init_pose
        
    def _create_goal_object(self):
        asset_root = "../assets"
        object_asset_file = "mjcf/pot/mobility.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000
        asset_options.disable_gravity = True
        
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)

        goal_init_pose = gymapi.Transform()
        goal_init_pose.p = gymapi.Vec3(0, -0.6, 0.85)
        
        self.goal_asset = goal_asset
        self.goal_init_pose = goal_init_pose
        self.goal_displacement = gymapi.Vec3(0, 0, 0.4)
        self.goal_displacement_tensor = to_torch([self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        
    def _create_cameras(self):
        # TODO: Implement this function for camera creation
        if not self.enable_pointcloud_observation:
            return
        
        self.cameras = []
        self.camera_transforms = []
        
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.width = 640
        self.camera_properties.height = 480
        self.camera_properties.enable_tensors = True
        print(self.camera_properties)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Create multiple parallel isaacgym environments

        Args:
            num_envs (int): The total number of environment 

            spacing (float): Specifies half the side length of the square area occupied by each environment

            num_per_row (int): Specify how many environments in a row
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self._create_robots()
        self._create_table()
        self._create_object()
        self._create_goal_object()
        self._create_cameras()

        # Compute aggregate size
        # - shadow-hand-left    : self.num_shadow_hand_*
        # - shadow-hand-right   : self.num_shadow_hand_*
        # - object              : self.num_object_*
        # - goal                : self.num_object_*
        # - table               : 1
        max_aggregate_bodies = self.num_shadow_hand_bodies * 2 + self.num_object_bodies * 2 + 1
        max_aggregate_shapes = self.num_shadow_hand_shapes * 2 + self.num_object_shapes * 2 + 1

        self.envs = []

        self.object_init_state = []

        shadow_hand_left_indices = []
        shadow_hand_right_indices = []
        object_indices = []
        goal_indices = []
        table_indices = []

        if self.obs_type in ["point_cloud"]:
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.pointCloudDownsampleNum = 768
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

            if self.point_cloud_debug:
                import open3d as o3d
                from bidexhands.utils.o3dviewer import PointcloudVisualizer
                self.pointCloudVisualizer = PointcloudVisualizer()
                self.pointCloudVisualizerInitialized = False
                self.o3d_pc = o3d.geometry.PointCloud()
            else :
                self.pointCloudVisualizer = None

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode > 0:
                self.gym.begin_aggregate(env, max_aggregate_bodies, max_aggregate_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor_left = self.gym.create_actor(env, self.shadow_hand_left_asset, self.shadow_hand_left_init_root_pose, "left", i, 0, 0)
            shadow_hand_actor_right = self.gym.create_actor(env, self.shadow_hand_right_asset, self.shadow_hand_right_init_root_pose, "right", i, 0, 0)

            self.gym.set_actor_dof_properties(env, shadow_hand_actor_left, self.shadow_hand_dof_props_left)
            shadow_hand_left_indices.append(self.gym.get_actor_index(env, shadow_hand_actor_left, gymapi.DOMAIN_SIM))

            self.gym.set_actor_dof_properties(env, shadow_hand_actor_right, self.shadow_hand_dof_props_right)
            shadow_hand_right_indices.append(self.gym.get_actor_index(env, shadow_hand_actor_right, gymapi.DOMAIN_SIM))

            # Randomize hand color
            self._random_colorize(env, shadow_hand_actor_left, "left", part_level=False)
            self._random_colorize(env, shadow_hand_actor_right, "right", part_level=False)
            
            # Enable dof force sensors
            self.gym.enable_actor_dof_force_sensors(env, shadow_hand_actor_left)
            self.gym.enable_actor_dof_force_sensors(env, shadow_hand_actor_right)
            
            # Add object
            object_actor = self.gym.create_actor(env, self.object_asset, self.object_init_pose, "object", i, 0, 0)
            self.object_init_state.append(self._transform_to_tensor(self.object_init_pose))
            object_indices.append(self.gym.get_actor_index(env, object_actor, gymapi.DOMAIN_SIM))

            # Add goal
            goal_actor = self.gym.create_actor(env, self.goal_asset, self.goal_init_pose, "goal", i + self.num_envs, 0, 0)
            goal_indices.append(self.gym.get_actor_index(env, goal_actor, gymapi.DOMAIN_SIM))

            # Add table
            table_actor = self.gym.create_actor(env, self.table_asset, self.table_init_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env, table_actor, 0, gymapi.MESH_VISUAL, self.table_texture)
            table_indices.append(self.gym.get_actor_index(env, table_actor, gymapi.DOMAIN_SIM))

            # Set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env, table_actor)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env, object_actor)
            table_shape_props[0].friction = 3
            object_shape_props[0].friction = 3
            self.gym.set_actor_rigid_shape_properties(env, table_actor, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env, object_actor, object_shape_props)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(env, object_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(env, goal_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.obs_type in ["point_cloud"]:
                camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
                self.gym.set_camera_location(camera_handle, env, gymapi.Vec3(0.25, -0.57, 0.75), gymapi.Vec3(-0.24, -0.57, 0.25))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env, camera_handle), device=self.device)

                origin = self.gym.get_env_origin(env)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)
                
            if self.enable_pointcloud_observation:
                camera = self.gym.create_camera_sensor(env, self.camera_properties)
                self.gym.set_camera_location(camera, env, gymapi.Vec3(0.25, -0.57, 0.75), gymapi.Vec3(-0.24, -0.57, 0.25))
                image = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera, gymapi.IMAGE_DEPTH)
                image = gymtorch.wrap_tensor(image)
                
                view_matrix = self.gym.get_camera_view_matrix(self.sim, env, camera)
                proj_matrix = self.gym.get_camera_proj_matrix(self.sim, env, camera)
                
                print("view_matrix: ", view_matrix.shape)
                print(view_matrix)
                print("proj_matrix: ", proj_matrix.shape)
                print(proj_matrix)
                print("image: ", image.shape)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env)

            self.envs.append(env)
        self.object_init_state = torch.stack(self.object_init_state).to(self.device).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        
        # Get dof indices in env-domain
        shadow_hand_left_dof_indices = [self.gym.get_actor_dof_index(env, shadow_hand_actor_left, i, gymapi.DOMAIN_ENV) for i in range(self.gym.get_actor_dof_count(env, shadow_hand_actor_left))]
        shadow_hand_right_dof_indices = [self.gym.get_actor_dof_index(env, shadow_hand_actor_right, i, gymapi.DOMAIN_ENV) for i in range(self.gym.get_actor_dof_count(env, shadow_hand_actor_right))]
        self.shadow_hand_left_dof_indices = to_torch(shadow_hand_left_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_right_dof_indices = to_torch(shadow_hand_right_dof_indices, dtype=torch.long, device=self.device)
        
        self.shadow_hand_left_dof_start = self.gym.get_actor_dof_index(env, shadow_hand_actor_left, 0, gymapi.DOMAIN_ENV)
        self.shadow_hand_left_dof_end = self.shadow_hand_left_dof_start + self.gym.get_actor_dof_count(env, shadow_hand_actor_left)
        self.shadow_hand_right_dof_start = self.gym.get_actor_dof_index(env, shadow_hand_actor_right, 0, gymapi.DOMAIN_ENV)
        self.shadow_hand_right_dof_end = self.shadow_hand_right_dof_start + self.gym.get_actor_dof_count(env, shadow_hand_actor_right)
        print("shadow_hand_left_dof_start: ", self.shadow_hand_left_dof_start)
        print("shadow_hand_left_dof_end: ", self.shadow_hand_left_dof_end)
        print("shadow_hand_right_dof_start: ", self.shadow_hand_right_dof_start)
        print("shadow_hand_right_dof_end: ", self.shadow_hand_right_dof_end)

        fingertip_left_rigid_body_indices = [self.gym.get_actor_rigid_body_index(env, shadow_hand_actor_left, i, gymapi.DOMAIN_ENV) for i in self.fingertip_left_asset_indices]
        fingertip_right_rigid_body_indices = [self.gym.get_actor_rigid_body_index(env, shadow_hand_actor_right, i, gymapi.DOMAIN_ENV) for i in self.fingertip_right_asset_indices]
        self.fingertip_left_rigid_body_indices = to_torch(fingertip_left_rigid_body_indices, dtype=torch.long, device=self.device)
        self.fingertip_right_rigid_body_indices = to_torch(fingertip_right_rigid_body_indices, dtype=torch.long, device=self.device)
        
        self.shadow_hand_left_center_index = self.gym.get_actor_rigid_body_index(env, shadow_hand_actor_left, self.shadow_hand_left_center_asset_index, gymapi.DOMAIN_ENV)
        self.shadow_hand_right_center_index = self.gym.get_actor_rigid_body_index(env, shadow_hand_actor_right, self.shadow_hand_right_center_asset_index, gymapi.DOMAIN_ENV)

        self.shadow_hand_left_indices = to_torch(shadow_hand_left_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_right_indices = to_torch(shadow_hand_right_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(object_indices, dtype=torch.long, device=self.device)
        self.goal_indices = to_torch(goal_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(table_indices, dtype=torch.long, device=self.device)
        
        self.shadow_hand_left_index = self.gym.get_actor_index(env, shadow_hand_actor_left, gymapi.DOMAIN_ENV)
        self.shadow_hand_right_index = self.gym.get_actor_index(env, shadow_hand_actor_right, gymapi.DOMAIN_ENV)
        self.object_index = self.gym.get_actor_index(env, object_actor, gymapi.DOMAIN_ENV)
        self.goal_index = self.gym.get_actor_index(env, goal_actor, gymapi.DOMAIN_ENV)
        self.table_index = self.gym.get_actor_index(env, table_actor, gymapi.DOMAIN_ENV)
        
        shadow_hand_left_actuated_arm_dof_indices = [self.gym.get_actor_dof_index(env, shadow_hand_actor_left, i, gymapi.DOMAIN_ENV) for i in self.actuated_arm_dof_indices]
        shadow_hand_left_actuated_hand_dof_indices = [self.gym.get_actor_dof_index(env, shadow_hand_actor_left, i, gymapi.DOMAIN_ENV) for i in self.actuated_hand_dof_indices]
        shadow_hand_left_actuated_dof_indices = shadow_hand_left_actuated_arm_dof_indices + shadow_hand_left_actuated_hand_dof_indices
        
        shadow_hand_right_actuated_arm_dof_indices = [self.gym.get_actor_dof_index(env, shadow_hand_actor_right, i, gymapi.DOMAIN_ENV) for i in self.actuated_arm_dof_indices]
        shadow_hand_right_actuated_hand_dof_indices = [self.gym.get_actor_dof_index(env, shadow_hand_actor_right, i, gymapi.DOMAIN_ENV) for i in self.actuated_hand_dof_indices]
        shadow_hand_right_actuated_dof_indices = shadow_hand_right_actuated_arm_dof_indices + shadow_hand_right_actuated_hand_dof_indices
        
        shadow_hand_actuated_dof_indices = shadow_hand_left_actuated_dof_indices + shadow_hand_right_actuated_dof_indices
        
        self.shadow_hand_left_actuated_dof_indices = to_torch(shadow_hand_left_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_right_actuated_dof_indices = to_torch(shadow_hand_right_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_actuated_dof_indices = to_torch(shadow_hand_actuated_dof_indices, dtype=torch.long, device=self.device)
        print("shadow_hand_left_actuated_dof_indices: ", shadow_hand_left_actuated_dof_indices)
        
    def compute_reward(self, actions):
        """
        Compute the reward of all environment. The core function is compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.pot_left_handle_pos, self.pot_right_handle_pos,
            self.left_hand_pos, self.right_hand_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        , which we will introduce in detail there

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_root_positions, self.object_root_orientations, self.goal_pos, self.goal_rot, self.pot_left_handle_pos, self.pot_right_handle_pos,
            self.shadow_hand_left_positions, self.shadow_hand_right_positions,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_rigid_bodies, 13)
        self.shadow_hand_left_positions = rigid_body_states[:, self.shadow_hand_left_center_index, :3]
        self.shadow_hand_left_orientations = rigid_body_states[:, self.shadow_hand_left_center_index, 3:7]
        self.shadow_hand_left_linear_velocities = rigid_body_states[:, self.shadow_hand_left_center_index, 7:10]
        self.shadow_hand_left_angular_velocities = rigid_body_states[:, self.shadow_hand_left_center_index, 10:13]
        
        self.shadow_hand_right_positions = rigid_body_states[:, self.shadow_hand_right_center_index, :3]
        self.shadow_hand_right_orientations = rigid_body_states[:, self.shadow_hand_right_center_index, 3:7]
        self.shadow_hand_right_linear_velocities = rigid_body_states[:, self.shadow_hand_right_center_index, 7:10]
        self.shadow_hand_right_angular_velocities = rigid_body_states[:, self.shadow_hand_right_center_index, 10:13]

        self.shadow_hand_left_positions = self.shadow_hand_left_positions + quat_apply(self.shadow_hand_left_orientations, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.shadow_hand_left_positions = self.shadow_hand_left_positions + quat_apply(self.shadow_hand_left_orientations, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.shadow_hand_right_positions = self.shadow_hand_right_positions + quat_apply(self.shadow_hand_right_orientations, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.shadow_hand_right_positions = self.shadow_hand_right_positions + quat_apply(self.shadow_hand_right_orientations, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.fingertip_left_states = rigid_body_states[:, self.fingertip_left_rigid_body_indices, :]
        self.fingertip_left_positions = self.fingertip_left_states[:, :, :3]
        self.fingertip_left_orientations = self.fingertip_left_states[:, :, 3:7]
        self.fingertip_left_linear_velocities = self.fingertip_left_states[:, :, 7:10]
        self.fingertip_left_angular_velocities = self.fingertip_left_states[:, :, 10:13]
        
        self.fingertip_right_states = rigid_body_states[:, self.fingertip_right_rigid_body_indices, :]
        self.fingertip_right_positions = self.fingertip_right_states[:, :, :3]
        self.fingertip_right_orientations = self.fingertip_right_states[:, :, 3:7]
        self.fingertip_right_linear_velocities = self.fingertip_right_states[:, :, 7:10]
        self.fingertip_right_angular_velocities = self.fingertip_right_states[:, :, 10:13]

        self.pot_left_handle_positions = self.object_root_positions + quat_apply(self.object_root_orientations, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.pot_left_handle_positions = self.pot_left_handle_positions + quat_apply(self.object_root_orientations, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.pot_right_handle_positions = self.object_root_positions + quat_apply(self.object_root_orientations, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.15)
        self.pot_right_handle_positions = self.pot_right_handle_positions + quat_apply(self.object_root_orientations, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.06)

    def compute_observations(self):
        """
        Compute the observations of all environment. The core function is self.compute_full_state(True), 
        which we will introduce in detail there

        """
        self._refresh_tensors()

        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.pot_right_handle_pos = self.object_root_positions + quat_apply(self.object_root_orientations, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.15)
        self.pot_right_handle_pos = self.pot_right_handle_pos + quat_apply(self.object_root_orientations, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.pot_left_handle_pos = self.object_root_positions + quat_apply(self.object_root_orientations, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.pot_left_handle_pos = self.pot_left_handle_pos + quat_apply(self.object_root_orientations, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.06)

        # TODO: figure out which rigid_body
        # self.left_hand_pos = self.rigid_body_states[:, 3 + 26, 0:3]
        # self.left_hand_rot = self.rigid_body_states[:, 3 + 26, 3:7]
        # self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        # self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        # self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        # self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:13]
        # self.fingertip_another_pos = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:3]

        # TODO: point-cloud observation and state-based observation should in one single pipeline
        
        # TODO: should only have one `compute_observations` function
        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        self.obs_buf[:] = self._pack_observations()

    def compute_point_cloud_observation(self, collect_demonstration=False):
        pass
        
    def reset_target_pose(self, env_ids, apply_reset=False):
        """
        Reset and randomize the goal pose

        Args:
            env_ids (tensor): The index of the environment that needs to reset goal pose

            apply_reset (bool): Whether to reset the goal directly here, usually used 
            when the same task wants to complete multiple goals

        """
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        # self.goal_states[env_ids, 1] -= 0.25
        self.goal_states[env_ids, 2] += 0.4

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_states[self.goal_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_states[self.goal_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_states[self.goal_indices[env_ids], 7:13] = 0.0

        if apply_reset:
            goal_indices = self.goal_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(goal_indices), 
                len(env_ids),
            )
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        """
        Reset and randomize the environment

        Args:
            env_ids (tensor): The index of the environment that needs to reset

            goal_env_ids (tensor): The index of the environment that only goals need reset

        """
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # Reset object initial states
        noise = torch.rand(env_ids.shape[0], 3, device=self.device) * 2.0 - 1.0
        self.root_states[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_states[self.object_indices[env_ids], :3] += noise * self.reset_position_noise
        self.root_states[self.object_indices[env_ids], 7:13] = 0.0
        
        indices = torch.unique(
            torch.cat(
                [
                    self.object_indices[env_ids],
                    self.goal_indices[env_ids],
                    self.goal_indices[goal_env_ids],
                ]
            ).to(torch.int32)
        )
        
        # Reset shadow hand DOF positions & velocities
        delta_upper = self.shadow_hand_dof_upper_limits - self.shadow_hand_left_dof_init_positions
        delta_lower = self.shadow_hand_dof_lower_limits - self.shadow_hand_left_dof_init_positions
        noise = torch.rand(env_ids.shape[0], self.num_shadow_hand_dofs, device=self.device)
        noise = noise * (delta_upper - delta_lower) + delta_lower
        dof_init_positions = self.shadow_hand_left_dof_init_positions + noise * self.reset_dof_pos_noise
        self.shadow_hand_left_dof_positions[env_ids, :] = dof_init_positions
        
        delta_upper = self.shadow_hand_dof_upper_limits - self.shadow_hand_right_dof_init_positions
        delta_lower = self.shadow_hand_dof_lower_limits - self.shadow_hand_right_dof_init_positions
        noise = torch.rand(env_ids.shape[0], self.num_shadow_hand_dofs, device=self.device)
        noise = noise * (delta_upper - delta_lower) + delta_lower
        dof_init_positions = self.shadow_hand_right_dof_init_positions + noise * self.reset_dof_pos_noise
        self.shadow_hand_right_dof_positions[env_ids, :] = dof_init_positions

        noise = torch.rand(env_ids.shape[0], self.num_shadow_hand_dofs, device=self.device) * 2.0 - 1.0
        self.shadow_hand_left_dof_velocities[env_ids, :] = self.shadow_hand_left_dof_init_velocities + noise * self.reset_dof_vel_noise
        self.shadow_hand_right_dof_velocities[env_ids, :] = self.shadow_hand_right_dof_init_velocities + noise * self.reset_dof_vel_noise
        
        self.prev_targets[env_ids, self.shadow_hand_left_dof_start:self.shadow_hand_left_dof_end] = dof_init_positions
        self.cur_targets[env_ids, self.shadow_hand_left_dof_start:self.shadow_hand_left_dof_end] = dof_init_positions

        self.prev_targets[env_ids, self.shadow_hand_right_dof_start:self.shadow_hand_right_dof_end] = dof_init_positions
        self.cur_targets[env_ids, self.shadow_hand_right_dof_start:self.shadow_hand_right_dof_end] = dof_init_positions

        indices = torch.unique(
            torch.cat(
                [
                    self.shadow_hand_left_indices[env_ids].to(torch.int32),
                    self.shadow_hand_right_indices[env_ids].to(torch.int32),
                    indices,
                ]
            ).to(torch.int32)
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(indices), 
            len(indices),
        )
        
        self.root_states[indices.to(torch.long), :] = self.cached_root_states[indices.to(torch.long), :]
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(indices), 
            len(indices),
        )
        
        indices = torch.unique(torch.cat([indices, self.table_indices[env_ids]]).to(torch.int32))                                      
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(indices), 
            len(indices),
        )
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        """
        The pre-processing of the physics step. Determine whether the reset environment is needed, 
        and calculate the next movement of Shadowhand through the given action. The 52-dimensional 
        action space as shown in below:
        
        Index   Description
        0 - 19 	right shadow hand actuated joint
        20 - 22	right shadow hand base translation
        23 - 25	right shadow hand base rotation
        26 - 45	left shadow hand actuated joint
        46 - 48	left shadow hand base translation
        49 - 51	left shadow hand base rotatio

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        # TODO: the FF/MF/RF/LF J2 and J1 should share 1 dof
        # TODO: the range of this dof should be [0, \pi]
        # TODO: value_J2 = min(dof, \pi / 2)
        # TODO: value_J1 = max(dof - \pi / 2, 0)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        self.shadow_hand_left_actions = self.actions[:, :self.actions.shape[1] // 2]
        self.shadow_hand_right_actions = self.actions[:, self.actions.shape[1] // 2:]

        # TODO: IK control
        if self.use_relative_control:
            self.cur_targets[:, self.shadow_hand_actuated_dof_indices] = (
                self.prev_targets[:, self.shadow_hand_actuated_dof_indices] 
                + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            )
        else:
            assert self.actions.shape[1] == len(self.shadow_hand_actuated_dof_indices)
            self.cur_targets[:, self.shadow_hand_left_actuated_dof_indices] = scale(
                self.shadow_hand_left_actions,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
            self.cur_targets[:, self.shadow_hand_right_actuated_dof_indices] = scale(
                self.shadow_hand_right_actions,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
            self.cur_targets[:, self.shadow_hand_actuated_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.shadow_hand_actuated_dof_indices] 
                + (1.0 - self.act_moving_average) * self.prev_targets[:, self.shadow_hand_actuated_dof_indices]
            )
            
        self.cur_targets[:, self.shadow_hand_left_actuated_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.shadow_hand_left_actuated_dof_indices],
            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
            self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
        )
        self.cur_targets[:, self.shadow_hand_right_actuated_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.shadow_hand_right_actuated_dof_indices],
            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
            self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
        )

        self.prev_targets[:, self.shadow_hand_actuated_dof_indices] = self.cur_targets[:, self.shadow_hand_actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def _draw_axes(self, positions: torch.Tensor, orientations: torch.Tensor, length: float = 0.2):
        assert positions.ndim == 2 and positions.shape == (self.num_envs, 3)
        assert orientations.ndim == 2 and orientations.shape == (self.num_envs, 4)
        
        x = (positions + quat_apply(orientations, self.x_unit_tensor * length)).detach().cpu().numpy()
        y = (positions + quat_apply(orientations, self.y_unit_tensor * length)).detach().cpu().numpy()
        z = (positions + quat_apply(orientations, self.z_unit_tensor * length)).detach().cpu().numpy()
        
        positions = positions.detach().cpu().numpy()
        
        for i in range(self.num_envs):
            self.gym.add_lines(self.viewer, self.envs[i], 1, np.concatenate([positions[i], x[i]]), np.array([1, 0, 0], dtype=np.float32))
            self.gym.add_lines(self.viewer, self.envs[i], 1, np.concatenate([positions[i], y[i]]), np.array([0, 1, 0], dtype=np.float32))
            self.gym.add_lines(self.viewer, self.envs[i], 1, np.concatenate([positions[i], z[i]]), np.array([0, 0, 1], dtype=np.float32))

    def post_physics_step(self):
        """
        The post-processing of the physics step. Compute the observation and reward, and visualize auxiliary 
        lines for debug when needed
        
        """
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            self._draw_axes(self.goal_root_positions, self.goal_root_orientations)
            
            self._draw_axes(self.object_root_positions, self.object_root_orientations)
            self._draw_axes(self.pot_left_handle_positions, self.object_root_orientations)
            self._draw_axes(self.pot_right_handle_positions, self.object_root_orientations)
            
            self._draw_axes(self.shadow_hand_left_positions, self.shadow_hand_left_orientations)
            self._draw_axes(self.shadow_hand_right_positions, self.shadow_hand_right_orientations)

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                         to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        return camera_image
        
#####################################################################
###=========================jit functions=========================###
#####################################################################


def pointcloud_from_depth(image, view_matrix, projection_matrix):
    # TODO: test this function
    # TODO: add support for batched inputs
    height, width = image.shape
    vinv = torch.inverse(view_matrix)
    
    fu, fv = 2 / projection_matrix[0, 0], 2 / projection_matrix[1, 1]
    cu, cv = width / 2, height / 2
    
    ii = torch.arange(0, width, device=image.device).reshape(1, -1).repeat(height, 1)
    jj = torch.arange(0, height, device=image.device).reshape(-1, 1).repeat(1, width)
    
    u = - (ii - cu) / width * image * fu
    v = (jj - cv) / height * image * fv
    z = image
    
    pointcloud = torch.stack([u, v, z], dim=-1)
    pointcloud = pointcloud.reshape(-1, 3)
    pointcloud = torch.matmul(pointcloud, vinv[:3, :3].t()) + vinv[:3, 3]
    return pointcloud
    

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, pot_left_handle_pos, pot_right_handle_pos, 
    left_hand_pos, right_hand_pos, 
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    """
    Compute the reward of all environment.

    Args:
        rew_buf (tensor): The reward buffer of all environments at this time

        reset_buf (tensor): The reset buffer of all environments at this time

        reset_goal_buf (tensor): The only-goal reset buffer of all environments at this time

        progress_buf (tensor): The porgress buffer of all environments at this time

        successes (tensor): The successes buffer of all environments at this time

        consecutive_successes (tensor): The consecutive successes buffer of all environments at this time

        max_episode_length (float): The max episode length in this environment

        object_pos (tensor): The position of the object

        object_rot (tensor): The rotation of the object

        target_pos (tensor): The position of the target

        target_rot (tensor): The rotate of the target

        pot_left_handle_pos (tensor): The position of the left handle of the pot

        pot_right_handle_pos (tensor): The position of the right handle of the pot

        left_hand_pos, right_hand_pos (tensor): The position of the bimanual hands

        dist_reward_scale (float): The scale of the distance reward

        rot_reward_scale (float): The scale of the rotation reward

        rot_eps (float): The epsilon of the rotation calculate

        actions (tensor): The action buffer of all environments at this time

        action_penalty_scale (float): The scale of the action penalty reward

        success_tolerance (float): The tolerance of the success determined

        reach_goal_bonus (float): The reward given when the object reaches the goal

        fall_dist (float): When the object is far from the Shadowhand, it is judged as falling

        fall_penalty (float): The reward given when the object is fell

        max_consecutive_successes (float): The maximum of the consecutive successes

        av_factor (float): The average factor for calculate the consecutive successes

        ignore_z_rot (bool): Is it necessary to ignore the rot of the z-axis, which is usually used 
            for some specific objects (e.g. pen)
    """
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # goal_dist = target_pos[:, 2] - object_pos[:, 2]
    right_hand_dist = torch.norm(pot_right_handle_pos - right_hand_pos, p=2, dim=-1)
    left_hand_dist = torch.norm(pot_left_handle_pos - left_hand_pos, p=2, dim=-1)
    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    right_hand_dist_rew = right_hand_dist
    left_hand_dist_rew = left_hand_dist

    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
    up_rew = torch.zeros_like(right_hand_dist_rew)
    mask = (right_hand_dist < 0.18) & (left_hand_dist < 0.18)
    up_rew = torch.where(mask, 3 * (0.385 - goal_dist), up_rew)
    
    reward = 0.5 - right_hand_dist_rew - left_hand_dist_rew + up_rew

    resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(right_hand_dist >= 0.3, torch.ones_like(resets), resets)
    resets = torch.where(left_hand_dist >= 0.3, torch.ones_like(resets), resets)

    # Find out which envs hit the goal and update successes count
    successes = torch.where(successes == 0, 
                    torch.where(goal_dist < 0.05, torch.ones_like(successes), successes), successes)

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
