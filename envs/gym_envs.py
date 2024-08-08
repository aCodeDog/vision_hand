from avp_stream.isaac_env import IsaacVisualizerEnv
import isaacgym
import torch 
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import numpy as np
import torch
import time
from pathlib import Path
import math
from avp_stream import VisionProStreamer
from avp_stream.utils.isaac_utils import * 
from avp_stream.utils.se3_utils import * 
from avp_stream.utils.trn_constants import * 
from copy import deepcopy
from typing import * 

from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

CUR_PATH = Path(__file__).parent.resolve()
asset_root = f'{CUR_PATH}/assets'
left_asset_path = "inspire_hand/inspire_hand_left.urdf"
right_asset_path = "inspire_hand/inspire_hand_right.urdf"
retarget_config = f'{CUR_PATH}/assets'+"/inspire_hand/inspire_hand.yml"
cube_asset_path = "cube.urdf"
ball_asset_path = "ball.urdf"


tip_indices = [4, 9, 14, 19, 24]

hand2inspire_right = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])


hand2inspire_left = np.array([[0, -1, 0, 0],
                              [-1, 0, 0, 0],
                              [0., 0, -1, 0],
                              [0, 0, 0, 1]])

head_mat = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 1.5],
                    [0, 0, 1, -0.2],
                    [0, 0, 0, 1]])


right_wrist_mat = np.array([[1, 0, 0, 0.5],
                            [0, 1, 0, 1],
                            [0, 0, 1, -0.5],
                            [0, 0, 0, 1]])
left_wrist_mat = np.array([[1, 0, 0, -0.5],
                            [0, 1, 0, 1],
                            [0, 0, 1, -0.5],
                            [0, 0, 0, 1]])

class IsaacHandVisualizerEnv(IsaacVisualizerEnv): 

    def __init__(self, args):

        self.args = args 
        
        # acquire gym interface
        self.gym = gymapi.acquire_gym()
 
        # set torch device
        self.device = 'cpu'  # i'll just fix this to CUDA 

        # configure sim
        self.sim_params = default_sim_params(use_gpu = True if self.device == 'cuda:0' else False) 

        # create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        self.left_key_point_indices = torch.zeros(3, dtype=torch.long, device=self.device, requires_grad=False)
        self.right_key_point_indices= torch.zeros(3, dtype=torch.long, device=self.device, requires_grad=False)
        # load assets
        self.num_envs = 1
        self.t = 0.0
        


        # create env 
        self._load_asset()
        self.create_env() 

        # setup viewer camera
        middle_env = self.num_envs // 2
        setup_viewer_camera(self.gym, self.envs[middle_env], self.viewer)

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)
        self.initialize_tensors()

        RetargetingConfig.set_default_urdf_dir(asset_root)
        config_file_path = retarget_config
        
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

        self.head_mat = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 1.5],
                                [0, 0, 1, -0.2],
                                [0, 0, 0, 1]])
        self.right_wrist_mat = np.array([[1, 0, 0, 0.5],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -0.5],
                                        [0, 0, 0, 1]])
        self.left_wrist_mat = np.array([[1, 0, 0, -0.5],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -0.5],
                                        [0, 0, 0, 1]])
        
    def to_torch(self, x, dtype=torch.float, requires_grad=False):
        return torch.tensor(x, dtype=dtype, device=self.device, requires_grad=requires_grad)


    def _load_asset(self):

        self.axis = load_axis(self.gym, self.sim, self.device, 'normal', f'{CUR_PATH}/assets')
        self.small_axis = load_axis(self.gym, self.sim, self.device, 'small', f'{CUR_PATH}/assets')
        self.huge_axis = load_axis(self.gym, self.sim, self.device, 'huge', f'{CUR_PATH}/assets')

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        self.sphere = self.gym.create_sphere(self.sim, 0.008, asset_options)


    def create_env(self):

        self.envs = []
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

            # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)





        cube_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_path, gymapi.AssetOptions())
        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_path, gymapi.AssetOptions())



        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.dof = self.gym.get_asset_dof_count(left_asset)





        # set up the env grid
        num_envs = 1
        self.num_envs = num_envs

        for env_idx in range(self.num_envs):



            num_per_row = int(math.sqrt(num_envs))
            env_spacing = 1.25
            env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
            env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
            np.random.seed(0)
                
            self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(self.env)

            # table
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0.5, 0.7)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', env_idx,0)
            color = gymapi.Vec3(0.5, 0.5, 0.5)
            self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # cube
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0.5, 0.80)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            cube_handle_0 = self.gym.create_actor(self.env, cube_asset, pose, 'cube0', env_idx,0)
            color = gymapi.Vec3(1, 0.5, 0.5)
            self.gym.set_rigid_body_color(self.env, cube_handle_0, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # cube
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0.3, 0.80)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            cube_handle_1 = self.gym.create_actor(self.env, cube_asset, pose, 'cube1', env_idx,0)
            color = gymapi.Vec3(1, 1, 1)
            self.gym.set_rigid_body_color(self.env, cube_handle_1, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            # ball 
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.3, 0.3, 0.80)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            ball_handle = self.gym.create_actor(self.env, ball_asset, pose, 'ball_0', env_idx,0)
            color = gymapi.Vec3(1, 1, 1)
            self.gym.set_rigid_body_color(self.env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            #axis
            self.head_axis = self.gym.create_actor(self.env, self.axis, gymapi.Transform(), 'head', env_idx)
            self.right_wrist_axis = self.gym.create_actor(self.env, self.axis, gymapi.Transform(), 'right_wrist', env_idx)
            self.left_wrist_axis = self.gym.create_actor(self.env, self.axis, gymapi.Transform(), 'left_wrist', env_idx)

            self.head_axis_idx = self.gym.get_actor_index(self.env, self.head_axis, gymapi.DOMAIN_SIM)
            self.right_wrist_axis_idx = self.gym.get_actor_index(self.env, self.right_wrist_axis, gymapi.DOMAIN_SIM)
            self.left_wrist_axis_idx = self.gym.get_actor_index(self.env, self.left_wrist_axis, gymapi.DOMAIN_SIM)




            self.head_actor_idx = self.gym.get_actor_index(self.env, self.head_axis, gymapi.DOMAIN_SIM)
            self.right_wrist_actor_idx = self.gym.get_actor_index(self.env, self.right_wrist_axis, gymapi.DOMAIN_SIM)
            self.left_wrist_actor_idx = self.gym.get_actor_index(self.env, self.left_wrist_axis, gymapi.DOMAIN_SIM)


            # left_hand
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(-0.6, 0, 1.6)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', env_idx, 0)
            self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                        gymapi.STATE_ALL)
            left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

            # right_hand
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(-0.6, 0, 1.6)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', env_idx, 0)
            self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                        gymapi.STATE_ALL)
            right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

            self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
            self.left_root_states = self.root_states[left_idx]
            self.right_root_states = self.root_states[right_idx]

            self.head_axis_states = self.root_states[self.head_axis_idx]
            self.right_wrist_axis_states = self.root_states[self.right_wrist_axis_idx]
            self.left_wrist_axis_states = self.root_states[self.left_wrist_axis_idx]




            self.left_wrist_root_state = self.root_states[self.left_wrist_actor_idx]
            self.right_wrist_root_state = self.root_states[self.right_wrist_actor_idx]
            self.head_root_state = self.root_states[self.head_actor_idx]

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 2, 3)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 3.2])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))


    def initialize_tensors(self): 
        
        refresh_tensors(self.gym, self.sim)
        # get jacobian tensor
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, -1, 13)

        # get actor root state tensor
        #_root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        # root_states = gymtorch.wrap_tensor(_root_states).view(self.num_envs, -1, 13)
        # self.root_state = root_states

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    # will be overloaded
    def euler_from_quat(self,quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians


    def compute_angle(self,T_W_Metacarpal,T_W_Intermediate,T_W_Tip):

        pos_B = T_W_Intermediate[:3, 3]
        pos_C = T_W_Tip[:3, 3]

        # Vector BC
        vec_B_to_C_W = pos_B - pos_C 


        T_A_W = np.linalg.inv(T_W_Metacarpal)

        # # 将B到C的连线向量转换到A的坐标系下
        vec_B_to_C_A = T_A_W[:3, :3].dot(vec_B_to_C_W)

        # # 计算与XYZ轴的夹角
        angle_x = np.arccos(vec_B_to_C_A[0] / np.linalg.norm(vec_B_to_C_A))
        angle_y = np.arccos(vec_B_to_C_A[1] / np.linalg.norm(vec_B_to_C_A))
        angle_z = np.arccos(vec_B_to_C_A[2] / np.linalg.norm(vec_B_to_C_A))
        return angle_x,angle_y,angle_z


    
    def mat_update(self,prev_mat, mat):
        if np.linalg.det(mat) == 0:
            return prev_mat
        else:
            return mat


    def fast_mat_inv(self,mat):
        ret = np.eye(4)
        ret[:3, :3] = mat[:3, :3].T
        ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
        return ret

    def step(self, transformation: Dict[str, torch.Tensor] , sync_frame_time = False): 

        self.simulate()

        self.head_mat = self.mat_update(self.head_mat,transformation['head'].view(4,4).clone().detach().numpy())
        self.right_wrist_mat = self.mat_update(self.right_wrist_mat, transformation['right_wrist'].view(4,4).clone().detach().numpy())
        self.left_wrist_mat = self.mat_update(self.left_wrist_mat, transformation['left_wrist'].view(4,4).clone().detach().numpy())


        self.rel_right_wrist_mat =  self.right_wrist_mat  @ hand2inspire_right

        self.rel_left_wrist_mat =  self.left_wrist_mat  @ hand2inspire_left



        # homogeneous
        # left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        # right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        

        right_finger_raw_mat = torch.cat([transformation['right_wrist'] @ finger for finger in transformation['right_fingers']], dim = 0)

        right_fingers_pose = mat2posquat(right_finger_raw_mat)   #0,4,14,[1,25,7].clone().detach().numpy()
        right_fingers = right_fingers_pose[:,:3].view(25,3).clone().detach().numpy()
        

        left_fingers_raw_mat = torch.cat([transformation['left_wrist'] @ finger for finger in transformation['left_fingers']], dim = 0)

        left_fingers_pose = mat2posquat(left_fingers_raw_mat)   #0,4,14,[1,25,7].clone().detach().numpy()
        #print(left_fingers_pose.shape) 
        left_fingers = left_fingers_pose[:,:3].view(25,3).clone().detach().numpy()
        # homogeneous
        left_fingers_reshape = np.concatenate([left_fingers.copy().T, np.ones((1, left_fingers.shape[0]))])
        right_fingers_reshape = np.concatenate([right_fingers.copy().T, np.ones((1, right_fingers.shape[0]))])
        # change of basis
        left_fingers = left_fingers_reshape
        right_fingers = right_fingers_reshape

        rel_left_fingers = self.fast_mat_inv(self.left_wrist_mat) @ left_fingers
        rel_right_fingers = self.fast_mat_inv(self.right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire_left.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire_right.T @ rel_right_fingers)[0:3, :].T

        head_rmat = self.head_mat[:3, :3]  

        
        left_pose = mat2posquat(torch.tensor(self.rel_left_wrist_mat).view(1,4,4))
        right_pose = mat2posquat(torch.tensor(self.rel_right_wrist_mat).view(1,4,4))

        left_qpos = self.left_retargeting.retarget(rel_left_fingers[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(rel_right_fingers[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]


        #set hand root state
        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        
        self.head_axis_states[0:7] = mat2posquat(torch.tensor(self.head_mat).view(1,4,4))
        self.right_wrist_axis_states[0:7] = left_pose
        self.left_wrist_axis_states[0:7] = right_pose
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


        #set joint state
        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        left_states['pos'] = left_qpos
        self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states['pos'] = right_qpos
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)

        self.render(sync_frame_time)

    def move_camera(self):

        head_xyz = self.visionos_head[:, :3, 3]
        head_ydir = self.visionos_head[:, :3, 1]

        cam_pos = head_xyz - head_ydir * 0.5
        cam_target = head_xyz + head_ydir * 0.5
        cam_target[..., -1] -= 0.2

        cam_pos = gymapi.Vec3(*cam_pos[0])
        cam_target = gymapi.Vec3(*cam_target[0])

        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def simulate(self): 
        # step the physics
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        # refresh tensors
        refresh_tensors(self.gym, self.sim)


    def render(self, sync_frame_time = True): 

        # update viewer
        if self.args.follow:
            self.move_camera()
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        if sync_frame_time:
            self.gym.sync_frame_time(self.sim)

def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:  
    for key in data.keys():
        data[key] = torch.tensor(data[key], dtype = torch.float32, device = device)
    return data

