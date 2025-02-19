import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

class UAVUSVEnv:
    def __init__(self, length=2, num_obstacle=3, num_uavs=2, num_usvs=2):
        self.length = length
        self.num_obstacle = num_obstacle
        self.num_uavs = num_uavs
        self.num_usvs = num_usvs
        self.num_agents = num_uavs + num_usvs
        self.time_step = 0.5
        self.v_max_uav = 0.1
        self.v_max_usv = 0.05
        self.a_max_uav = 0.04
        self.a_max_usv = 0.02
        self.L_sensor = 0.2
        self.num_lasers = 16
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['uav_0', 'uav_1', 'usv_0', 'usv_1']
        self.info = np.random.get_state()
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(self.num_agents)]

        self.action_space = {
            'uav_0': spaces.Box(low=-1, high=1, shape=(2,)),
            'uav_1': spaces.Box(low=-1, high=1, shape=(2,)),
            'usv_0': spaces.Box(low=-1, high=1, shape=(2,)),
            'usv_1': spaces.Box(low=-1, high=1, shape=(2,))
        }
        self.observation_space = {
            'uav_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),  # 4 + 4 + 16 + 2
            'uav_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),  # 4 + 4 + 16 + 2
            'usv_0': spaces.Box(low=-np.inf, high=np.inf, shape=(20,)),  # 4 + 16
            'usv_1': spaces.Box(low=-np.inf, high=np.inf, shape=(20,))   # 4 + 16
        }
        

    def reset(self):
        SEED = random.randint(1,1000)
        random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i < self.num_uavs: # if UAV
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else: # if USV
                self.multi_current_pos.append(np.array([0.5,1.75]))
            self.multi_current_vel.append(np.zeros(2)) # initial velocity = [0,0]

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self,actions):
        last_d2target = []
        # print(actions)
        # time.sleep(0.1)
        for i in range(self.num_agents):

            pos = self.multi_current_pos[i]
            if i < self.num_uavs:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))
            
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i < self.num_uavs:
                if vel_magnitude >= self.v_max_uav:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_uav
            else:
                if vel_magnitude >= self.v_max_usv:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_usv

            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # Update obstacle positions
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # Check for boundary collisions and adjust velocities
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)   
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max_uav,
                vel[1]/self.v_max_uav
            ]
            total_obs.append(S_uavi)
        return total_obs
    
    def get_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            
            # 基础状态信息 (4维)
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max_uav if i < self.num_uavs else vel[0]/self.v_max_usv,
                vel[1]/self.v_max_uav if i < self.num_uavs else vel[1]/self.v_max_usv
            ]
            
            if i < self.num_uavs:  # UAV agents (26维 = 4 + 4 + 16 + 2)
                # UAV观察空间包含更多信息
                S_team = []  # 4维
                S_target = []  # 2维
                for j in range(self.num_agents):
                    if j != i and j != self.num_agents - 1:
                        pos_other = self.multi_current_pos[j]
                        S_team.extend([pos_other[0]/self.length, pos_other[1]/self.length])
                    elif j == self.num_agents - 1:
                        pos_target = self.multi_current_pos[j]
                        d = np.linalg.norm(pos - pos_target)
                        theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                        S_target.extend([d/np.linalg.norm(2*self.length), theta])
                
                S_obser = self.multi_current_lasers[i]  # 16维
                single_obs = [S_uavi, S_team, S_obser, S_target]
                
            elif i < self.num_agents - 1:  # USV agents (20维 = 4 + 16)
                # USV观察空间较简单
                S_obser = self.multi_current_lasers[i]  # 16维
                single_obs = [S_uavi, S_obser]  # 4 + 16 = 20维
                
            else:  # Target agent (20维 = 4 + 16)
                # Target的观察空间也保持20维一致
                S_obser = self.multi_current_lasers[i]  # 16维
                single_obs = [S_uavi, S_obser]  # 4 + 16 = 20维
            
            _single_obs = list(itertools.chain(*single_obs))
            
            # 验证观察空间维度
            expected_dims = 26 if i < self.num_uavs else 20
            assert len(_single_obs) == expected_dims, f"Agent {i} observation dimension mismatch. Expected {expected_dims}, got {len(_single_obs)}"
            
            total_obs.append(_single_obs)
        
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1 = 0.7 # r_near
        mu2 = 0.4 # r_safe
        mu3 = 0.01 # r_multi_stage
        mu4 = 5 # r_finish
        d_capture = 0.3
        d_limit = 0.75
        ## 1 reward for single rounding-up-UAVs:
        for i in range(self.num_uavs):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            r_near = abs(2*v_i/self.v_max_uav)*cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV:
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        # print(rewards[-1])
        # 3.2 stage-1 track
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:self.num_uavs] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:self.num_uavs] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max_uav))
            rewards[0:self.num_uavs] += mu3*r_capture
        
        ## 4 finish rewards
        if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3]):
            rewards[0:self.num_uavs] += mu4*10
            dones = [True] * self.num_agents

        return rewards,dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):

        plt.clf()
        
        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_uavs):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)
            
            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()
        
        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_uavs):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()

class obstacle():
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03 
        speed = 0.00 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)