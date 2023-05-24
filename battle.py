import numpy as np
from math import cos, sin, tan, pi, atan2, acos
from random import random, uniform, randint
from gym import spaces


# 4DoF UAV model
class UAVModel(object):
    def __init__(self, args):
        super(UAVModel, self).__init__()
        # 无人机的标识
        self.id = None
        self.enemy = None
        self.size = 0.02
        self.color = None
        # 无人机的状态
        self.pos = np.zeros(2)
        self.speed = 0  # 速度标量
        self.vel = None  # 速度矢量
        self.yaw = 0  # 偏航角，0为x轴正方向，逆时针为正，(-pi, pi)
        self.roll = 0  # 横滚角, 顺时针为正
        # 无人机飞行约束
        self.speed_max = args.speed_max
        self.speed_min = args.speed_min
        self.roll_max = args.roll_max
        self.roll_min = args.roll_min
        # 无人机对抗相关
        self.being_attacked = False
        self.death = False
        # 无人机的感知范围和攻击范围
        self.detect_range = args.detect_range
        self.attack_range = args.attack_range
        self.attack_angle = args.attack_angle


class Battle(object):
    """为避免神经网络输入数值过大，采用等比例缩小模型"""

    def __init__(self, args):
        super(Battle, self).__init__()
        self.args = args
        self.dt = 1  # simulation interval，1 second
        self.t = 0
        self.render_geoms = None
        self.render_geoms_xform = None
        self.num_UAVs = args.num_RUAVs + args.num_BUAVs
        self.num_RUAVs = args.num_RUAVs
        self.num_BUAVs = args.num_BUAVs
        self.UAVs = [UAVModel(args) for _ in range(self.num_UAVs)]
        self.RUAVs = []
        self.BUAVs = []
        for i, UAV in enumerate(self.UAVs):
            UAV.id = i
            if i < args.num_RUAVs:
                UAV.enemy = False
                UAV.color = np.array([1, 0, 0])
                self.RUAVs.append(UAV)
            else:
                UAV.enemy = True
                UAV.color = np.array([0, 0, 1])
                self.BUAVs.append(UAV)
        self.viewer = None
        self.action_space = []
        self.reset()
        self.r_action_spaces = [spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32) for _ in
                                range(self.num_RUAVs)]
        self.b_action_spaces = [spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32) for _ in
                                range(self.num_BUAVs)]
        r_obs_n, b_obs_n = self.get_obs()
        self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             r_obs_n]
        self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             b_obs_n]

    def reset(self):
        self.t = 0
        # reset render
        self.render_geoms = None
        self.render_geoms_xform = None
        random_side = randint(0, 1)
        for i, UAV in enumerate(self.UAVs):
            UAV.being_attacked = False
            UAV.death = False
            if not UAV.enemy:
                interval = 2.0 / (self.num_RUAVs + 1)
                UAV.pos = np.array([random_side*1.8-0.9, 1 - (i+1)*interval])
                UAV.speed = random() * (UAV.speed_max - UAV.speed_min) + UAV.speed_min
                UAV.yaw = pi * random_side
                UAV.vel = UAV.speed * np.array([cos(UAV.yaw), sin(UAV.yaw)])
                UAV.roll = random() * (UAV.roll_max - UAV.roll_min) + UAV.roll_min
            else:
                interval = 2.0 / (self.num_BUAVs + 1)
                UAV.pos = np.array([(1 - random_side) * 1.8 - 0.9, 1 - (i - self.num_RUAVs + 1) * interval])
                UAV.speed = random() * (UAV.speed_max - UAV.speed_min) + UAV.speed_min
                UAV.yaw = pi * (1-random_side)
                UAV.vel = UAV.speed * np.array([cos(UAV.yaw), sin(UAV.yaw)])
                UAV.roll = random() * (UAV.roll_max - UAV.roll_min) + UAV.roll_min

    def step(self, r_actions, b_actions):
        acceleration = 0.01     # m/s2
        omega = pi / 12         # rad/s
        actions = r_actions + b_actions
        for UAV, action in zip(self.UAVs, actions):
            if UAV.death: continue
            UAV.speed += action[0] * acceleration * self.dt
            UAV.speed = np.clip(UAV.speed, UAV.speed_min, UAV.speed_max)
            UAV.roll += action[1] * omega * self.dt
            UAV.roll = np.clip(UAV.roll, UAV.roll_min, UAV.roll_max)
            UAV.yaw -= 0.004 / UAV.speed * tan(UAV.roll) * self.dt
            UAV.yaw = atan2(sin(UAV.yaw), cos(UAV.yaw))     # map yaw to [-pi, +pi]
            UAV.vel = UAV.speed * np.array([cos(UAV.yaw), sin(UAV.yaw)])
            UAV.pos += UAV.vel * self.dt
            UAV.pos = UAV.pos.clip(-0.98, 0.98)
        r_obs_n, b_obs_n = self.get_obs()
        r_reward_n, b_reward_n = self.get_reward()
        done = self.get_done()
        return r_obs_n, b_obs_n, r_reward_n, b_reward_n, done

    def get_obs(self):
        r_obs_n = []
        b_obs_n = []
        for i in range(self.num_RUAVs):
            own_feat = np.zeros((7,), dtype=np.float32)
            ally_feats = np.zeros((self.num_RUAVs - 1, 5), dtype=np.float32)
            adv_feats = np.zeros((self.num_BUAVs, 6), dtype=np.float32)
            ally_alive_mask = np.zeros((self.num_RUAVs - 1,), dtype=np.float32)
            adv_alive_mask = np.zeros((self.num_BUAVs,), dtype=np.float32)
            own = self.RUAVs[i]
            own_feat[0:2] = own.pos
            own_feat[2:4] = own.vel
            own_feat[4] = own.speed
            own_feat[5] = own.yaw
            own_feat[6] = own.roll
            ally_idx = 0
            for ally in self.RUAVs:
                if ally is own: continue
                if not ally.death:
                    relative_pos = ally.pos - own.pos
                    distance = np.linalg.norm(relative_pos)
                    ally_feats[ally_idx, 0:2] = relative_pos
                    ally_feats[ally_idx, 2] = distance
                    ally_feats[ally_idx, 3] = ally.yaw - own.yaw
                    ally_feats[ally_idx, 4] = ally.roll
                    ally_alive_mask[ally_idx] = 1
                else:
                    ally_alive_mask[ally_idx] = 0
                ally_idx += 1
            for adv_idx, adv in enumerate(self.BUAVs):
                if not adv.death:
                    relative_pos = adv.pos - own.pos
                    distance = np.linalg.norm(relative_pos)
                    adv_feats[adv_idx, 0:2] = relative_pos
                    adv_feats[adv_idx, 2] = distance
                    adv_feats[adv_idx, 3] = atan2(relative_pos[1], relative_pos[0])
                    adv_feats[adv_idx, 4] = adv.yaw - own.yaw
                    adv_feats[adv_idx, 5] = adv.roll
                    adv_alive_mask[adv_idx] = 1
                else:
                    adv_alive_mask[adv_idx] = 0
            r_obs = np.concatenate([
                own_feat.flatten(),
                ally_feats.flatten(),
                adv_feats.flatten(),
                ally_alive_mask.flatten(),
                adv_alive_mask.flatten()
            ])
            r_obs_n.append(r_obs)
        for i in range(self.num_BUAVs):
            own_feat = np.zeros((7,), dtype=np.float32)
            ally_feats = np.zeros((self.num_BUAVs - 1, 5), dtype=np.float32)
            adv_feats = np.zeros((self.num_RUAVs, 6), dtype=np.float32)
            ally_alive_mask = np.zeros((self.num_BUAVs - 1,), dtype=np.float32)
            adv_alive_mask = np.zeros((self.num_RUAVs,), dtype=np.float32)
            own = self.BUAVs[i]
            own_feat[0:2] = own.pos
            own_feat[2:4] = own.vel
            own_feat[4] = own.speed
            own_feat[5] = own.yaw
            own_feat[6] = own.roll
            ally_idx = 0
            for ally in self.BUAVs:
                if ally is own: continue
                if not ally.death:
                    relative_pos = ally.pos - own.pos
                    distance = np.linalg.norm(relative_pos)
                    ally_feats[ally_idx, 0:2] = relative_pos
                    ally_feats[ally_idx, 2] = distance
                    ally_feats[ally_idx, 3] = ally.yaw - own.yaw
                    ally_feats[ally_idx, 4] = ally.roll
                    ally_alive_mask[ally_idx] = 1
                else:
                    ally_alive_mask[ally_idx] = 0
                ally_idx += 1
            for adv_idx, adv in enumerate(self.RUAVs):
                if not adv.death:
                    relative_pos = adv.pos - own.pos
                    distance = np.linalg.norm(relative_pos)
                    adv_feats[adv_idx, 0:2] = relative_pos
                    adv_feats[adv_idx, 2] = distance
                    adv_feats[adv_idx, 3] = atan2(relative_pos[1], relative_pos[0])
                    adv_feats[adv_idx, 4] = adv.yaw - own.yaw
                    adv_feats[adv_idx, 5] = adv.roll
                    adv_alive_mask[adv_idx] = 1
                else:
                    adv_alive_mask[adv_idx] = 0
            b_obs = np.concatenate([
                own_feat.flatten(),
                ally_feats.flatten(),
                adv_feats.flatten(),
                ally_alive_mask.flatten(),
                adv_alive_mask.flatten()
            ])
            b_obs_n.append(b_obs)
        return np.stack(r_obs_n), np.stack(b_obs_n)

    def get_reward(self):
        hit_prob = 0.6
        r_launch_attack = [False] * self.num_RUAVs
        b_launch_attack = [False] * self.num_BUAVs
        r_reward_n = np.zeros((self.num_RUAVs,), dtype=np.float32)
        b_reward_n = np.zeros((self.num_BUAVs,), dtype=np.float32)
        for adv_idx, BUAV in enumerate(self.BUAVs):
            if BUAV.death: continue
            for ally_idx, RUAV in enumerate(self.RUAVs):
                if RUAV.death: continue
                # Blue UAV's reward
                relative_pos = RUAV.pos - BUAV.pos
                distance = np.linalg.norm(relative_pos)
                attack_angle = acos(np.clip(relative_pos.dot(BUAV.vel) / (BUAV.speed * distance), -1, 1))
                if attack_angle < BUAV.attack_angle and distance < BUAV.attack_range:
                    # 敌方在这一时刻未被其它队友击中，并且此无人机未在此时刻发起攻击
                    if (not RUAV.being_attacked) and (not b_launch_attack[adv_idx]):
                        hit = random() < hit_prob
                        RUAV.being_attacked = hit
                        if hit:
                            b_reward_n[adv_idx] += 5.0  # killing an enemy
                        else:
                            b_reward_n[adv_idx] += 0.2  # attacking an enemy
                        r_reward_n[ally_idx] -= 0.1  # being attacked or killed
                    b_launch_attack[adv_idx] = True
                # Red UAV's reward
                relative_pos = -relative_pos
                attack_angle = acos(np.clip(relative_pos.dot(RUAV.vel) / (RUAV.speed * distance), -1, 1))
                if attack_angle < RUAV.attack_angle and distance < RUAV.attack_range:
                    if (not BUAV.being_attacked) and (not r_launch_attack[ally_idx]):
                        hit = random() < hit_prob
                        BUAV.being_attacked = hit
                        if hit:
                            r_reward_n[ally_idx] += 5.0
                        else:
                            r_reward_n[ally_idx] += 0.2
                        b_reward_n[adv_idx] -= 0.1
                    r_launch_attack[ally_idx] = True
                # collision
                if distance < RUAV.size + BUAV.size + 0.02:
                    b_reward_n[adv_idx] -= 0.1
                    r_reward_n[ally_idx] -= 0.1
        # -0.005 for every move
        r_reward_n -= 0.005
        b_reward_n -= 0.005
        self.clear_death()
        return r_reward_n, b_reward_n

    def get_done(self):
        r_alive = [UAV.being_attacked for UAV in self.RUAVs]
        b_alive = [UAV.being_attacked for UAV in self.BUAVs]
        if all(r_alive) or all(b_alive):
            return True
        return False

    def clear_death(self):
        for UAV in self.UAVs:
            UAV.death = UAV.being_attacked

    def render(self, mode='rgb_array'):
        import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(900, 900)
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            for UAV in self.UAVs:
                xform = rendering.Transform()
                for x in rendering.make_UAV(UAV.size):
                    x.set_color(*UAV.color)
                    x.add_attr(xform)
                    self.render_geoms.append(x)
                    self.render_geoms_xform.append(xform)
                # render detect range
                # circle = rendering.make_circle(radius=0.5*UAV.detect_range, filled=False)
                # circle.set_color(*UAV.color, 0.2)
                # circle.add_attr(xform)
                # self.render_geoms.append(circle)
                # self.render_geoms_xform.append(xform)
                # render attack range
                sector = rendering.make_sector(radius=UAV.attack_range, theta=2*UAV.attack_angle)
                sector.set_color(*UAV.color, 0.2)
                sector.add_attr(xform)
                self.render_geoms.append(sector)
                self.render_geoms_xform.append(xform)
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)
        results = []
        self.viewer.set_bounds(-1, +1, -1, +1)
        for i, UAV in enumerate(self.UAVs):
            idx_ratio = len(self.render_geoms) // self.num_UAVs
            for idx in range(idx_ratio):
                if UAV.death:
                    self.render_geoms_xform[idx_ratio * i + idx].set_translation(-10, -10)
                else:
                    self.render_geoms_xform[idx_ratio * i + idx].set_translation(*UAV.pos)
                    self.render_geoms_xform[idx_ratio * i + idx].set_rotation(UAV.yaw)
        results.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))
        return results

    def close(self):
        pass
