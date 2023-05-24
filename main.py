import argparse
import time
from battle import Battle
from math import pi


def get_args():
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=1200, help="maximum episode length")
    parser.add_argument("--num-RUAVs", type=int, default=6, help="number of red UAVs")
    parser.add_argument("--num-BUAVs", type=int, default=6, help="number of blue UAVs")
    parser.add_argument("--speed-max", type=float, default=0.01, help="")
    parser.add_argument("--speed-min", type=float, default=0.005, help="")
    parser.add_argument("--roll-max", type=float, default=+pi / 3, help="")
    parser.add_argument("--roll-min", type=float, default=-pi / 3, help="")
    parser.add_argument("--detect-range", type=float, default=0.5, help="")
    parser.add_argument("--attack-range", type=float, default=0.15, help="")
    parser.add_argument("--attack-angle", type=float, default=pi / 8, help="")
    parser.add_argument("--render", action="store_true", default=True, help="whether to render the env")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    env = Battle(args)
    env.reset()
    r_obs_spaces, b_obs_spaces = env.r_obs_spaces, env.b_obs_spaces
    r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
    for i in range(args.max_episode_len):
        if args.render:
            time.sleep(0.1)
            env.render()
        r_obs_n, b_obs_n = env.get_obs()
        r_action_n = [[0, -1] for i in range(args.num_RUAVs)]
        b_action_n = [b_action_spaces[i].sample() for i in range(args.num_BUAVs)]
        r_reward_n, b_reward_n, r_reward_n, b_reward_n, done = env.step(r_action_n, b_action_n)


if __name__ == "__main__":
    main()
