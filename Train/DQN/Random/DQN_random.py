## Code for training and testing with PPO, from Stable Baselines, in Pokémon Showdown
##
## ddpg Keras 2018: https://github.com/keras-rl/keras-rl/blob/master/rl/agents/ddpg.py
##
## gamma = 0.75
##
## 300k steps (~30 steps per battle. Up to 10k battles. Divided into 30 epochs)
##

import numpy as np
import asyncio
# # from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.vec_env import VecNormalize
from poke_env.player.env_player import Gen8EnvSinglePlayer, Gen9EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder
from stable_baselines3 import DQN

# from gym.spaces import Box, Discrete
from gymnasium.spaces import Box, Discrete

from distutils.util import strtobool
import neptune.new as neptune

import pandas as pd
import time
import json
import os
from collections import defaultdict
from datetime import date
from itertools import product
from scipy.interpolate import griddata
import argparse
import matplotlib.pyplot as plt



from poke_env.data import GenData

GEN_9_DATA = GenData.from_gen(9)
rewards_a = []
win_a = []


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, debug will be enabled')
    parser.add_argument('--neptune', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, neptune will be enabled')
    parser.add_argument('--train', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, train will be realized')
    parser.add_argument('--saved', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, use saved trained model will be realized')
    parser.add_argument('--test', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, use saved trained model will be realized')
    parser.add_argument('--model-folder', type=str, default="",
                        help='folder of trained model (just for validation)')
    parser.add_argument('--env', type=str, default="stochastic",
                        help='type of environment (stochastic or deterministic). Define teams. OBS: must change showdown too.')

    # Agent parameters
    parser.add_argument('--policy', type=str, default="lin_epsGreedy",
                        help='applied policy')
    parser.add_argument('--hidden', type=int, default=128,
                        help="Hidden layers applied on our nn")
    parser.add_argument('--gamma', type=float, default=0.75,
                        help="gamma value used on ddpg")
    parser.add_argument('--adamlr', type=float, default=0.00025,
                        help="learning rate used on Adam")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help="n epochs")
    parser.add_argument('--battles', type=int, default=10000,
                        help="n steps per epoch")

    args = parser.parse_args()
    return args


np.random.seed(0)


# Definition of PPO player
class RLPlayer(Gen9EnvSinglePlayer):

   
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart

                )

        # We count how many pokemons have not fainted in each team
        n_fainted_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted])
        )
        n_fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted])
        )

        state = np.concatenate([

            [move_base_power for move_base_power in moves_base_power],
            [move_dmg_multiplier for move_dmg_multiplier in moves_dmg_multiplier],
            [n_fainted_mon_team,
             n_fainted_mon_opponent]])

        return state

    # Computing rewards

    def reward_computing_helper(
            self,
            battle: AbstractBattle,
            *,
            fainted_value: float = 0.15,
            hp_value: float = 0.15,
            number_of_pokemons: int = 6,
            starting_value: float = 0.0,
            status_value: float = 0.15,
            victory_value: float = 1.0
    ) -> float:
        # 1st compute
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        # Verify if pokemon have fainted or have status
        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        # Verify if opponent pokemon have fainted or have status
        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        # Verify if we won or lost
        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        # Value to return
        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value
        if args.neptune:
            run[f'{self.mode} reward_buffer'].log(current_value)
        else:
            rewards_a.append(current_value)
        return to_return

    # Calling reward_computing_helper
    def calc_reward(self, last_battle, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)

    def _battle_finished_callback(self, battle):
        self.num_battles += 1
        if args.neptune:
            run[f'{self.mode} win_acc'].log(self.n_won_battles / self.num_battles)
        else:
            win_a.append(self.n_won_battles / self.num_battles)

        self._observations[battle].put(self.embed_battle(battle))

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0,  0]
        high = [ 3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


# Definition of DQN validation player
class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


# Main program
if __name__ == "__main__":

    args = parse_args()

    if args.neptune:
        run = neptune.init(project='jukainite/pokeREL',
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MTM2NzQ3NS0yODQ0LTRmNGItYWRmZi0yNjI1MDRiMDMxYjYifQ==',
                           tags=["DeepRL", args.exp_name, args.env, str(args.epochs) + "epochs"])

    EPOCHS = args.epochs
    NB_TRAINING_EPISODES = args.battles
    NB_TRAINING_STEPS = NB_TRAINING_EPISODES * EPOCHS
    NB_EVALUATION_EPISODES = int(NB_TRAINING_EPISODES / 3)
    N_STATE_COMPONENTS = 12

    # num of features = num of state components + action
    N_FEATURES = N_STATE_COMPONENTS + 1

    N_OUR_MOVE_ACTIONS = 4
    N_OUR_SWITCH_ACTIONS = 5
    N_OUR_ACTIONS = N_OUR_MOVE_ACTIONS + N_OUR_SWITCH_ACTIONS

    ALL_OUR_ACTIONS = np.array(range(0, N_OUR_ACTIONS))

    second_opponent = RandomPlayer(battle_format="gen9randombattle")
    opponent = MaxDamagePlayer(battle_format="gen9randombattle")
    env_player = RLPlayer(battle_format="gen9randombattle", opponent=opponent)
    # if args.train:
    #     env_player = RLPlayer(battle_format="gen9randombattle", team=OUR_TEAM, mode = "train")
    # else: env_player = RLPlayer(battle_format="gen9randombattle", team=OUR_TEAM, mode = "val")
    model = DQN(MlpPolicy, env=env_player, learning_rate=args.adamlr, gamma=args.gamma, verbose=1)

    if args.train:
        # env_player._opponent = opponent
        try:
            env_player.reset_battles()
        except:
            pass
        print("Training...")
        model.learn(total_timesteps=NB_TRAINING_STEPS)
        obs, reward, done, _, info = env_player.step(0)
        while not done:
            while not done:
                try:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env_player.step(action)
                except:
                    done = True
        print("Training complete.")
        print(
            "DQN Evaluation: %d victories out of %d episodes"
            % (env_player.n_won_battles, NB_TRAINING_STEPS)
        )
        model.save("model_%d" % NB_TRAINING_STEPS)
    else:
        model.load("model_%d" % NB_TRAINING_STEPS)
    if args.test:

        # Evaluate against different opponents after training or loading the model
        env_player.mode = "val_max"
        print("Results against max player:")
        env_player.num_battles = 0
        # env_player.play_against(env_algorithm=dqn_evaluating, opponent=opponent)
        env_player._opponent = opponent
        try:
            env_player.reset_battles()
        except:
            pass
        # env_player.reset_battles()
        for _ in range(NB_EVALUATION_EPISODES):
            done = False
            obs, info = env_player.reset()
            while not done:
                try:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env_player.step(action)
                except:
                    done = True
        # env_player.complete_current_battle()


        print(
            "DQN Evaluation: %d victories out of %d episodes"
            % (env_player.n_won_battles, NB_EVALUATION_EPISODES)
        )

        try:
            env_player.reset_battles()
        except:
            pass

        env_player.mode = "val_rand"
        print("\nResults against random player:")
        env_player.num_battles = 0
        env_player._opponent = second_opponent
        # env_player.reset_battles()
        for _ in range(NB_EVALUATION_EPISODES):
            done = False
            obs, info = env_player.reset()
            while not done:
                try:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env_player.step(action)
                except:
                    done = True


        print(
            "DQN Evaluation: %d victories out of %d episodes"
            % (env_player.n_won_battles, NB_EVALUATION_EPISODES)
        )






