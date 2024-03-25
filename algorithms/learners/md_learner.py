import copy
from collections import defaultdict
from math import prod
from typing import Union, Dict, List
import time
import numpy as np

from open_spiel.python.policy import TabularPolicy
from pyspiel import Game

from .base_learner import Learner
from ..evaluation import Evaluator
from configs import coop_games, single_agent_games, mix_coop_comp_games
from ..tree import Node
from ..utils import project, project_energy_to_positive, kl

EPS = 1e-6
SMALL_POSITIVE = 1e-10

class CMD(Learner):
    def __init__(self, args, game, temp, lr, mag_lr, objective):
        self.args = args
        self._game = game
        self.temp = temp
        self.lr = lr
        self.mag_lr = mag_lr
        self.objective = objective
        self._iteration = 0
        self.policy: TabularPolicy = TabularPolicy(game)
        self.magnet: TabularPolicy = TabularPolicy(game)
        self.evaluator = Evaluator(args, game)

        self.single_agent_game_list = list(single_agent_games.keys())
        if args.game in self.single_agent_game_list:
            # for single agent cases, will only update the policy of the focal agent, other agents use uniform policy
            self.focal_agent_id = single_agent_games[args.game]
            self.focal_agent_inforstates = self.policy.states_per_player[self.focal_agent_id]
        self.mcc_game_list = list(mix_coop_comp_games.keys())

        sample_n = args.sample_n
        if args.alpha_optim in ["drs", "rs"]:
            sample_n *= 2
        self.policy_list = [TabularPolicy(game) for _ in range(sample_n)]
        max_k = args.max_k + 1 if args.add_magnet else args.max_k
        self.alphas_list = np.array([np.array([1.0 / max_k] * max_k) for _ in range(sample_n)])
        self.alpha_u_list = []
        self.history_policy = [copy.deepcopy(self.policy)]
        self.curr_k = len(self.history_policy)
        self.alpha_k = np.array([1.0])
        self.init_policy = copy.deepcopy(self.policy)
        self.prev_policy = copy.deepcopy(self.policy)

        self.ite_time_pl = 0
        self.ite_time_mt = 0

    def __build_tree(self, policy):
        infostate_values = defaultdict(list)
        reach_probs = defaultdict(list)
        queue = [
            Node(
                self.game.new_initial_state(),
                policy,
                {i: 1 for i in range(-1, self.game.num_players())},
                self.temp(self.iteration),
                self.objective,
                self.args.game,
            )
        ]
        while len(queue) > 0:
            node = queue.pop()
            if node.history.is_terminal():
                continue
            if not node.history.is_chance_node():
                cur_player = node.history.current_player()
                infostate = node.history.information_state_string(cur_player)
                acting_qs = {a: q[cur_player] for a, q in node.action_values.items()}
                infostate_values[infostate].append(acting_qs)
                reach_conts = node.reach_contributions.values()
                reach_probs[infostate].append(prod(list(reach_conts)))
            queue += list(node.children.values())
        return infostate_values, reach_probs

    def __get_policy(self, infostate: str, magnet: bool = False) -> np.ndarray:
        if magnet:
            pol = self.magnet
        else:
            pol = self.policy
        return pol.action_probability_array[pol.state_lookup[infostate]]

    def __set_policy(self, infostate: str, new_policy: Union[dict, np.ndarray], magnet: bool = False, idx: int=-1) -> None:
        if idx == -1:
            index = self.policy.state_lookup[infostate]
            if isinstance(new_policy, dict):
                pol = np.zeros_like(self.policy.action_probability_array[index])
                for a, p in new_policy.items():
                    pol[a] = p
            elif isinstance(new_policy, np.ndarray):
                pol = new_policy
            else:
                raise ValueError
            if magnet:
                self.magnet.action_probability_array[index] = pol
            else:
                self.policy.action_probability_array[index] = pol
        else:
            index = self.policy_list[idx].state_lookup[infostate]
            if isinstance(new_policy, dict):
                pol = np.zeros_like(self.policy_list[idx].action_probability_array[index])
                for a, p in new_policy.items():
                    pol[a] = p
            elif isinstance(new_policy, np.ndarray):
                pol = new_policy
            else:
                raise ValueError
            self.policy_list[idx].action_probability_array[index] = pol

    def __update_infostate(self, infostate: str, values: List[Dict[int, float]], reach_probs: List[float]) -> None:
        expected_values: Dict[int, float] = defaultdict(float)
        mu = sum(reach_probs)
        for action_values, p in zip(values, reach_probs):
            for a, q in action_values.items():
                expected_values[a] += q * p / mu
        lr = self.lr(self.iteration)
        temp = self.temp(self.iteration)
        old_pol = self.__get_policy(infostate)
        mag_pol = self.__get_policy(infostate, magnet=True)

        if self.args.comb == "kl":
            # kl induced policy
            energy = {
                a: (np.log(old_pol[a])
                    + lr * temp * np.log(mag_pol[a])
                    + lr * expected_values[a])
                   / (1 + lr * temp)
                for a in expected_values.keys()
            }
            max_energy = max(energy.values())
            temp_policy = {a: np.exp(e - max_energy) for a, e in energy.items()}
            prop_policy = project(temp_policy)
        elif self.args.comb == "eu":
            # euclidean induced policy
            num_actions = len(expected_values.keys())
            eu_induced_policy = {
                a: (temp * mag_pol[a]
                    + 1 / lr * old_pol[a]
                    + expected_values[a]
                    - sum(expected_values.values()) / num_actions)
                   / (temp + 1 / lr)
                for a in expected_values.keys()
            }
            prop_policy = project(project_energy_to_positive(eu_induced_policy))
        else:
            raise ValueError(f"Unknown type of regularizer {self.args.comb}.")

        # update old policy and mag policy
        self.__set_policy(infostate, prop_policy, False)
        mag_lr = self.mag_lr(self.iteration)
        if self.args.comb == "kl":
            prop_mag = np.power(mag_pol, 1 - mag_lr) * np.power(self.__get_policy(infostate), mag_lr)
        elif self.args.comb == "eu":
            prop_mag = (1 - mag_lr) * mag_pol + mag_lr * self.__get_policy(infostate)
        self.__set_policy(infostate, project(prop_mag), magnet=True)

    def __update_infostate_gmd(self, infostate: str, values: List[Dict[int, float]], reach_probs: List[float], idx: int=-1) -> None:
        expected_values: Dict[int, float] = defaultdict(float)
        mu = sum(reach_probs)
        for action_values, p in zip(values, reach_probs):
            for a, q in action_values.items():
                expected_values[a] += q * p / mu
        index = self.policy.state_lookup[infostate]
        alpha_k = copy.deepcopy(self.alpha_k) if idx == -1 else copy.deepcopy(self.alphas_list[idx])
        temp = 1
        if self.args.add_magnet:
            if self.args.alpha_sc == "mtctl" and self.iteration <= self.args.opt_inter:
                temp = self.temp(self.iteration)
            else:
                temp = copy.deepcopy(alpha_k[0])
                alpha_k = alpha_k[1:]
        B = sum(alpha_k)

        if self.args.phi == 1:  # x^n, n > 1
            n = self.args.n_for_x_n
            policy = self.history_policy[0].action_probability_array[index]
            pol = alpha_k[0] * n * np.power(policy, n - 1)
            if self.curr_k > 1:
                for i, p in enumerate(self.history_policy[1:]):
                    policy = p.action_probability_array[index]
                    pol += alpha_k[i + 1] * n * np.power(policy, n - 1)
            if self.args.add_magnet:
                policy = self.magnet.action_probability_array[index]
                pol += temp * n * np.power(policy, n - 1)
                B += temp
            A = {a: expected_values[a] + pol[a] for a in expected_values.keys()}
            A_values = np.array(list(A.values()))
            assert n == 2
            init_lam = (sum(A_values) - n * B) / len(expected_values.keys()) + 0.01
            lam = copy.deepcopy(init_lam)
            for _ in range(self.args.lam_ite):
                glam = np.sum((A_values - lam) / (n * B)) - 1
                glam_prime = -len(expected_values.keys()) / B / 2
                lam -= glam / glam_prime
            energy = {
                a: max([np.power((A[a] - lam) / (n * B), n - 1), SMALL_POSITIVE])
                for a in expected_values.keys()
            }
        elif self.args.phi == 2:    # -x^n, 0 <= n <= 1
            n = self.args.n_f
            policy = self.history_policy[0].action_probability_array[index]
            policy = np.maximum(policy, SMALL_POSITIVE)
            pol = -alpha_k[0] * n * np.power(policy, n - 1)
            if self.curr_k > 1:
                for i, p in enumerate(self.history_policy[1:]):
                    policy = p.action_probability_array[index]
                    policy = np.maximum(policy, SMALL_POSITIVE)
                    pol += -alpha_k[i + 1] * n * np.power(policy, n - 1)
            if self.args.add_magnet:
                policy = self.magnet.action_probability_array[index]
                policy = np.maximum(policy, SMALL_POSITIVE)
                pol += temp * n * np.power(policy, n - 1)
                B += temp
            A = {a: expected_values[a] + pol[a] for a in expected_values.keys()}
            A_values = np.array(list(A.values()))
            init_lam = max(A_values) + 0.1
            lam = copy.deepcopy(init_lam)
            valid_lam = copy.deepcopy(init_lam)
            for _ in range(self.args.lam_ite):
                glam = np.sum(np.power((lam - A_values) / (n * B), 1 / (n - 1))) - 1
                glam_prime = np.sum(1 / (B *(n - 1)) * np.power((lam - A_values) / (n * B), (2 - n) / (n - 1)))
                lam -= glam / glam_prime
                if lam < max(A_values):
                    lam = copy.deepcopy(valid_lam)
                else:
                    valid_lam = copy.deepcopy(lam)
            energy = {
                a: max([np.power((lam - A[a]) / (n * B), n - 1), SMALL_POSITIVE])
                for a in expected_values.keys()
            }
        elif self.args.phi == 3: # x ln x (entropy)
            policy = self.history_policy[0].action_probability_array[index]
            policy = np.maximum(policy, SMALL_POSITIVE)
            pol = alpha_k[0] * (np.log(policy) + 1)
            if self.curr_k > 1:
                for i, p in enumerate(self.history_policy[1:]):
                    policy = p.action_probability_array[index]
                    policy = np.maximum(policy, SMALL_POSITIVE)
                    pol += alpha_k[i + 1] * (np.log(policy) + 1)
            if self.args.add_magnet:
                policy = self.magnet.action_probability_array[index]
                policy = np.maximum(policy, SMALL_POSITIVE)
                pol += temp * (np.log(policy) + 1)
                B += temp
            A = {a: expected_values[a] + pol[a] for a in expected_values.keys()}
            A_values = np.array(list(A.values()))
            init_lam = np.random.uniform(max(A_values) - B, max(A_values))
            lam = copy.deepcopy(init_lam)
            valid_lam = copy.deepcopy(init_lam)
            for _ in range(self.args.lam_ite):
                glam = np.sum(np.exp((A_values - lam) / B - 1)) - 1
                glam_prime = np.sum(-np.exp((A_values - lam) / B - 1) / B)
                lam -= glam / glam_prime
                if lam >= max(A_values):
                    lam = copy.deepcopy(valid_lam)
                else:
                    valid_lam = copy.deepcopy(lam)
            energy = {
                a: max([np.exp((A[a] - lam) / B - 1), SMALL_POSITIVE])
                for a in expected_values.keys()
            }
        elif self.args.phi == 4:    # e^{kx}
            k = self.args.k_for_e_k
            policy = self.history_policy[0].action_probability_array[index]
            pol = alpha_k[0] * k * np.exp(k * policy)
            if self.curr_k > 1:
                for i, p in enumerate(self.history_policy[1:]):
                    policy = p.action_probability_array[index]
                    pol += alpha_k[i + 1] * k * np.exp(k * policy)
            if self.args.add_magnet:
                policy = self.magnet.action_probability_array[index]
                pol += temp * k * np.exp(k * policy)
                B += temp
            A = {a: expected_values[a] + pol[a] for a in expected_values.keys()}
            A_values = np.array(list(A.values()))
            init_lam = min(A_values) - 0.1
            lam = copy.deepcopy(init_lam)
            valid_lam = copy.deepcopy(init_lam)
            for _ in range(self.args.lam_ite):
                glam = np.sum(np.log((A_values - lam) / B / k) / k) - 1
                glam_prime = np.sum(1 / (lam - A_values))
                lam -= glam / glam_prime
                if lam >= min(A_values):
                    lam = copy.deepcopy(valid_lam)
                else:
                    valid_lam = copy.deepcopy(lam)
            energy = {
                a: max([np.log((A[a] - lam) / B / k) / k, SMALL_POSITIVE])
                for a in expected_values.keys()
            }
        elif self.args.phi == 5:    # -ln x
            policy = self.history_policy[0].action_probability_array[index]
            policy = np.maximum(policy, SMALL_POSITIVE)
            pol = -alpha_k[0] / policy
            if self.curr_k > 1:
                for i, p in enumerate(self.history_policy[1:]):
                    policy = p.action_probability_array[index]
                    policy = np.maximum(policy, SMALL_POSITIVE)
                    pol += -alpha_k[i + 1] / policy
            if self.args.add_magnet:
                policy = self.magnet.action_probability_array[index]
                policy = np.maximum(policy, SMALL_POSITIVE)
                pol += -temp / policy
                B += temp
            A = {a: expected_values[a] + pol[a] for a in expected_values.keys()}
            A_values = np.array(list(A.values()))
            init_lam = max(A_values) + 0.1
            lam = copy.deepcopy(init_lam)
            valid_lam = copy.deepcopy(init_lam)
            for _ in range(self.args.lam_ite):
                glam = np.sum(B / (lam - A_values)) - 1
                glam_prime = np.sum(-B / (A_values - lam)**2)
                lam -= glam / glam_prime
                if len(np.where(lam == A_values)[0]) > 0:
                    lam = copy.deepcopy(valid_lam)
                else:
                    valid_lam = copy.deepcopy(lam)
            energy = {
                a: max([B / (lam - A[a]), SMALL_POSITIVE])
                for a in expected_values.keys()
            }
        else:
            raise ValueError('Convex function is not specified.')

        gmd_policy = project(energy)
        self.__set_policy(infostate, gmd_policy, False, idx)
        if self.args.add_magnet:
            if idx == -1:
                mag_pol = self.magnet.action_probability_array[index]
                mag_lr = self.mag_lr(self.iteration)
                if self.args.phi == 1 or self.args.phi == 2:
                    prop_mag = (1 - mag_lr) * mag_pol + mag_lr * self.__get_policy(infostate)
                elif self.args.phi == 3 or self.args.phi == 4:
                    prop_mag = np.power(mag_pol, 1 - mag_lr) * np.power(self.__get_policy(infostate), mag_lr)
                self.__set_policy(infostate, project(prop_mag), magnet=True)

    def __perturb_alpha(self, idx):
        sz = self.alphas_list[idx].size
        if self.args.alpha_optim == "gld":
            radii = (self.args.R - self.args.r) * np.random.rand(1) + self.args.r
            u = np.random.randn(sz)  # a data point from normal dist N(0, 1)
            u = radii * u / np.linalg.norm(u)  # the data point on the sphere with sampled radius
            self.alphas_list[idx] += u
            self.alphas_list[idx] = np.clip(self.alphas_list[idx], EPS, 1)
        elif self.args.alpha_optim in ["dglds", "glds"]:
            radii = (self.args.R - self.args.r) * np.random.rand(1) + self.args.r
            u = np.random.randn(sz)  # a data point from normal dist N(0, 1)
            u = radii * u / np.linalg.norm(u)  # the data point on the sphere with sampled radius
            self.alpha_u_list.append(u)
            self.alphas_list[idx] += u
            self.alphas_list[idx] = np.clip(self.alphas_list[idx], EPS, 1)
        elif self.args.alpha_optim in ["drs", "rs"]:
            if idx < self.args.sample_n:
                u = np.random.randn(sz)  # a data point from normal dist N(0, 1)
                u = self.args.mu * u / np.linalg.norm(u)  # the data point on the sphere with fixed radius
                self.alpha_u_list.append(u)
                self.alphas_list[idx] += u
            else:
                # cur_val - \mu * u
                self.alphas_list[idx] -= self.alpha_u_list[idx - self.args.sample_n]
            self.alphas_list[idx] = np.clip(self.alphas_list[idx], EPS, 1)

    def __update_alpha(self, alpha_optim, L, base_L=None):
        if alpha_optim == "dglds":
            us = []
            for idx in range(self.args.sample_n):
                diff = L[idx] - base_L
                if diff < 0.:
                    us.append(-self.alpha_u_list[idx])
                elif diff > 0.:
                    us.append(self.alpha_u_list[idx])
                else:
                    us.append(0.)
            u_star = copy.deepcopy(us[0])
            for idx in range(1, self.args.sample_n):
                u_star += us[idx]
            self.alpha_k -= u_star
            self.alpha_k = np.clip(self.alpha_k, EPS, 1)
        elif alpha_optim == "glds":
            us = [(L[idx] - base_L) * self.alpha_u_list[idx] for idx in range(self.args.sample_n)]
            u_star = copy.deepcopy(us[0])
            for idx in range(1, self.args.sample_n):
                u_star += us[idx]
            u_star /= (-self.args.sample_n)
            self.alpha_k -= u_star
            self.alpha_k = np.clip(self.alpha_k, EPS, 1)
        elif alpha_optim == "drs":
            us = []
            for idx in range(self.args.sample_n):
                diff = L[idx] - L[idx + self.args.sample_n]
                if diff < 0.:
                    us.append(-self.alpha_u_list[idx])
                elif diff > 0.:
                    us.append(self.alpha_u_list[idx])
                else:
                    us.append(0.)
            u_star = copy.deepcopy(us[0])
            for idx in range(1, self.args.sample_n):
                u_star += us[idx]
            self.alpha_k -= u_star
            self.alpha_k = np.clip(self.alpha_k, EPS, 1)
        elif alpha_optim == "rs":
            us = [(L[idx] - L[idx + self.args.sample_n]) * self.alpha_u_list[idx] for idx in range(self.args.sample_n)]
            u_star = copy.deepcopy(us[0])
            for idx in range(1, self.args.sample_n):
                u_star += us[idx]
            u_star *= - self.args.xi / (2 * self.args.mu * self.args.sample_n)
            self.alpha_k -= u_star
            self.alpha_k = np.clip(self.alpha_k, EPS, 1)

    def update(self):
        """Perform update for policies, increment `iteration` by one"""
        self._iteration += 1
        infostate_values, infostate_reach_probs = self.__build_tree(self.policy)
        self.prev_policy = copy.deepcopy(self.policy)

        if self.args.comb == "gmd":
            max_k = self.args.max_k
            if self.args.add_magnet:
                max_k += 1
            if self.args.alpha_sc == "mtctl":
                if self._iteration <= max_k:
                    self.alpha_k = np.array([1.0 / max_k] * max_k)
                else:
                    if self.iteration % self.args.opt_inter == 0:
                        start_t = time.time()
                        sample_n = self.args.sample_n
                        if self.args.alpha_optim in ["drs", "rs"]:
                            sample_n *= 2
                        L = []
                        for idx in range(sample_n):
                            self.__perturb_alpha(idx)
                            if self.args.game in self.single_agent_game_list:
                                for infostate in infostate_values.keys():
                                    if infostate in self.focal_agent_inforstates:
                                        # only update the policy of the focal agent
                                        self.__update_infostate_gmd(infostate, infostate_values[infostate], infostate_reach_probs[infostate], idx)
                            else:
                                for infostate in infostate_values.keys():
                                    self.__update_infostate_gmd(infostate, infostate_values[infostate], infostate_reach_probs[infostate], idx)
                            if self.args.game in coop_games + self.single_agent_game_list:
                                obj = -self.evaluator.eval(self.policy_list[idx])
                            else:
                                if self.args.alpha_optim_obj == "negap":
                                    obj = self.evaluator.eval(self.policy_list[idx])
                                elif self.args.alpha_optim_obj == "ccegap":
                                    obj = self.evaluator.eval_ccegap(self.policy_list[idx])
                                elif self.args.alpha_optim_obj == "sw":
                                    obj = -self.evaluator.eval_sw(self.policy_list[idx])
                            L.append(obj)
                        if self.args.alpha_optim == "gld":
                            if self.args.game in coop_games + self.single_agent_game_list:
                                base_L = -self.evaluator.eval(self.test_policy())
                            else:
                                if self.args.alpha_optim_obj == "negap":
                                    base_L = self.evaluator.eval(self.test_policy())
                                elif self.args.alpha_optim_obj == "ccegap":
                                    base_L = self.evaluator.eval_ccegap(self.test_policy())
                                elif self.args.alpha_optim_obj == "sw":
                                    base_L = -self.evaluator.eval_sw(self.test_policy())
                            step_to_new_para = False
                            sel_idx = np.argmin(L)
                            if min(L) < base_L:
                                step_to_new_para = True
                            if step_to_new_para:
                                self.alpha_k = copy.deepcopy(self.alphas_list[sel_idx])
                                for idx in range(sample_n):
                                    if idx != sel_idx:
                                        self.alphas_list[idx] = copy.deepcopy(self.alphas_list[sel_idx])
                            else:
                                for idx in range(sample_n):
                                    self.alphas_list[idx] = copy.deepcopy(self.alpha_k)
                        else:
                            if self.args.alpha_optim in ["dglds", "glds"]:
                                if self.args.game in coop_games + self.single_agent_game_list:
                                    base_L = -self.evaluator.eval(self.test_policy())
                                else:
                                    if self.args.alpha_optim_obj == "negap":
                                        base_L = self.evaluator.eval(self.test_policy())
                                    elif self.args.alpha_optim_obj == "ccegap":
                                        base_L = self.evaluator.eval_ccegap(self.test_policy())
                                    elif self.args.alpha_optim_obj == "sw":
                                        base_L = -self.evaluator.eval_sw(self.test_policy())
                            else:
                                base_L = None
                            self.__update_alpha(self.args.alpha_optim, L, base_L)
                            del self.alpha_u_list[:]
                            for idx in range(sample_n):
                                self.alphas_list[idx] = copy.deepcopy(self.alpha_k)
                        self.ite_time_mt = time.time() - start_t
            else:
                if self.args.alpha_sc == "uniform":
                    self.alpha_k = np.array([1.0 / max_k] * max_k)
                elif self.args.alpha_sc == "linear_decay":
                    decayed_alpha = 1.0 - (1 - 0.01) * self._iteration / self.args.num_updates
                    self.alpha_k = np.array([decayed_alpha] * max_k)
                elif self.args.alpha_sc == "inv_sqrt_root":
                    decayed_alpha = 1.0 / np.sqrt(self._iteration)
                    self.alpha_k = np.array([decayed_alpha] * max_k)

        p_update_start = time.time()
        if self.args.game in self.single_agent_game_list:
            for infostate in infostate_values.keys():
                if infostate in self.focal_agent_inforstates:
                    # only update the policy of the focal agent
                    if self.args.comb == "gmd":
                        self.__update_infostate_gmd(infostate, infostate_values[infostate], infostate_reach_probs[infostate])
                    else:
                        self.__update_infostate(infostate, infostate_values[infostate], infostate_reach_probs[infostate])
        else:
            for infostate in infostate_values.keys():
                if self.args.comb == "gmd":
                    self.__update_infostate_gmd(infostate, infostate_values[infostate], infostate_reach_probs[infostate])
                else:
                    self.__update_infostate(infostate, infostate_values[infostate], infostate_reach_probs[infostate])
        if self.args.comb == "gmd":
            self.history_policy.append(copy.deepcopy(self.policy))
            if len(self.history_policy) > self.args.max_k:
                self.history_policy = self.history_policy[1:]
            self.curr_k = len(self.history_policy)
        self.ite_time_pl = time.time() - p_update_start

    def log_info(self) -> Dict[str, List[Union[float, str]]]:
        """Return relevant learning information"""
        exploit = self.evaluator.eval(self.test_policy())
        if self.args.game in (self.single_agent_game_list + coop_games):
            social_welfare = exploit
        else:
            social_welfare = self.evaluator.eval_sw(self.test_policy())
        if self.args.game in (self.single_agent_game_list + coop_games + self.mcc_game_list):
            ccegap = 0
        else:
            ccegap = self.evaluator.eval_ccegap(self.test_policy())
        return {
            "Iteration": [self.iteration],
            "Exploitability": [exploit],
            "Social_Welfare": [social_welfare],
            "CCE_Gap": [ccegap],
            "Temperature": [self.temp(self.iteration)],
            "Stepsize": [self.lr(self.iteration)],
            "Magnet Stepsize": [self.mag_lr(self.iteration)],
            "Objective": [self.objective.value],
            "alpha_k": [copy.deepcopy(self.alpha_k)],
            "kl": [kl(self.test_policy().action_probability_array, self.prev_policy.action_probability_array)],
        }

    def test_policy(self) -> TabularPolicy:
        """Return test policy"""
        return self.policy

    @property
    def game(self) -> Game:
        """Return the game for learning"""
        return self._game

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""
        return self._iteration



