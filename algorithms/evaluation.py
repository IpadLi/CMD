import numpy as np
import itertools
import collections
from collections import defaultdict
from math import prod

from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_lib
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms.exploitability import exploitability, nash_conv
import pyspiel

from .utils import convert_returns, project
from .tree import Node
from configs import (
    coop_games,
    mix_coop_comp_games,
    single_agent_games,
    hyperparameters,
    objective_choices
)

def _memoize_method(key_fn=lambda x: x):
    def memoizer(method):
        cache_name = "cache_" + method.__name__

        def wrap(self, arg):
            key = key_fn(arg)
            cache = vars(self).setdefault(cache_name, {})
            if key not in cache:
                cache[key] = method(self, arg)
            return cache[key]

        return wrap

    return memoizer


class BestResponsePolicy(openspiel_policy.Policy):
    def __init__(
        self, game, player_id, policy, game_str, root_state=None, cut_threshold=0.0
    ):
        self._num_players = game.num_players()
        self._player_id = player_id
        self._policy = policy
        if root_state is None:
            root_state = game.new_initial_state()
        self.game_str = game_str
        self._root_state = root_state
        self.infosets = self.info_sets(root_state)
        self._cut_threshold = cut_threshold

    def info_sets(self, state):
        infosets = collections.defaultdict(list)
        for s, p in self.decision_nodes(state):
            infosets[s.information_state_string(self._player_id)].append((s, p))
        return dict(infosets)

    def decision_nodes(self, parent_state):
        if not parent_state.is_terminal():
            if (
                parent_state.current_player() == self._player_id
                or parent_state.is_simultaneous_node()
            ):
                yield (parent_state, 1.0)
            for action, p_action in self.transitions(parent_state):
                for state, p_state in self.decision_nodes(
                    openspiel_policy.child(parent_state, action)
                ):
                    yield (state, p_state * p_action)

    def joint_action_probabilities_counterfactual(self, state):
        (
            actions_per_player,
            probs_per_player,
        ) = openspiel_policy.joint_action_probabilities_aux(state, self._policy)
        probs_per_player[self._player_id] = [
            1.0 for _ in probs_per_player[self._player_id]
        ]
        return [
            (list(actions), np.prod(probs))
            for actions, probs in zip(
                itertools.product(*actions_per_player),
                itertools.product(*probs_per_player),
            )
        ]

    def transitions(self, state):
        if state.current_player() == self._player_id:
            return [(action, 1.0) for action in state.legal_actions()]
        elif state.is_chance_node():
            return state.chance_outcomes()
        elif state.is_simultaneous_node():
            return self.joint_action_probabilities_counterfactual(state)
        else:
            return list(self._policy.action_probabilities(state).items())

    @_memoize_method(key_fn=lambda state: state.history_str())
    def value(self, state):
        if state.is_terminal():
            return convert_returns(state.returns(), self.game_str)[self._player_id]
        elif state.current_player() == self._player_id or state.is_simultaneous_node():
            action = self.best_response_action(
                state.information_state_string(self._player_id)
            )
            return self.q_value(state, action)
        else:
            return sum(
                p * self.q_value(state, a)
                for a, p in self.transitions(state)
                if p > self._cut_threshold
            )

    def q_value(self, state, action):
        if state.is_simultaneous_node():

            def q_value_sim(sim_state, sim_actions):
                child = sim_state.clone()
                sim_actions[self._player_id] = action
                child.apply_actions(sim_actions)
                return self.value(child)

            actions, probabilities = zip(*self.transitions(state))
            return sum(
                p * q_value_sim(state, a)
                for a, p in zip(actions, probabilities / sum(probabilities))
                if p > self._cut_threshold
            )
        else:
            return self.value(state.child(action))

    @_memoize_method()
    def best_response_action(self, infostate):
        infoset = self.infosets[infostate]
        return max(
            infoset[0][0].legal_actions(self._player_id),
            key=lambda a: sum(cf_p * self.q_value(s, a) for s, cf_p in infoset),
        )

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            if state.is_simultaneous_node():
                player_id = self._player_id
            else:
                player_id = state.current_player()
        return {self.best_response_action(state.information_state_string(player_id)): 1}


def _state_values(state, num_players, policy, game_str):
    if state.is_terminal():
        return convert_returns(state.returns(), game_str)
    else:
        if state.is_simultaneous_node():
            p_action = tuple(policy_lib.joint_action_probabilities(state, policy))
        else:
            p_action = (
                state.chance_outcomes()
                if state.is_chance_node()
                else policy.action_probabilities(state).items()
            )
        return sum(
            prob
            * _state_values(
                policy_lib.child(state, action), num_players, policy, game_str
            )
            for action, prob in p_action
        )

def nash_conv_for_mcc_games(game, policy, args):
    root_state = game.new_initial_state()
    best_response_values = np.array(
        [
            BestResponsePolicy(game, best_responder, policy, args.game).value(root_state)
            for best_responder in range(game.num_players())
        ]
    )
    on_policy_values = _state_values(root_state, game.num_players(), policy, args.game)
    player_improvements = best_response_values - on_policy_values
    nash_conv_ = sum(player_improvements)
    return nash_conv_

def build_tree(state, num_players, policy, objective, temp, game_str):
    infostate_values = defaultdict(list)
    reach_probs = defaultdict(list)
    queue = [
        Node(
            state,
            policy,
            {i: 1 for i in range(-1, num_players)},
            temp,
            objective,
            game_str,
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

def nash_conv_for_mcc_games_using_mmd(game, policy, args):

    temp_schedule = hyperparameters[args.game]["temp_schedule"]
    lr_schedule = hyperparameters[args.game]["lr_schedule"]
    mag_lr_schedule = hyperparameters[args.game]["mag_lr_schedule"]
    objective = objective_choices[args.objective]

    root_state = game.new_initial_state()
    num_players = game.num_players()

    on_policy_values = _state_values(root_state, num_players, policy, args.game)

    teams = mix_coop_comp_games[args.game]
    nash_conv_ = 0
    for team in teams:
        if len(team) == 1:
            player_id = team[0]
            br_value = BestResponsePolicy(game, player_id, policy, args.game).value(root_state)
            nash_conv_ += max(0, br_value - on_policy_values[player_id])
        else:
            # using MMD to get br policy of the team
            br_policy = policy.__copy__()   # other agents will keep current policy
            br_magnet = policy.__copy__()
            team_inforstates = []
            for pid in team:
                team_inforstates += policy.states_per_player[pid]
            for ite in range(100):
                lr = lr_schedule(ite)
                temp = temp_schedule(ite)
                mag_lr = mag_lr_schedule(ite)
                infostate_values, infostate_reach_probs = build_tree(root_state, num_players, br_policy, objective, temp, args.game)
                for infostate in infostate_values.keys():
                    if infostate in team_inforstates:
                        # only update in the inforstates of the team
                        values, reach_probs = infostate_values[infostate], infostate_reach_probs[infostate]
                        expected_values = defaultdict(float)
                        mu = sum(reach_probs)
                        for action_values, p in zip(values, reach_probs):
                            for a, q in action_values.items():
                                expected_values[a] += q * p / mu
                        index = br_policy.state_lookup[infostate]
                        old_pol = br_policy.action_probability_array[index]
                        mag_pol = br_magnet.action_probability_array[index]
                        energy = {
                            a: (np.log(old_pol[a])
                                + lr * temp * np.log(mag_pol[a])
                                + lr * expected_values[a])
                               / (1 + lr * temp)
                            for a in expected_values.keys()
                        }
                        max_energy = max(energy.values())
                        temp_policy = {a: np.exp(e - max_energy) for a, e in energy.items()}
                        new_policy = project(temp_policy)
                        pol = np.zeros_like(br_policy.action_probability_array[index])
                        for a, p in new_policy.items():
                            pol[a] = p
                        br_policy.action_probability_array[index] = pol
                        br_magnet.action_probability_array[index] = project(np.power(mag_pol, 1 - mag_lr) * np.power(pol, mag_lr))
            br_value = _state_values(root_state, num_players, br_policy, args.game)
            nash_conv_ += max(0, br_value[team[0]] - on_policy_values[team[0]])
    return nash_conv_

def nc(game, policy, args):
    return nash_conv(game, policy)

def exploit(game, policy, args):
    return exploitability(game, policy)

GAP_TOL = 1e-8

def ccegap(game, policy, args):
    pyspiel_tabular_policy = policy_lib.python_policy_to_pyspiel_policy(policy)
    joint_returns = pyspiel.expected_returns(game.new_initial_state(), pyspiel_tabular_policy, -1, True)
    num_players = game.num_players()
    gap = 0
    for player in range(num_players):
        info = pyspiel.cce_dist(
            game,
            [(1.0, pyspiel_tabular_policy)],
            player,
            prob_cut_threshold=0.0,
            action_value_tolerance=-1)
        deviation_incentive = max(info.best_response_values[0] - joint_returns[player], 0)
        if deviation_incentive < GAP_TOL:
            deviation_incentive = 0.0
        gap += deviation_incentive
    return gap

def ccegap_for_mcc_games_using_mmd(game, policy, args):

    temp_schedule = hyperparameters[args.game]["temp_schedule"]
    lr_schedule = hyperparameters[args.game]["lr_schedule"]
    mag_lr_schedule = hyperparameters[args.game]["mag_lr_schedule"]
    objective = objective_choices[args.objective]

    root_state = game.new_initial_state()
    num_players = game.num_players()

    pyspiel_tabular_policy = policy_lib.python_policy_to_pyspiel_policy(policy)
    on_policy_values = pyspiel.expected_returns(root_state, pyspiel_tabular_policy, -1, True)

    teams = mix_coop_comp_games[args.game]
    gap = 0
    for team in teams:
        if len(team) == 1:
            player = team[0]
            info = pyspiel.cce_dist(
                game,
                [(1.0, pyspiel_tabular_policy)],
                player,
                prob_cut_threshold=0.0,
                action_value_tolerance=-1)
            deviation_incentive = max(info.best_response_values[0] - on_policy_values[player], 0)
            if deviation_incentive < GAP_TOL:
                deviation_incentive = 0.0
            gap += deviation_incentive
        else:
            # using MMD to get br policy of the team
            br_policy = policy.__copy__()   # other agents will keep current policy
            br_magnet = policy.__copy__()
            team_inforstates = []
            for pid in team:
                team_inforstates += policy.states_per_player[pid]
            for ite in range(100):
                lr = lr_schedule(ite)
                temp = temp_schedule(ite)
                mag_lr = mag_lr_schedule(ite)
                infostate_values, infostate_reach_probs = build_tree(root_state, num_players, br_policy, objective, temp, args.game)
                for infostate in infostate_values.keys():
                    if infostate in team_inforstates:
                        # only update in the inforstates of the team
                        values, reach_probs = infostate_values[infostate], infostate_reach_probs[infostate]
                        expected_values = defaultdict(float)
                        mu = sum(reach_probs)
                        for action_values, p in zip(values, reach_probs):
                            for a, q in action_values.items():
                                expected_values[a] += q * p / mu
                        index = br_policy.state_lookup[infostate]
                        old_pol = br_policy.action_probability_array[index]
                        mag_pol = br_magnet.action_probability_array[index]
                        energy = {
                            a: (np.log(old_pol[a])
                                + lr * temp * np.log(mag_pol[a])
                                + lr * expected_values[a])
                               / (1 + lr * temp)
                            for a in expected_values.keys()
                        }
                        max_energy = max(energy.values())
                        temp_policy = {a: np.exp(e - max_energy) for a, e in energy.items()}
                        new_policy = project(temp_policy)
                        pol = np.zeros_like(br_policy.action_probability_array[index])
                        for a, p in new_policy.items():
                            pol[a] = p
                        br_policy.action_probability_array[index] = pol
                        br_magnet.action_probability_array[index] = project(np.power(mag_pol, 1 - mag_lr) * np.power(pol, mag_lr))
            br_value = _state_values(root_state, num_players, br_policy, args.game)
            gap += max(0, br_value[team[0]] - on_policy_values[team[0]])
    return gap

def single_agent_eval_exact(game, policy, args):
    focal_agent_id = single_agent_games[args.game]
    root_state = game.new_initial_state()
    on_policy_values = _state_values(root_state, game.num_players(), policy, args.game)
    return on_policy_values[focal_agent_id]

def single_agent_eval(game, policy, args):
    focal_agent_id = single_agent_games[args.game]
    env = rl_environment.Environment(game)
    episode_rews = []
    for _ in range(500):
        time_step = env.reset()
        while not time_step.last():
            p = time_step.observations["current_player"]
            infostate = env.get_state.information_state_string()
            actions = time_step.observations["legal_actions"][p]
            index = policy.state_lookup[infostate]
            probs = policy.action_probability_array[index][np.array(actions)]
            act = np.random.choice(actions, p=probs)
            time_step = env.step([act])
        episode_rews.append(time_step.rewards[focal_agent_id])
    return np.mean(episode_rews)

def coop_eval(game, policy, args):
    env = rl_environment.Environment(game)
    episode_rews = []
    for _ in range(500):
        time_step = env.reset()
        while not time_step.last():
            p = time_step.observations["current_player"]
            infostate = env.get_state.information_state_string()
            actions = time_step.observations["legal_actions"][p]
            index = policy.state_lookup[infostate]
            probs = policy.action_probability_array[index][np.array(actions)]
            act = np.random.choice(actions, p=probs)
            time_step = env.step([act])
        episode_rews.append(time_step.rewards[0])
    return np.mean(episode_rews)

def coop_eval_exact(game, policy, args):
    root_state = game.new_initial_state()
    on_policy_values = _state_values(root_state, game.num_players(), policy, args.game)
    return on_policy_values[0]

def social_welfare(game, policy, args):
    env = rl_environment.Environment(game)
    episode_rew = 0
    num_eps = 100
    for _ in range(num_eps):
        time_step = env.reset()
        while not time_step.last():
            p = time_step.observations["current_player"]
            infostate = env.get_state.information_state_string()
            actions = time_step.observations["legal_actions"][p]
            index = policy.state_lookup[infostate]
            probs = policy.action_probability_array[index][np.array(actions)]
            act = np.random.choice(actions, p=probs)
            time_step = env.step([act])
        episode_rew += sum(time_step.rewards)
    return episode_rew / num_eps


class Evaluator:
    def __init__(self, args, game):
        self.args = args
        self.game = game

        game_info = game.get_type()
        if args.game in list(single_agent_games.keys()):
            self.evaluator = single_agent_eval_exact
        elif args.game in coop_games:
            self.evaluator = coop_eval_exact
        elif args.game in mix_coop_comp_games.keys():
            self.evaluator = nash_conv_for_mcc_games_using_mmd
        else:
            if args.num_agents > 2 or game_info.utility in (
                    pyspiel.GameType.Utility.GENERAL_SUM,
                    pyspiel.GameType.Utility.IDENTICAL,
            ):
                self.evaluator = nc
            else:
                self.evaluator = exploit

    def eval(self, policy):
        return self.evaluator(self.game, policy, self.args)

    def eval_sw(self, policy):
        return social_welfare(self.game, policy, self.args)

    def eval_ccegap(self, policy):
        if self.args.game in mix_coop_comp_games.keys():
            return ccegap_for_mcc_games_using_mmd(self.game, policy, self.args)
        else:
            return ccegap(self.game, policy, self.args)