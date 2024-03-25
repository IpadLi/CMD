import collections
import numpy as np
import time
import pickle

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from open_spiel.python import rl_environment
from open_spiel.python.policy import TabularPolicy
from pyspiel import Game

from ...algorithms.evaluation import _state_values
from ...configs import single_agent_games
from ...algorithms.utils import schedule

def valuedict():
    return collections.defaultdict(float)

class QLearner(rl_agent.AbstractAgent):
    def __init__(self,
                 args,
                 game,
                 fn,
                 step_size=0.1,
                 epsilon_schedule=rl_tools.ConstantSchedule(0.1),
                 discount_factor=1.0,
                 centralized=False):
        self.env = rl_environment.Environment(game)
        self._num_actions = self.env.action_spec()["num_actions"]

        self._step_size = step_size
        self._epsilon_schedule = epsilon_schedule
        self._epsilon = epsilon_schedule.value
        self._discount_factor = discount_factor
        self._centralized = centralized
        self._q_values = collections.defaultdict(valuedict)
        self._prev_info_state = None
        self._last_loss_value = None
        self._prev_action = None

        self.args = args
        self._game = game
        self._fn = fn
        self._iteration = 0
        self.policy: TabularPolicy = TabularPolicy(game)
        self.single_agent_game_list = list(single_agent_games.keys())
        # for single agent cases, will only update the policy of the focal agent, other agents use uniform policy
        self._player_id = single_agent_games[args.game]
        self.focal_player_inforstates = self.policy.states_per_player[self._player_id]

    def q_to_policy(self, info_state, legal_actions, epsilon=0.0):
        probs = np.zeros(self._num_actions)
        greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
        greedy_actions = [a for a in legal_actions if self._q_values[info_state][a] == greedy_q]
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)


    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        probs = np.zeros(self._num_actions)
        greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
        greedy_actions = [a for a in legal_actions if self._q_values[info_state][a] == greedy_q]
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs

    def _get_action_probs(self, info_state, legal_actions, epsilon):
        return self._epsilon_greedy(info_state, legal_actions, epsilon)

    def step(self, time_step, is_evaluation=False):
        if self._centralized:
            info_state = str(time_step.observations["info_state"])
        else:
            info_state = str(time_step.observations["info_state"][self._player_id])
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None

        # Act step: don't act at terminal states.
        if not time_step.last():
            epsilon = 0.0 if is_evaluation else self._epsilon
            action, probs = self._get_action_probs(info_state, legal_actions, epsilon)

        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            target = time_step.rewards[self._player_id]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max([self._q_values[info_state][a] for a in legal_actions])
            prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
            self._last_loss_value = target - prev_q_value
            self._q_values[self._prev_info_state][self._prev_action] += (self._step_size * self._last_loss_value)

            # Decay epsilon, if necessary.
            self._epsilon = self._epsilon_schedule.step()
            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action

        return rl_agent.StepOutput(action=action, probs=probs)

    def update(self):
        ret = []
        ite = []
        for i, should_save in schedule(self.args.num_updates, self.args.power_sc):
            self._iteration += 1
            start_update = time.time()

            time_step = self.env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == self._player_id:
                    agent_output = self.step(time_step)
                    act = agent_output.action
                else:
                    infostate = self.env.get_state.information_state_string()
                    actions = time_step.observations["legal_actions"][player_id]
                    index = self.policy.state_lookup[infostate]
                    probs = self.policy.action_probability_array[index][np.array(actions)]
                    act = np.random.choice(actions, p=probs)
                time_step = self.env.step([act])
            # Episode is over, step all agents with final info state.
            self.step(time_step)

            end = time.time()
            time_update = end - start_update

            if should_save:
                root_state = self._game.new_initial_state()
                on_policy_values = _state_values(root_state, self._game.num_players(), self.policy, self.args.game)
                mean_eps_rew = on_policy_values[self._player_id]
                time_eval = time.time() - end
                ret.append(mean_eps_rew)
                ite.append(self._iteration)
                pickle.dump(
                    {"ret": ret, "ite": ite},
                    open(self._fn + ".pik", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                print(
                    "QL, Env {}, N {}, Iter {}/{}, Ret {:.10f}, Time {:.5f}[Train {:.5f}, Eval {:.5f}], Seed {}".format(
                        self.args.game,
                        self.args.num_agents,
                        i,
                        self.args.num_updates,
                        mean_eps_rew,
                        time.time() - start_update,
                        time_update,
                        time_eval,
                        self.args.seed,
                    )
                )

    @property
    def loss(self):
        return self._last_loss_value

    @property
    def game(self) -> Game:
        """Return the game for learning"""
        return self._game

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""
        return self._iteration