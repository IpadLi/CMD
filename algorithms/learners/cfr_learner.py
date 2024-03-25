import copy
import time
import numpy as np
from typing import Optional, Union, Dict, List

from open_spiel.python.algorithms.cfr import _CFRSolver
from pyspiel import Game

from ..evaluation import Evaluator
from configs import single_agent_games
from ..utils import kl
from .cfr_learner_single_agent import _CFRSolverSingleAgent
from .base_learner import Learner

class CFR(Learner):
    def __init__(self, game, use_plus, args):
        self._game = game
        self.single_agent_game_list = list(single_agent_games.keys())
        if args.game in self.single_agent_game_list:
            # for single agent cases, will only update the policy of the focal agent, other agents use uniform policy
            self.focal_agent_id = single_agent_games[args.game]
            self.solver = _CFRSolverSingleAgent(
                game,
                self.focal_agent_id,
                regret_matching_plus=use_plus,
                alternating_updates=use_plus,
                linear_averaging=use_plus,
            )
        else:
            self.solver = _CFRSolver(
                game,
                regret_matching_plus=use_plus,
                alternating_updates=use_plus,
                linear_averaging=use_plus,
            )
        self._comparator = None
        self._iteration = 0
        self.args = args
        self.evaluator = Evaluator(args, game)
        self.prev_policy = self.solver.average_policy()
        self.ite_time_pl = 0
        self.ite_time_mt = 0

    def update(self):
        """Perform update for policies, increment `iteration` by one"""
        p_update_start = time.time()
        self.prev_policy = copy.deepcopy(self.solver.average_policy())
        self.solver.evaluate_and_update_policy()
        self._iteration += 1
        self.ite_time_pl = time.time() - p_update_start

    def test_policy(self):
        """Return test policy"""
        return self.solver.average_policy()

    def log_info(self) -> Dict[str, List[Union[float, str]]]:
        """Return relevant learning information"""
        return {
            "Iteration": [self.iteration],
            "Exploitability": [self.evaluator.eval(self.test_policy())],
            "Social_Welfare": [self.evaluator.eval_sw(self.test_policy())],
            "CCE_Gap": [self.evaluator.eval_ccegap(self.test_policy())],
            "kl": [kl(self.test_policy().action_probability_array, self.prev_policy.action_probability_array)],
        }

    @property
    def game(self) -> Game:
        """Return the game for learning"""
        return self._game

    @property
    def comparator(self) -> Optional[np.ndarray]:
        """Return the policy from which to compute KL divergence"""
        return self._comparator

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""
        return self._iteration