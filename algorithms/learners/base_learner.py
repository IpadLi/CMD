from typing import Optional, Protocol, Union, Dict, List
import numpy as np

from open_spiel.python.policy import TabularPolicy
from pyspiel import Game

class Learner(Protocol):
    def update(self) -> List[float]:
        """Perform update for policies, increment `iteration` by one"""

    def test_policy(self) -> TabularPolicy:
        """Return test policy"""

    def log_info(self) -> Dict[str, List[Union[float, str]]]:
        """Return relevant learning information"""

    @property
    def game(self) -> Game:
        """Return the game for learning"""

    @property
    def comparator(self) -> Optional[np.ndarray]:
        """Return the policy from which to compute KL divergence"""

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""