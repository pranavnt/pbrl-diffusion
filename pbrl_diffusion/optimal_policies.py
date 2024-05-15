from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class OptimalDrawerClose(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_grasp_info": obs[3],
            "drwr_pos": obs[4:7],
            "unused_info": obs[7:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = 1.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_drwr = o_d["drwr_pos"] + np.array([0.0, 0.0, -0.02])

        # if further forward than the drawer...
        if pos_curr[1] > pos_drwr[1]:
            if pos_curr[2] < pos_drwr[2] + 0.23:
                # rise up quickly (Z direction)
                return np.array([pos_curr[0], pos_curr[1], pos_drwr[2] + 0.5])
            else:
                # move to front edge of drawer handle, but stay high in Z
                return pos_drwr + np.array([0.0, -0.075, 0.23])
        # drop down to touch drawer handle
        elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
            return pos_drwr + np.array([0.0, -0.075, 0.0])
        # push toward drawer handle's centroid
        else:
            return pos_drwr

class OptimalDrawerOpen(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "drwr_pos": obs[4:7],
            "unused_info": obs[7:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # NOTE this policy looks different from the others because it must
        # modify its p constant part-way through the task
        pos_curr = o_d["hand_pos"]
        pos_drwr = o_d["drwr_pos"] + np.array([0.0, 0.0, -0.02])

        # align end effector's Z axis with drawer handle's Z axis
        if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
            to_pos = pos_drwr + np.array([0.0, 0.0, 0.3])
            action["delta_pos"] = move(o_d["hand_pos"], to_pos, p=4.0)
        # drop down to touch drawer handle
        elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
            to_pos = pos_drwr
            action["delta_pos"] = move(o_d["hand_pos"], to_pos, p=4.0)
        # push toward a point just behind the drawer handle
        # also increase p value to apply more force
        else:
            to_pos = pos_drwr + np.array([0.0, -0.06, 0.0])
            action["delta_pos"] = move(o_d["hand_pos"], to_pos, p=50.0)

        # keep gripper open
        action["grab_effort"] = -1.0

        return action.array