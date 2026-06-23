#!/usr/bin/env python

# Copyright (c) 2018-2022 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Hard break scenario:

The scenario spawn a vehicle in front of the ego that drives for a while before
suddenly hard breaking, forcing the ego to avoid the collision
"""

import py_trees
import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      SwitchWrongDirectionTest,
                                                                      Idle,
                                                                      BasicAgentBehavior,
                                                                      HandBrakeVehicle,
                                                                      ConstantVelocityAgentBehavior, 
                                                                      ScenarioTimeout,
                                                                      WaitForever)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest, OvertakeTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToVehicle,
                                                                               WaitUntilInFront,
                                                                               WaitUntilInFrontPosition)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, ChangeOppositeBehavior, ChangeRoadBehavior, SetMaxSpeed

def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

def get_interval_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]['from']),
            p_type(config.other_parameters[name]['to'])
        ]
    else:
        return default

class CleanLane(BasicScenario):

    """
    In this class, the ego's lane is clean, no other vehicles disturbing.

    This is a single ego vehicle scenario
    """

    timeout = 600          # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True,
                 timeout=600):
        """
        Setup all relevant parameters and create scenario
        """
        self.timeout = timeout
        self._scenario_timeout = 240
        self._stop_duration = 10
        # self.end_distance = 15
        self._trigger_distance = 30
        self._end_distance = 50
        self._wait_duration = 0 # 5
        self._map = CarlaDataProvider.get_map()
        print(f"[debug] CleanLane Scenario Activated.")

        super().__init__("CleanLane",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

        
    def _initialize_actors(self, config):
        """
        Spawn a front vehicle for overtaking
        """
        rng = CarlaDataProvider.get_random_seed()
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
    
    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or (next_wps[0].is_junction and dist > 60.0):
            # if not next_wps: # when junction, still go
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence(name="CleanLane")
        if self.route_mode:
            total_dist = 999
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=30))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._starting_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(ChangeRoadBehavior(extra_space=0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root


    def _create_test_criteria(self):
        """
        The route already has a collision criteria + Overtake
        """
        criterias = []
        if not self.route_mode:
            criterias.append(CollisionTest(self.ego_vehicles[0]))
        return criterias

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()