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

class OvertakeRoute(BasicScenario):

    """
    In this class the ego vehicle must decide wether overtake or follow the leading vehicle.

    This is a single ego vehicle scenario
    """

    timeout = 120            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self.timeout = timeout
        self._scenario_timeout = 240
        self._stop_duration = 10
        # self.end_distance = 15
        self.front_vehicle_speed = get_value_parameter(config, 'speed', float, 5)
        self.front_vehicle_distance = get_value_parameter(config, 'distance', float, 30)
        self._behavior = get_value_parameter(config, 'behavior', str, 'overtake')
        if self._behavior not in ('follow', 'overtake'):
            raise ValueError(f"'behavior' must be either 'follow' or 'overtake' but {self._behavior} was given")
        self._trigger_distance = 30
        self._end_distance = 50
        self._wait_duration = 0 # 5
        self._map = CarlaDataProvider.get_map()
        self._is_two_ways = False
        print(f"[debug] OvertakeRoute, self.front_vehicle_speed = {self.front_vehicle_speed}, self.front_vehicle_distance = {self.front_vehicle_distance}, self._behavior = {self._behavior}")

        super().__init__("Overtake",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

        
    def _initialize_actors(self, config):
        """
        Spawn a front vehicle for overtaking
        """
        # randomize
        if self.rand_flag:
            self.front_vehicle_speed = self.front_vehicle_speed * (1.0 + self.spd_shift)
            self.front_vehicle_distance = self.front_vehicle_distance + self.pos_shift
        
        rng = CarlaDataProvider.get_random_seed()
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        spawn_distance = self.front_vehicle_distance    
        front_wp = self._move_waypoint_forward(self._starting_wp, spawn_distance)
        if not front_wp:
            raise RuntimeError("Cannot find waypoint to spawn front vehicle")
        front_vehicle = self._spawn_obstacle(front_wp, "vehicle.*")
        if front_vehicle is None:
            raise RuntimeError("Failed to spawn front vehicle")

        # front_vehicle.apply_control(carla.VehicleControl(speed=self.front_vehicle_speed))
        self.other_actors.append(front_vehicle)

        self._end_wp = self._move_waypoint_forward(front_wp, self._end_distance)
        
        # debug!
        from webcolors import (
            CSS2_HEX_TO_NAMES,
            hex_to_rgb,
        )
        from scipy.spatial import KDTree
        
        def convert_rgb_to_names(rgb_tuple):
            # a dictionary of all the hex and their respective names in css3
            css3_db = CSS2_HEX_TO_NAMES
            names = []
            rgb_values = []
            for color_hex, color_name in css3_db.items():
                names.append(color_name)
                rgb_values.append(hex_to_rgb(color_hex))
            
            kdt_db = KDTree(rgb_values)
            distance, index = kdt_db.query(rgb_tuple)
            return f'{names[index]}'
        
        npc_id = str(front_vehicle.id)
        location = front_vehicle.get_transform().location
        rotation = front_vehicle.get_transform().rotation
        vehicle_brake = front_vehicle.get_control().brake
        vehicle_throttle = front_vehicle.get_control().throttle
        vehicle_steer = front_vehicle.get_control().steer
        rgb = tuple(map(int, front_vehicle.attributes['color'].split(',')))
        color_name = convert_rgb_to_names(rgb)
        result = {
            'class': 'vehicle',
            'state': 'dynamic',
            'id': npc_id,
            'location': [location.x, location.y, location.z],
            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
            'type_id': front_vehicle.type_id,
            'color': color_name,
            'base_type': front_vehicle.attributes['base_type'],
            'brake': vehicle_brake,
            'throttle': vehicle_throttle,
            'steer': vehicle_steer,
        }
        print(f"[debug] front vehicle's info: {result}")
        
        # add actors that are relevant for the Expert to CarlaDataProvider.active_scenarios
        side_lane_wp = front_wp.get_left_lane()
        scenario_name = 'OvertakeRoute' if side_lane_wp.lane_id * front_wp.lane_id > 0 else 'OvertakeRouteTwoWays'
        self._is_two_ways = side_lane_wp.lane_id * front_wp.lane_id > 0
        CarlaDataProvider.active_scenarios.append((scenario_name, [front_vehicle, None, False, 1e9, 1e9, False, self.front_vehicle_speed]))
    
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

    def _spawn_obstacle(self, wp, blueprint):
        """
        Spawns the obstacle front actor
        """
        spawn_transform = wp.transform
        actor = CarlaDataProvider.request_new_actor(blueprint, spawn_transform, attribute_filter={'base_type': 'car', 'generation': 2})
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")
        self.pass_scenario_actor(actor, spawn_transform) # for vqa gen

        return actor

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence(name="Overtake")
        if self.route_mode:
            total_dist = self.front_vehicle_distance + 999
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=30))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        # if overtake and distance > 1 car space (factor = 3.0), the front vehicle vanishes.
        end_condition.add_child(WaitUntilInFront(self.ego_vehicles[0], self.other_actors[-1], factor=3.0, check_distance=False))

        front_behavior = py_trees.composites.Sequence(name="Front behavior")
        target_loc = self._move_waypoint_forward(self._starting_wp, 9999.0).transform.location
        front_behavior.add_child(BasicAgentBehavior(self.other_actors[0], target_loc, target_speed=self.front_vehicle_speed, opt_dict=None))
        front_behavior.add_child(WaitForever())
        # vanishes at the first junction
        end_condition.add_child(front_behavior)

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._starting_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            # if self._is_two_ways:
            behavior.add_child(SwitchWrongDirectionTest(False))
            # behavior.add_child(SetMaxSpeed(self.front_vehicle_speed))
            pass
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
            root.add_child(ChangeRoadBehavior(extra_space=0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root


    def _create_test_criteria(self):
        """
        The route already has a collision criteria + Overtake
        """
        criterias = []
        criterias.append(OvertakeTest(self.ego_vehicles[0], self.other_actors[0], reverse=(self._behavior == "follow")))
        if not self.route_mode:
            criterias.append(CollisionTest(self.ego_vehicles[0]))
        return criterias

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

class OvertakeRouteTwoWays(OvertakeRoute):
    """
    Variation of the OvertakeRoute scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        print("[debug] initializing OvertakeRouteTwoWays")
        self._opposite_frequency = get_value_parameter(config, 'frequency', float, 100)

        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence(name="OvertakeTwoWays")
        if self.route_mode:
            total_dist = self.front_vehicle_distance + 999
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=30))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        # if overtake and distance > 1 car space (factor = 3.0), the front vehicle vanishes.
        end_condition.add_child(WaitUntilInFront(self.ego_vehicles[0], self.other_actors[-1], factor=3.0, check_distance=False))

        front_behavior = py_trees.composites.Sequence(name="Front behavior")
        target_loc = self._move_waypoint_forward(self._starting_wp, 9999.0).transform.location
        front_behavior.add_child(BasicAgentBehavior(self.other_actors[0], target_loc, target_speed=self.front_vehicle_speed, opt_dict=None))
        front_behavior.add_child(WaitForever())
        # vanishes at the first junction
        end_condition.add_child(front_behavior)

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._starting_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
            root.add_child(ChangeOppositeBehavior(spawn_dist=40))
            root.add_child(ChangeRoadBehavior(extra_space=0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root