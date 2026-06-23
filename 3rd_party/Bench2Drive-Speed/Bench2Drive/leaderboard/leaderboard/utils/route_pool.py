"""
File-based shared route pool for multi-GPU evaluation.

Instead of statically sharding routes across ranks, all ranks share a single
pool of routes. Each rank acquires a file lock, picks the next available route
(preferring one with the same town to avoid expensive world reloads), marks it
as taken, and releases the lock.
"""

import fcntl
import json
import os
import time

import copy
from collections import OrderedDict

from leaderboard.utils.route_parser import RouteParser


class RoutePool:
    """A shared, file-lock-coordinated pool of routes for multi-GPU evaluation."""

    def __init__(self, pool_path, routes_file, repetitions, routes_subset,
                 rank, world_size, keep_repetition_order=True):
        """
        Args:
            pool_path: Path to the shared pool JSON file (inside the output dir).
            routes_file: Path to the routes XML file.
            repetitions: Number of repetitions per route.
            routes_subset: Subset string for route filtering.
            rank: This process's global rank.
            world_size: Total number of processes.
            keep_repetition_order: Whether to keep repetition order.
        """
        self.pool_path = pool_path
        self.lock_path = pool_path + '.lock'
        self.rank = rank
        self.world_size = world_size
        self._local_index = 0  # Tracks how many routes this rank has completed

        # Parse all route configs (same as RouteIndexer, but without sharding)
        route_configurations = RouteParser.parse_routes_file(routes_file, routes_subset)
        self.total_routes = len(route_configurations) * repetitions

        configs_list = []
        if keep_repetition_order:
            configs_dict = OrderedDict()
            for i, config in enumerate(route_configurations):
                for repetition in range(repetitions):
                    config.index = i * repetitions + repetition
                    config.repetition_index = repetition
                    configs_dict['{}.{}'.format(config.name, repetition)] = copy.copy(config)
        else:
            configs_dict = OrderedDict()
            for repetition in range(repetitions):
                for i, config in enumerate(route_configurations):
                    config_copy = copy.copy(config)
                    config_copy.index = repetition * len(route_configurations) + i
                    config_copy.repetition_index = repetition
                    configs_dict[f'{config.name}.{repetition}'] = config_copy
        configs_list = list(configs_dict.values())

        # Store configs keyed by their unique route key for lookup
        self._configs = {}
        for cfg in configs_list:
            key = f'{cfg.name}.{cfg.repetition_index}'
            self._configs[key] = cfg

        # Initialize the pool file (only one rank does it, protected by lock)
        self._initialize_pool(configs_list)

    def _initialize_pool(self, configs_list):
        """Create the pool file if it doesn't exist yet."""
        os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)

        with open(self.lock_path, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                if not os.path.exists(self.pool_path):
                    pool_data = {
                        'routes': [],
                        'total': len(configs_list),
                    }
                    for cfg in configs_list:
                        pool_data['routes'].append({
                            'key': f'{cfg.name}.{cfg.repetition_index}',
                            'name': cfg.name,
                            'town': str(cfg.town),
                            'status': 'available',  # available | taken | done
                            'rank': -1,
                        })
                    with open(self.pool_path, 'w') as f:
                        json.dump(pool_data, f, indent=2)
                    print(f"[Rank {self.rank}] Initialized route pool with {len(configs_list)} routes", flush=True)
                else:
                    print(f"[Rank {self.rank}] Route pool already exists", flush=True)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def acquire_next(self, current_town=None):
        """
        Acquire the next available route from the pool.

        Prefers routes with the same town as current_town to avoid world reloads.

        Args:
            current_town: The town currently loaded (e.g. "Town12"). None if no town loaded yet.

        Returns:
            Route config object, or None if no routes are available.
        """
        with open(self.lock_path, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                with open(self.pool_path, 'r') as f:
                    pool_data = json.load(f)

                # Find all available routes
                available = [(i, r) for i, r in enumerate(pool_data['routes']) if r['status'] == 'available']

                if not available:
                    return None

                # Prefer routes with matching town
                chosen_idx = None
                if current_town:
                    for i, r in available:
                        if r['town'] == current_town:
                            chosen_idx = i
                            break

                # Fall back to first available if no town match
                if chosen_idx is None:
                    chosen_idx = available[0][0]

                # Mark as taken
                pool_data['routes'][chosen_idx]['status'] = 'taken'
                pool_data['routes'][chosen_idx]['rank'] = self.rank

                with open(self.pool_path, 'w') as f:
                    json.dump(pool_data, f, indent=2)

                route_key = pool_data['routes'][chosen_idx]['key']
                route_town = pool_data['routes'][chosen_idx]['town']
                print(f"[Rank {self.rank}] Acquired route {route_key} (town={route_town}, "
                      f"preferred={current_town}, match={route_town == current_town})", flush=True)

            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

        # Look up the full config object
        config = copy.copy(self._configs[route_key])
        config.index = self._local_index
        self._local_index += 1
        return config

    def mark_done(self, config):
        """Mark a route as done in the pool."""
        route_key = f'{config.name}.{config.repetition_index}'

        with open(self.lock_path, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                with open(self.pool_path, 'r') as f:
                    pool_data = json.load(f)

                for r in pool_data['routes']:
                    if r['key'] == route_key:
                        r['status'] = 'done'
                        break

                with open(self.pool_path, 'w') as f:
                    json.dump(pool_data, f, indent=2)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def get_progress(self):
        """Return (completed_count, total_count) across all ranks."""
        with open(self.lock_path, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                with open(self.pool_path, 'r') as f:
                    pool_data = json.load(f)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

        done = sum(1 for r in pool_data['routes'] if r['status'] == 'done')
        return done, pool_data['total']
