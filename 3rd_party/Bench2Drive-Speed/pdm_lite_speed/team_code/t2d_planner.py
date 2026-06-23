import os
from collections import deque

import numpy as np


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class T2DRoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        # self.mean = np.array([0.0, 0.0]) # for carla 9.10
        # self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10
        self.mean = np.array([0.0, 0.0]) # for pdm-lite
        self.scale = np.array([111319.49082349832, 111319.49079327358]) # for pdm-lite

        self.debug = Plotter(debug_size)
        
        # default for big maps
        # self.lat_ref = 42.0
        # self.lon_ref = 2.0

    def set_route(self, global_plan, gps=False, global_plan_world=None):
        self.route.clear()

        if global_plan_world:
            for (pos, cmd), (pos_word, _ )in zip(global_plan, global_plan_world):
                if gps:
                    # if abs(pos['lat'] - self.lat_ref) < abs(pos['lat']):
                    #     pos['lat'] -= self.lat_ref
                    # if abs(pos['lon'] - self.lon_ref) < abs(pos['lon']):
                    #     pos['lon'] -= self.lon_ref
                    pos = np.array([pos['lat'], pos['lon']])
                    pos -= self.mean
                    pos *= self.scale
                    pos = np.array([pos[1], -pos[0]]) # align to pdm-lite
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean
                self.route.append((pos, cmd, pos_word))
        else:
            for pos, cmd in global_plan:
                if gps:
                    # if abs(pos['lat'] - self.lat_ref) < abs(pos['lat']):
                    #     pos['lat'] -= self.lat_ref
                    # if abs(pos['lon'] - self.lon_ref) < abs(pos['lon']):
                    #     pos['lon'] -= self.lon_ref
                    pos = np.array([pos['lat'], pos['lon']])
                    pos -= self.mean
                    pos *= self.scale
                    pos = np.array([pos[1], -pos[0]]) # align to pdm-lite
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean

                self.route.append((pos, cmd))
    
    def get_remaining_route(self, gps, step=0.2):
        if len(self.route) < 2:
            return list(self.route)

        distances = [np.linalg.norm(p[0] - gps) for p in self.route]
        nearest_idx = int(np.argmin(distances))

        remaining_route = list(self.route)[nearest_idx:]

        interpolated_route = []
        for i in range(len(remaining_route) - 1):
            p1, cmd1 = remaining_route[i][0], remaining_route[i][1]
            p2, cmd2 = remaining_route[i + 1][0], remaining_route[i + 1][1]

            interpolated_route.append((p1, cmd1))

            segment_vec = p2 - p1
            segment_dist = np.linalg.norm(segment_vec)
            if segment_dist < 1e-3:
                continue

            num_steps = int(segment_dist // step)
            if num_steps > 1:
                direction = segment_vec / segment_dist
                for j in range(1, num_steps):
                    new_point = p1 + direction * (j * step)
                    interpolated_route.append((new_point, cmd1))

        interpolated_route.append(remaining_route[-1])

        # print(f"[debug] before transform, points are {interpolated_route}")

        return interpolated_route

        # def transform_point(pt):
        #     return np.array((pt[1], -pt[0]), dtype=pt.dtype)

        # transformed = [(transform_point(p[0]), p[1]) for p in interpolated_route]

        # print(f"[debug] after transform, points are {transformed}")
        # return transformed

    # def run_step(self, gps):
    #     # self.debug.clear()
    #     # print(f"[debug] in RoutePlanner's run_step, self.route = {self.route}")
    #     if len(self.route) == 1:
    #         return self.route[0]

    #     to_pop = 0
    #     farthest_in_range = -np.inf
    #     cumulative_distance = 0.0

    #     for i in range(1, len(self.route)):
    #         if cumulative_distance > self.max_distance:
    #             break

    #         cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
    #         # print(f"[debug] in run_step, self.route[i][0] = {self.route[i][0]}, gps = {gps}")
    #         distance = np.linalg.norm(self.route[i][0] - gps)

    #         if distance <= self.min_distance and distance > farthest_in_range:
    #             farthest_in_range = distance
    #             to_pop = i

    #         r = 255 * int(distance > self.min_distance)
    #         g = 255 * int(self.route[i][1].value == 4)
    #         b = 255
    #         # self.debug.dot(gps, self.route[i][0], (r, g, b))

    #     for _ in range(to_pop):
    #         if len(self.route) > 2:
    #             self.route.popleft()

    #     # self.debug.dot(gps, self.route[0][0], (0, 255, 0))
    #     # self.debug.dot(gps, self.route[1][0], (255, 0, 0))
    #     # self.debug.dot(gps, gps, (0, 0, 255))
    #     # self.debug.show()
    #     # print(f"[debug] in run_step, self.route = {self.route}")

    #     return self.route[1]
    
    def run_step(self, gps):
        """
        Returns the next waypoint in front of the vehicle based on GPS.
        
        Logic:
        1. Iterate through the route starting from the second waypoint.
        2. Find the waypoint closest to the current GPS and within min_distance.
        3. Ensure cumulative distance along the route does not exceed max_distance.
        4. Return the next waypoint (route[1]) as the target.
        """
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        closest_distance = np.inf
        cumulative_distance = 0.0

        # Start from the second waypoint (index 1)
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            # Update cumulative distance along the route
            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            # Choose the closest waypoint within min_distance
            if distance <= self.min_distance and distance < closest_distance:
                closest_distance = distance
                to_pop = i

        # Pop the first to_pop waypoints to keep route[0] as the current position
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        # Return the next waypoint as the target
        return self.route[1]

