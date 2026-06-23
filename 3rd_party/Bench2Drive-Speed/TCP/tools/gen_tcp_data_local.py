import os
import json
import numpy as np
from tqdm import trange
import gzip
import multiprocessing as mp 
import time
import numpy as np
from math import radians, cos, sin

INPUT_FRAMES = 1
FUTURE_FRAMES = 4*5 # 10hz --> 2hz
TRAIN = True

val_list = [
    'OvertakeRoute+Follow_Town12_Road1096_Route21278_Weather19_01-02-22-01-58',
    'OvertakeRoute+Follow_Town13_Road1411_Route29239_Weather8_12-29-12-31-46',
    'OvertakeRoute+Follow_Town15_Road288_Route26658_Weather15_01-02-16-10-47',
    'OvertakeRoute+Overtake_Town12_Road563_Route2564_Weather1_12-31-08-44-35',
    'OvertakeRoute+Overtake_Town13_Road519_Route28443_Weather1_12-30-12-57-21',
    'OvertakeRoute+Overtake_Town15_Road140_Route26633_Weather9_01-02-09-32-44'
] # b2ds

# val_list = [
# 	'Accident_Town03_Route101_Weather23',
# 	'ParkedObstacle_Town03_Route103_Weather25',
#  	'SignalizedJunctionLeftTurn_Town12_Route799_Weather0',
#  	'StaticCutIn_Town05_Route226_Weather18',
#  	'PedestrianCrossing_Town12_Route1014_Weather0',
#  	'NonSignalizedJunctionRightTurn_Town03_Route126_Weather18'
# ] # base1000

Discrete_Actions_DICT = {
	0:  (0, 0, 1, False),
	1:  (0.7, -0.5, 0, False),
	2:  (0.7, -0.3, 0, False),
	3:  (0.7, -0.2, 0, False),
	4:  (0.7, -0.1, 0, False),
	5:  (0.7, 0, 0, False),
	6:  (0.7, 0.1, 0, False),
	7:  (0.7, 0.2, 0, False),
	8:  (0.7, 0.3, 0, False),
	9:  (0.7, 0.5, 0, False),
	10: (0.3, -0.7, 0, False),
	11: (0.3, -0.5, 0, False),
	12: (0.3, -0.3, 0, False),
	13: (0.3, -0.2, 0, False),
	14: (0.3, -0.1, 0, False),
	15: (0.3, 0, 0, False),
	16: (0.3, 0.1, 0, False),
	17: (0.3, 0.2, 0, False),
	18: (0.3, 0.3, 0, False),
	19: (0.3, 0.5, 0, False),
	20: (0.3, 0.7, 0, False),
	21: (0, -1, 0, False),
	22: (0, -0.6, 0, False),
	23: (0, -0.3, 0, False),
	24: (0, -0.1, 0, False),
	25: (1, 0, 0, False),
	26: (0, 0.1, 0, False),
	27: (0, 0.3, 0, False),
	28: (0, 0.6, 0, False),
	29: (0, 1.0, 0, False),
	30: (0.5, -0.5, 0, True),
	31: (0.5, -0.3, 0, True),
	32: (0.5, -0.2, 0, True),
	33: (0.5, -0.1, 0, True),
	34: (0.5, 0, 0, True),
	35: (0.5, 0.1, 0, True),
	36: (0.5, 0.2, 0, True),
	37: (0.5, 0.3, 0, True),
	38: (0.5, 0.5, 0, True),
}

def get_action(index):
	throttle, steer, brake, reverse = Discrete_Actions_DICT[index]
	return throttle, steer, brake

def get_closest_action_index(throttle, steer, brake, reverse):
    valid_actions = {index: action for index, action in Discrete_Actions_DICT.items() if action[3] == reverse}
    
    closest_index = -1
    closest_diff = float('inf')
    
    for index, (throttle_action, steer_action, brake_action, reverse_action) in valid_actions.items():
        diff = abs(throttle_action - throttle) + abs(steer_action - steer) + abs(brake_action - brake)
        
        if diff < closest_diff:
            closest_diff = diff
            closest_index = index
    
    return closest_index

# Converts a Carla vector to a numpy array
def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector[0], carla_vector[1]])
    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)
    return result

# Returns the orientation (direction vector) from a given yaw (rotation angle)
def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

# Main collision detection function
def collision_detect(bounding_boxes):
    ego_vehicle = None
    for box in bounding_boxes:
        if box['class'] == 'ego_vehicle':
            ego_vehicle = box
            break

    if ego_vehicle is None:
        raise ValueError("No ego vehicle found in the bounding boxes!")

    is_vehicle_present = 0
    is_pedestrian_present = 0

    # Check for vehicle and pedestrian hazards
    vehicle = _is_vehicle_hazard(ego_vehicle, bounding_boxes, is_pedestrian=False)
    walker = _is_walker_hazard(ego_vehicle, bounding_boxes)

    is_vehicle_present = 1 if vehicle is not None else 0
    is_pedestrian_present = 1 if walker is not None else 0

    return any(x is not None for x in [vehicle, walker])

# Detect collision with pedestrians
def _is_walker_hazard(ego_vehicle, bounding_boxes):
    for box in bounding_boxes:
        if box['class'] != 'walker':
            continue
        
        p2 = _numpy(box['location'])
        v2_hat = _orientation(box['rotation'][2])
        s2 = box.get('speed', 0)

        if s2 < 0.05:
            v2_hat *= s2
        
        p1 = _numpy(ego_vehicle['location'])
        v1 = 10.0 * _orientation(ego_vehicle['rotation'][2])

        collides, collision_point = get_collision(p1, v1, p2, v2_hat)
        
        if collides:
            return box  # Return pedestrian that collided

    return None

# Detect collision with other vehicles
def _is_vehicle_hazard(ego_vehicle, bounding_boxes, is_pedestrian=False):
    for box in bounding_boxes:
        if box['class'] != 'vehicle' or (is_pedestrian and box['class'] == 'walker'):
            continue

        p2 = _numpy(box['location'])
        v2_hat = _orientation(box['rotation'][2])
        s2 = max(5.0, 2.0 * np.linalg.norm(np.array(box.get('speed', 0))))
        v2 = s2 * v2_hat

        p1 = _numpy(ego_vehicle['location'])
        s1 = max(10.0, 3.0 * np.linalg.norm(np.array(ego_vehicle.get('speed', 0))))
        v1 = s1 * _orientation(ego_vehicle['rotation'][2])

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(np.dot(v2_hat, p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(np.dot(_orientation(ego_vehicle['rotation'][2]), _orientation(box['rotation'][2]))))

        angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
        angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue

        return box  # Return the vehicle that collided

    return None

# Simple collision detection (linear prediction)
def get_collision(p1, v1, p2, v2):
    t = (p2[0] - p1[0]) / (v1[0] - v2[0]) if v1[0] != v2[0] else float('inf')
    if t < 0:
        return False, None
    collision_point = p1 + v1 * t
    return np.linalg.norm(collision_point - p2) < 0.5, collision_point


class Colors:
	RED = '\033[91m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	BLUE = '\033[94m'
	MAGENTA = '\033[95m'
	CYAN = '\033[96m'
	WHITE = '\033[97m'
	RESET = '\033[0m'

def gen_single_route(route_folder, count):

	folder_path = os.path.join(route_folder, 'appended_anno')
	# if not os.path.exists(folder_path):
	# 	folder_path = os.path.join(route_folder, 'anno')

	existing_files = [f for f in os.listdir(folder_path) if f.endswith('.json.gz')]
 
	sample_file = os.path.join(folder_path, existing_files[0])  # pick the first valid file
	actual_folder_path = os.path.dirname(os.path.realpath(sample_file))
	length = len([name for name in os.listdir(actual_folder_path)]) - 1 # drop last frame
	
	if length < INPUT_FRAMES + FUTURE_FRAMES:
		return

	seq_future_x = []
	seq_future_y = []
	seq_future_theta = []
	# seq_future_feature = []
	seq_future_action = []
	seq_future_action_index = []

	seq_future_only_ap_brake = []


	seq_input_x = []
	seq_input_y = []
	seq_input_theta = []

	seq_front_img = []
	# seq_feature = []
	seq_value = []
	seq_speed = []

	seq_action = []
	seq_action_index = []

	seq_x_target = []
	seq_y_target = []
	seq_target_command = []

	seq_only_ap_brake = []

	seq_target_speed = []
	seq_virtual_target_speed = []
	seq_tendency_speed = []
	seq_do_overtake = []

	full_seq_x = []
	full_seq_y = []
	full_seq_theta = []

	# full_seq_feature = []
	full_seq_action = []
	full_seq_action_index = []
	full_seq_only_ap_brake = []

	for i in trange(length):
		with gzip.open(os.path.join(actual_folder_path, f'{i:05}.json.gz'), 'rt', encoding='utf-8') as gz_file:
			anno = json.load(gz_file)

		# expert_feature = np.load(os.path.join(route_folder, f'expert_assessment/{i:05}.npz'), allow_pickle=True)['arr_0']

		full_seq_x.append(anno['x'])
		full_seq_y.append(anno['y'])  # TODO(yzj): need to align sign
		full_seq_theta.append(anno['theta'])
		# full_seq_feature.append(expert_feature[:-2])
		# throttle, steer, brake = get_action(int(expert_feature[-1]))
		full_seq_action.append(np.array([anno['throttle'], anno['steer'], anno['brake']], dtype=np.float32))
		full_seq_action_index.append(int(get_closest_action_index(throttle=anno['throttle'],
                                                       		 	  steer=anno['steer'],
                                                          	 	  brake=anno['brake'],
                                                             	  reverse=anno['reverse'])))
  
		if 'should_brake' not in anno:
			anno['should_brake'] = collision_detect(anno['bounding_boxes'])
		if 'only_ap_brake' not in anno:
			anno['only_ap_brake'] = True if (anno['brake'] <= 0 and anno['should_brake']) else False
		full_seq_only_ap_brake.append(anno['only_ap_brake'])

	for i in trange(INPUT_FRAMES-1, length-FUTURE_FRAMES-5):
		frame_path = os.path.join(route_folder, f'appended_anno/{i:05}.json.gz')
		if not os.path.exists(frame_path):
			continue
		with gzip.open(frame_path, 'rt', encoding='utf-8') as gz_file:
			anno = json.load(gz_file)

		# expert_feature = np.load(os.path.join(route_folder, f'expert_assessment/{i:05}.npz'), allow_pickle=True)['arr_0']

		seq_input_x.append(full_seq_x[i-(INPUT_FRAMES-1):i+5:5])
		seq_input_y.append(full_seq_y[i-(INPUT_FRAMES-1):i+5:5])
		seq_input_theta.append(full_seq_theta[i-(INPUT_FRAMES-1):i+5:5])

		seq_future_x.append(full_seq_x[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_y.append(full_seq_y[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_theta.append(full_seq_theta[i+5:i+FUTURE_FRAMES+5:5])

		# seq_future_feature.append(full_seq_feature[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_action.append(full_seq_action[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_action_index.append(full_seq_action_index[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_only_ap_brake.append(full_seq_only_ap_brake[i+5:i+FUTURE_FRAMES+5:5])

		# seq_feature.append(expert_feature[:-2])
		# seq_value.append(expert_feature[-2])
		
		front_img_list = [os.path.join(route_folder, f'camera/rgb_front/{i:05}.jpg') for _ in range(INPUT_FRAMES-1, -1, -1)]
		seq_front_img.append(front_img_list)

		seq_speed.append(anno["speed"])

		# throttle, steer, brake = get_action(int(expert_feature[-1]))
		seq_action.append(np.array([anno['throttle'], anno['steer'], anno['brake']], dtype=np.float32))  # step + action = next_step
		# seq_action_index.append(int(expert_feature[-1]))
		seq_action_index.append(int(get_closest_action_index(throttle=anno['throttle'],
                                                       		 steer=anno['steer'],
                                                          	 brake=anno['brake'],
                                                             reverse=anno['reverse'])))

		seq_x_target.append(anno["x_target"])
		seq_y_target.append(anno["y_target"])
		seq_target_command.append(anno["command_far"])
		if 'should_brake' not in anno:
			anno['should_brake'] = collision_detect(anno['bounding_boxes'])
		if 'only_ap_brake' not in anno:
			anno['only_ap_brake'] = True if (anno['brake'] <= 0 and anno['should_brake']) else False
		seq_only_ap_brake.append(anno["only_ap_brake"])

		if 'given_target_speed' not in anno:
			anno['given_target_speed'] = anno['virtual_target_speed']
		seq_target_speed.append(anno['given_target_speed'])
		seq_virtual_target_speed.append(anno['virtual_target_speed'])
		seq_tendency_speed.append(anno['tendency_speed'])
		seq_do_overtake.append(anno.get('do_overtake', 0))

	with count.get_lock():
		count.value += 1
	return seq_future_x, seq_future_y, seq_future_theta, seq_future_action, seq_future_action_index, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_front_img, seq_value, seq_speed, seq_action, seq_action_index, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake, seq_target_speed, seq_do_overtake, seq_virtual_target_speed, seq_tendency_speed

def gen_sub_folder(seq_data_list):
	print('begin saving...', flush=True)
	total_future_x = []
	total_future_y = []
	total_future_theta = []

	total_future_feature = []
	total_future_action = []
	total_future_action_index = []
	total_future_only_ap_brake = []

	total_input_x = []
	total_input_y = []
	total_input_theta = []

	total_front_img = []
	total_feature = []
	total_value = []
	total_speed = []

	total_action = []
	total_action_index = []

	total_x_target = []
	total_y_target = []
	total_target_command = []

	total_only_ap_brake = []

	total_target_speed = []
	total_virtual_target_speed = []
	total_tendency_speed = []
	total_do_overtake = []

	for seq_data in seq_data_list:
		# seq_data = gen_single_route(os.path.join(folder_path, route))
		if not seq_data:
			continue
		seq_future_x, seq_future_y, seq_future_theta, seq_future_action, seq_future_action_index, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_front_img, seq_value, seq_speed, seq_action, seq_action_index, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake, seq_target_speed, seq_do_overtake, seq_virtual_target_speed, seq_tendency_speed = seq_data
		total_future_x.extend(seq_future_x)
		total_future_y.extend(seq_future_y)
		total_future_theta.extend(seq_future_theta)
		# total_future_feature.extend(seq_future_feature)
		total_future_action.extend(seq_future_action)
		total_future_action_index.extend(seq_future_action_index)
		total_future_only_ap_brake.extend(seq_future_only_ap_brake)
		total_input_x.extend(seq_input_x)
		total_input_y.extend(seq_input_y)
		total_input_theta.extend(seq_input_theta)
		total_front_img.extend(seq_front_img)
		# total_feature.extend(seq_feature)
		total_value.extend(seq_value)
		total_speed.extend(seq_speed)
		total_action.extend(seq_action)
		total_action_index.extend(seq_action_index)
		total_x_target.extend(seq_x_target)
		total_y_target.extend(seq_y_target)
		total_target_command.extend(seq_target_command)
		total_only_ap_brake.extend(seq_only_ap_brake)
		
		total_target_speed.extend(seq_target_speed)
		total_tendency_speed.extend(seq_tendency_speed)
		total_virtual_target_speed.extend(seq_virtual_target_speed)
		total_do_overtake.extend(seq_do_overtake)

	data_dict = {}
	data_dict['future_x'] = total_future_x
	data_dict['future_y'] = total_future_y
	data_dict['future_theta'] = total_future_theta
	# data_dict['future_feature'] = total_future_feature
	data_dict['future_action'] = total_future_action
	data_dict['future_action_index'] = total_future_action_index
	data_dict['future_only_ap_brake'] = total_future_only_ap_brake
	data_dict['input_x'] = total_input_x
	data_dict['input_y'] = total_input_y
	data_dict['input_theta'] = total_input_theta
	data_dict['front_img'] = total_front_img
	# data_dict['feature'] = total_feature
	# data_dict['value'] = total_value
	data_dict['speed'] = total_speed
	data_dict['action'] = total_action
	data_dict['action_index'] = total_action_index
	data_dict['x_target'] = total_x_target
	data_dict['y_target'] = total_y_target
	data_dict['target_command'] = total_target_command
	data_dict['only_ap_brake'] = total_only_ap_brake

	data_dict['target_speed'] = total_target_speed
	data_dict['virtual_target_speed'] = total_virtual_target_speed
	data_dict['tendency_speed'] = total_tendency_speed
	data_dict['do_overtake'] = total_do_overtake

	if TRAIN:
		file_path = os.path.join("tcp_b2ds-train")
	else:
		file_path = os.path.join("tcp_b2ds-val")
	np.save(file_path, data_dict)
	print(f'begin saving, length={len(total_future_x)}', flush=True)

def get_folder_path(folder_paths, total):
	path = '/path/to/dataset'
	for d0 in os.listdir(path):
		if 'copy' in path:
			print(f"[debug] ignored copied path: {path}")
			continue
		if TRAIN:
			if d0 not in val_list:
				folder_paths.put(os.path.join(path, d0))
				with total.get_lock():
					total.value += 1
		else:
			if d0 in val_list:
				folder_paths.put(os.path.join(path, d0))
				with total.get_lock():
					total.value += 1
	return folder_paths

def worker(folder_paths, count, seq_data_list, stop_event, worker_num, completed_workers):
	while True:
		if folder_paths.qsize()<=0:
			with completed_workers.get_lock():
				completed_workers.value += 1
				if completed_workers.value == worker_num:
					stop_event.set()
			break
		folder_path = folder_paths.get()
		if '_' in folder_path:
			seq_data = gen_single_route(folder_path, count)
			seq_data_list.append(seq_data)

def display(count, total, stop_event, completed_workers):
	t1 = time.time()
	while True:
		print(f'{Colors.GREEN}[count/total]=[{count.value}/{total.value}, {count.value/(time.time()-t1):.2f}it/s, completed_workers={completed_workers.value}]{Colors.RESET}', flush=True)
		time.sleep(3)
		if stop_event.is_set():
			break

if __name__ == '__main__':
	folder_paths = mp.Queue()
	seq_data_list = mp.Manager().list()
	count = mp.Value('d', 0)
	total = mp.Value('d', 0)
	stop_event = mp.Event()
	completed_workers = mp.Value('d', 0)

	get_folder_path(folder_paths, total)
	ps = []
	worker_num = 64
	for i in range(worker_num):
		p = mp.Process(target=worker, args=(folder_paths, count, seq_data_list, stop_event, worker_num, completed_workers, ))
		p.daemon = True
		p.start()
		ps.append(p)
	
	p = mp.Process(target=display, args=(count, total, stop_event, completed_workers))
	p.daemon = True
	p.start()
	ps.append(p)
	
	for p in ps:
		p.join()
	
	display(count, total, stop_event, completed_workers)
	gen_sub_folder(seq_data_list)
	display(count, total, stop_event, completed_workers)