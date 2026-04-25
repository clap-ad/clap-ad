# carla_offline_inference.py

import json
import os
import matplotlib.pyplot as plt
import torch

from lead.data_loader.carla_dataset import CARLAData
from lead.inference.config_open_loop import OpenLoopConfig
from lead.inference.open_loop_inference import OpenLoopInference
from lead.training.config_training import TrainingConfig
from lead.visualization.visualizer import Visualizer


def load_model_and_config(checkpoint_dir):
    open_loop_config = OpenLoopConfig()
    # training_config = TrainingConfig.load_from_file(
    #     os.path.join(checkpoint_dir, "config.json")
    # )
    config_dict = {}
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    training_config = TrainingConfig(loaded_config=config_dict, raise_error_on_missing_key=False)
    # training_config = training_config.load_from_args(loaded_config=config_dict, raise_error_on_missing_key=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    open_loop_inference = OpenLoopInference(
        config_training=training_config,
        config_open_loop=open_loop_config,
        model_path=checkpoint_dir,
        device=device,
    )
    return training_config, open_loop_inference


def get_dataset(training_config):
    training_config.carla_root = "data/carla_leaderboard2"
    dataset = CARLAData(
        root=training_config.carla_data, config=training_config, random=False
    )
    return dataset


def visualize_sample(training_config, dataset, index=42):
    sample = dataset[index]
    sample = torch.utils.data._utils.collate.default_collate([sample])
    visualizer = Visualizer(config=training_config, data=sample, prediction=None)
    image = visualizer.visualize_training_labels()
    plt.figure(figsize=(20, 10))
    plt.imshow(image)
    plt.show()
    return sample


def run_inference(open_loop_inference, sample):
    return open_loop_inference.forward(sample)


def visualize_prediction(training_config, sample, prediction, file_path):
    visualizer = Visualizer(config=training_config, data=sample, prediction=prediction)
    image = visualizer.visualize_training_prediction()
    plt.figure(figsize=(20, 10))
    plt.imshow(image)
    plt.savefig(file_path)  # Save the figure to a file
    plt.show()


def main():
    checkpoint_dir = "/workspace/lead/outputs/local_training/TestByeaghajanzadeh_20260420_120624"
    training_config, open_loop_inference = load_model_and_config(checkpoint_dir)
    dataset = get_dataset(training_config)
    seed = 42
    torch.manual_seed(seed)
    # indexes = [199821, 113647, 439060, 255368, 263793] # debugging
    indexes = torch.randperm(len(dataset))[:50].tolist()
    for index in indexes:
        sample = visualize_sample(training_config, dataset, index=index)
        prediction = run_inference(open_loop_inference, sample)
        file_path = f"prediction_{index}.png"
        folder = "/workspace/temp_vis"
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"prediction_{index}.png")
        visualize_prediction(training_config, sample, prediction, file_path)


if __name__ == "__main__":
    main()

# export PYTHONPATH=/workspace/lead:/workspace/lead:/workspace/lead/3rd_party/Bench2Drive/scenario_runner/srunner:/workspace/lead/3rd_party/CARLA_0915/PythonAPI:/workspace/lead/3rd_party/CARLA_0915/PythonAPI/carla:/workspace/lead/3rd_party/Bench2Drive:/workspace/lead/3rd_party/Bench2Drive/scenario_runner && python3.10 /workspace/lead/notebooks/carla_offline_inference_2.py