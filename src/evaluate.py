import torch

import numpy as np
import os

from src.models.simple_ml import Oracle, Naive, DirectRandomForest, DirectLightGBM
from src.models.simple_ml import SmallLightGBM, SmallRandomForest
from src.Simulator import RealSimulator

import pandas as pd

np.random.seed(0)


def find_the_best_next_fast(state, model, cur_optimal, next_optimal):
    """
    approximates going 3 steps in one direction, by going 1 step and multiplying the difference by 3
    """
    from itertools import product

    controls = np.array(list(product(range(-3, 4), range(-3, 4))))
    inputs = np.concatenate((np.tile(cur_optimal, (49, 1)), controls), axis=1)
    next_states = state + model.batch_call(inputs)

    dist = ((next_states - next_optimal) ** 2).sum(axis=1)
    opt_control = controls[np.argmin(dist)]
    cur_min = dist.min()
    chosen_next_state = next_states[np.argmin(dist)]

    return opt_control, cur_min, chosen_next_state


def find_the_best_next_exact(state, model, cur_optimal, next_optimal):
    from itertools import product

    cur_min = 1999999999

    for control in product(range(-3, 4), range(-3, 4)):
        next_imagined_state = np.clip(state + model(cur_optimal, control), 0, 1100)

        dist = np.square(next_optimal[0] - next_imagined_state[0]) + np.square(next_optimal[1] - next_imagined_state[1])
        if dist < cur_min:
            cur_min = dist
            opt_control = control
            chosen_next_state = next_imagined_state

    return opt_control, cur_min, chosen_next_state


def print_with_world_model(starting_point, next_optimal_states, model):
    """
    cur_optimal point represents here the destination we would reach if we used given controls on a perfect printer,
    it might be different than the shape we are actually trying to print
    It is calculated from previously selected actions and used as a predictor for the corrupted path
    """
    cur_imagined_point = starting_point.copy()
    cur_optimal = starting_point.copy()
    imagined_points = [cur_imagined_point]
    selected_controls = []

    # cur optimal is the starting point + sum of the actions
    # next optimal is the point we are trying to reach
    for i, next_optimal in enumerate(next_optimal_states):
        if fast:
            chosen_control, _, cur_imagined_point = find_the_best_next_fast(cur_imagined_point, model, cur_optimal, next_optimal)
        else:
            chosen_control, _, cur_imagined_point = find_the_best_next_exact(cur_imagined_point, model, cur_optimal, next_optimal)
        imagined_points.append(cur_imagined_point)
        selected_controls.append(chosen_control)
        cur_optimal = np.clip(cur_optimal + chosen_control, 0, 1100)
    return selected_controls, imagined_points


def print_with_given_controls(starting_point, controls, corr_id, real_simulation):
    """
    optimal point represents here the destination we would reach if we used chosen controls on a perfect printer,
    it might be different than the shape we are actually trying to print.
    It is calculated from previously selected actions and used as a predictor for the corrupted path
    """
    cur_point = starting_point
    optimal_point = starting_point
    printed_points = [tuple(cur_point)]
    for ctrl in controls:
        cur_point = np.clip(cur_point + real_simulation.get_next(optimal_point, ctrl, corr_id) - optimal_point, 0, 1100)

        optimal_point = np.clip(optimal_point + ctrl, 0, 1100)
        printed_points.append(cur_point)
    return printed_points


def RMSE(optimal_points, printed_points):
    MSE = ((optimal_points[:, 0] - printed_points[:, 0]) ** 2 + (
            optimal_points[:, 1] - printed_points[:, 1]) ** 2).mean()

    return np.sqrt(MSE)


def evaluate_model_adapt_to_body(model, method, split='val'):
    """
    calculates the metric on the dataset for evaluation
    :param model: function that serves as a world model, model: position x control ｜ helper_trajectory -> position_prediction
    :return: MSE over the validation data
    """
    true_simulation = RealSimulator()

    losses = []

    data_for_csv = {'names': [], 'errors': []}
    print('files to visit: ', len(files))
    for i ,file in enumerate(files):
        if i % 5 == 0:
            print('visiting file number ', i)
        file_path = os.path.join(files_directory, file)
        #print(file_path)
        data = np.genfromtxt(file_path, delimiter=',')
        if len(data) == 0:
            continue

        all_trajectories_counts = np.unique(data[:, 3], return_counts=True)
        all_trajectories = all_trajectories_counts[0]
        longest_traj_id = all_trajectories[all_trajectories_counts[1].argmax()]
        #print('longest traj length: ', all_trajectories_counts[1].max())

        longest_traj = data[data[:, 3] == longest_traj_id]
        model.adapt_to_traj(longest_traj)


        opt_trajs = []
        printed_trajs = []
        for traj_id in np.random.choice(all_trajectories, min(3, len(all_trajectories)), replace=False):
            cur_trajectory = data[data[:, 3] == traj_id]
            starting_point = cur_trajectory[0, [4, 9]]

            # calculate the controls that the model will select
            next_optimal_states = cur_trajectory[:, [7, 12]]
            corr_id = cur_trajectory[0, [1, 2]]

            selected_controls, imagined_points = print_with_world_model(starting_point, next_optimal_states, model)

            # calculate what the model will actually print
            printed_points = print_with_given_controls(starting_point, selected_controls, corr_id, true_simulation)

            optimal_states = np.concatenate([np.expand_dims(starting_point, 0), next_optimal_states])

            opt_trajs.append(optimal_states)
            printed_trajs.append(np.array(printed_points))

        err = RMSE(np.concatenate(opt_trajs), np.concatenate(printed_trajs))
        data_for_csv['names'].append(file)
        data_for_csv['errors'].append(err)

    pd.DataFrame.from_dict(data_for_csv).to_csv(f'data/results/{method}_dist_with_names.csv')
    np.savetxt('data/results/distribution2.csv', np.array(losses))
    return np.mean(losses)


if __name__ == '__main__':
    files_directory = os.path.join('data', 'dataset')
    files = np.random.choice(os.listdir(files_directory), 500, replace=False)

    fast = False
    print('Naive: ', evaluate_model_adapt_to_body(Naive(), 'naive'))
    print('Oracle: ', evaluate_model_adapt_to_body(Oracle(), 'oracle'))

    fast = True
    print('small rf: ', evaluate_model_adapt_to_body(SmallRandomForest(), 'small_rf'))
    print('full rf: ', evaluate_model_adapt_to_body(DirectRandomForest(), 'full_rf'))
    print('lgb small: ', evaluate_model_adapt_to_body(SmallLightGBM(), 'lgb'))
    print('lgb full: ', evaluate_model_adapt_to_body(DirectLightGBM(), 'lgb'))
