import numpy as np
from matplotlib import cm
from utils.general_utils import rescale


def convert_trial_to_tensor(star_plus_data, trial_idx):

    num_time_points = star_plus_data['data'][trial_idx][0].shape[0]

    dim_x = star_plus_data['meta']['dimx'].item().item()
    dim_y = star_plus_data['meta']['dimy'].item().item()
    dim_z = star_plus_data['meta']['dimz'].item().item()

    trial_tensor = np.zeros([dim_x, dim_y, dim_z, num_time_points])

    for i in range(num_time_points):
        trial_tensor[:, :, :, i] = convert_timepoint_to_tensor(star_plus_data, trial_idx, i)

    return trial_tensor


def convert_timepoint_to_tensor(star_plus_data, trial_idx, time_point):
    dim_x = star_plus_data['meta']['dimx'].item().item()
    dim_y = star_plus_data['meta']['dimy'].item().item()
    dim_z = star_plus_data['meta']['dimz'].item().item()

    time_tensor = np.zeros([dim_x, dim_y, dim_z])

    snapshot = star_plus_data['data'][trial_idx][0][time_point]

    # snapshot = rescale(snapshot, 1, 64)

    x = star_plus_data['meta']['colToCoord'].item()[:, 0] - 1
    y = star_plus_data['meta']['colToCoord'].item()[:, 1] - 1
    z = star_plus_data['meta']['colToCoord'].item()[:, 2] - 1

    time_tensor[x, y, z] = snapshot

    return time_tensor


def visualize_roi(star_plus_data):
    meta = star_plus_data['meta']
    num_roi = len(meta['rois'][0][0][0])
    dim_x = star_plus_data['meta']['dimx'].item().item()
    dim_y = star_plus_data['meta']['dimy'].item().item()
    dim_z = star_plus_data['meta']['dimz'].item().item()

    roi_tensor = np.zeros([dim_x, dim_y, dim_z])

    names = []
    for i in range(num_roi):
        coords = meta['rois'][0].item()['coords'][:, i][0]
        x = coords[:, 0] - 1
        y = coords[:, 1] - 1
        z = coords[:, 2] - 1
        roi_tensor[x, y, z] = i
        names.append(meta['rois'][0].item()['name'][0][i][0])

    my_color_map = cm.get_cmap('viridis', num_roi)
    my_color_map.colors[0] = [0, 0, 0, 1]

    return roi_tensor, my_color_map, names


def get_labels(star_plus_data):
    num_trials = star_plus_data['data'].shape[0]
    idx = np.arange(num_trials)

    info = star_plus_data['info']
    trial_cond = idx[(info['cond'] > 1)[0]]
    info = info[:, trial_cond]

    unique_cond = np.unique(info['cond'])
    num_cond = len(unique_cond)

    idx = np.arange(len(trial_cond))
    labels = np.zeros(len(trial_cond))
    labels[idx[(info['firstStimulus'] == 'S')[0]]] = 1
    labels = np.kron(labels, np.ones([1, 2]))

    for i in range(len(trial_cond)):
        trial_tensor = convert_trial_to_tensor(star_plus_data, trial_cond[i])

        if i == 0:
            tensor_P = trial_tensor[:, :, :, :16, np.newaxis]
            tensor_S = trial_tensor[:, :, :, 16:32, np.newaxis]
        else:
            tensor_P = np.concatenate((tensor_P, trial_tensor[:, :, :, :16, np.newaxis]), axis=4)
            tensor_S = np.concatenate((tensor_S, trial_tensor[:, :, :, 16:32, np.newaxis]), axis=4)

    tensor_PS = np.concatenate((tensor_P, tensor_S), axis=4)

    return tensor_PS, labels.reshape(-1)
