import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_data():
    bad_ROIs = sio.loadmat('Data/bad_ROIs.mat')['bad_ROIs']
    df_F = sio.loadmat('Data/df_F.mat')['df_F']
    events_above_min = sio.loadmat('Data/events_above_min.mat')['events_above_min']
    include_frames = sio.loadmat('Data/include_frames.mat')['include_frames']
    quad_data_norm = sio.loadmat('Data/quad_data_norm.mat')['quad_data_norm']
    scaled_centroids = sio.loadmat('Data/scaled_centroids.mat')['scaled_centroids']

    df_F = df_F[include_frames[0][0] - 1:include_frames[1][0] - 1, :]
    df_F = np.delete(df_F, np.squeeze(bad_ROIs) - 1, axis=1)
    events_above_min = events_above_min[include_frames[0][0] - 1:include_frames[1][0] - 1, :]
    events_above_min = np.delete(events_above_min, np.squeeze(bad_ROIs) - 1, axis=1)
    scaled_centroids = np.delete(scaled_centroids, np.squeeze(bad_ROIs) - 1, axis=0)
    quad_data_norm = np.squeeze(quad_data_norm)[include_frames[0][0] - 1:include_frames[1][0] - 1]

    return bad_ROIs, df_F, events_above_min, include_frames, quad_data_norm, scaled_centroids


def generate_video_data(HP):
    bad_ROIs, df_F, events_above_min, include_frames, quad_data_norm, scaled_centroids = load_data()

    total_frames = len(events_above_min)

    min_x = np.min(scaled_centroids[:, 0])
    min_y = np.min(scaled_centroids[:, 1])
    max_x = np.max(scaled_centroids[:, 0])
    max_y = np.max(scaled_centroids[:, 1])

    scaled_centroids[:, 0] = np.rint((HP['grid_size'] - 1) * (scaled_centroids[:, 0] - min_x) / (max_x - min_x))
    scaled_centroids[:, 1] = np.rint((HP['grid_size'] - 1) * (scaled_centroids[:, 1] - min_y) / (max_y - min_y))
    scaled_centroids = scaled_centroids.astype(int)

    video = np.zeros((len(range(0, total_frames, HP['hop'])), 2, HP['grid_size'], HP['grid_size']))
    mouse_position = np.zeros(len(range(0, total_frames, HP['hop'])))

    layer_0 = np.where(scaled_centroids[:, 2] == 0)[0]
    layer_1 = np.where(scaled_centroids[:, 2] == 1)[0]

    count = 0
    for i in range(0, total_frames, HP['hop']):
        active = np.where(events_above_min[i, :] == 1)[0]
        inactive = np.where(events_above_min[i, :] == 0)[0]

        x_active_layer0, y_active_layer0 = scaled_centroids[np.intersect1d(layer_0, active), 0], scaled_centroids[
            np.intersect1d(layer_0, active), 1]

        x_inactive_layer0, y_inactive_layer0 = scaled_centroids[np.intersect1d(layer_0, inactive), 0], scaled_centroids[
            np.intersect1d(layer_0, inactive), 1]

        x_active_layer1, y_active_layer1 = scaled_centroids[np.intersect1d(layer_1, active), 0], scaled_centroids[
            np.intersect1d(layer_1, active), 1]

        x_inactive_layer1, y_inactive_layer1 = scaled_centroids[np.intersect1d(layer_1, inactive), 0], scaled_centroids[
            np.intersect1d(layer_1, inactive), 1]

        video[count, 0, x_active_layer0, y_active_layer0] = 0.5
        video[count, 1, x_active_layer1, y_active_layer1] = 0.5
        video[count, 0, x_inactive_layer0, y_inactive_layer0] = -0.5
        video[count, 1, x_inactive_layer1, y_inactive_layer1] = -0.5

        mouse_position[count] = quad_data_norm[i] * 2 * np.pi

        count += 1


    return torch.from_numpy(video), torch.from_numpy(mouse_position)


def generate_features_data(HP):
    _, df_F, events_above_min, _, quad_data_norm, _ = load_data()

    features = events_above_min

    mouse_position = quad_data_norm * 2 * np.pi

    return torch.from_numpy(features), torch.from_numpy(mouse_position)


if __name__ == '__main__':
    HP = {'grid_size': 100, 'hop': 1}
    video, _ = generate_video_data(HP)
    print(len(video))

    HP = {'grid_size': 100, 'hop': 2}
    video, _ = generate_video_data(HP)
    print(len(video))

    # features, mouse = generate_features_data(HP)
