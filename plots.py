import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import data


def plot_video(frames=1000, save_video=False):
    bad_ROIs, df_F, events_above_min, include_frames, quad_data_norm, scaled_centroids = data.load_data()

    plt.ion()

    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132, projection='polar')
    ax3 = plt.subplot(133)
    fig = plt.gcf()
    fig.set_size_inches(15, 5)

    total_frames = len(events_above_min)
    min_x = np.min(scaled_centroids[:, 0])
    min_y = np.min(scaled_centroids[:, 1])
    max_x = np.max(scaled_centroids[:, 0])
    max_y = np.max(scaled_centroids[:, 1])
    delta = (max_x - min_x) / 20

    plt.xlim(min_x - delta, max_x + delta)
    plt.ylim(min_y - delta, max_y + delta)

    layer_0 = np.where(scaled_centroids[:, 2] == 0)[0]
    layer_1 = np.where(scaled_centroids[:, 2] == 1)[0]

    for i in range(frames):
        print('Frame: ' + str(i))
        active = np.where(events_above_min[i, :] == 1)[0]
        inactive = np.where(events_above_min[i, :] == 0)[0]

        ax1.cla()
        ax1.scatter(scaled_centroids[np.intersect1d(layer_0, active), 0],
                    scaled_centroids[np.intersect1d(layer_0, active), 1], c='b', s=10)
        ax1.scatter(scaled_centroids[np.intersect1d(layer_0, inactive), 0],
                    scaled_centroids[np.intersect1d(layer_0, inactive), 1], c='r', s=10)
        ax1.set_title(f'Activity ~ {np.round(100 * len(np.intersect1d(layer_0, active)) / len(layer_0), 1)}%')

        ax3.cla()
        ax3.scatter(scaled_centroids[np.intersect1d(layer_1, active), 0],
                    scaled_centroids[np.intersect1d(layer_1, active), 1], c='b', s=10)
        ax3.scatter(scaled_centroids[np.intersect1d(layer_1, inactive), 0],
                    scaled_centroids[np.intersect1d(layer_1, inactive), 1], c='r', s=10)
        ax3.set_title(f'Activity ~ {np.round(100 * len(np.intersect1d(layer_1, active)) / len(layer_1), 1)}%')

        ax2.cla()
        ax2.set_yticklabels([])
        plt.sca(ax2)
        ax2.scatter(quad_data_norm[i] * 2 * np.pi, 0.1)
        ax2.set_title(f'Frame: {i + 1}/{total_frames}')

        if save_video:
            plt.savefig("animation/file%02d.png" % i)
        else:
            plt.pause(0.01)

    if save_video:
        os.chdir("animation")
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'video_name.mp4'
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
