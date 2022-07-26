import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import data
import torch
from matplotlib import cm
import scipy.stats as st


def plot_video(frames=1000, save_video=False, hop=1):
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
    count = 0
    for i in range(0, hop * frames, hop):
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
        ax2.set_title(f'Frame: {i + 1}/{total_frames}, frame count = {count}')
        count += 1

        if save_video:
            plt.savefig("animation/file%02d.png" % i)
        else:
            plt.pause(1)

    if save_video:
        os.chdir("animation")
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'video_name.mp4'
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)


def plot_representations(xs, e_net, t_net, epoch, HP, sixth_image):
    xs = torch.cat((xs, sixth_image), dim=0)
    e_seq = e_net(xs)
    e_t_seq = t_net(e_seq)
    plt.plot(range(len(e_seq)), e_seq.detach().cpu(), '.', ms=12, label=r'$Z(x_i)$')
    plt.plot(range(1, len(e_t_seq) + 1), e_t_seq.detach().cpu(), 'r+', ms=12, label=r'$T(Z(x_{i-1}))$')
    plt.legend()
    # plt.ylim(-3, 3)
    plt.title(f'epoch={epoch}')
    plt.show()


def plot_average_anomaly(hop=30, network=''):

    if network == 'FC':
        network = '_FC'

    plt.figure(figsize=(12, 4))
    scores = np.load('results/anomaly_scores'+network+'_hop'+str(hop)+'.npy')
    mouse_positions = np.load('results/mouse_positions'+network+'_hop'+str(hop)+'.npy')

    scores_mean = np.nanmean(np.array(scores), axis=0)
    # scores = gaussian_filter1d(scores, 10)

    scores_sem = 1.96 * (st.sem(np.array(scores), axis=0))

    range_idx = range(len(scores_mean))
    plt.plot(scores_mean, color='b', label='score')
    plt.fill_between(range_idx, scores_mean - scores_sem,
                     scores_mean + scores_sem, color='b', alpha=0.4)

    plt.plot(range_idx, mouse_positions, color='r', label='mouse angle')

    plt.ylabel('Anomaly score', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlabel('Frame count (hop='+str(hop)+')', fontsize=16)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.title(f'Anomaly_scores{network}_hop{hop}', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/Anomaly_scores_video_data'+network+'_hop'+str(hop)+'.png')
    # plt.show()


def plot_average_anomaly_polar(network='', hop=30):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    if network == 'FC':
        network = '_FC'

    scores = np.load('results/anomaly_scores'+network+'_hop'+str(hop)+'.npy')
    mouse_positions = np.load('results/mouse_positions'+network+'_hop'+str(hop)+'.npy')

    scores_mean = np.mean(np.array(scores), axis=0)


    ax.plot(mouse_positions, scores_mean)
    ax.grid(True)
    # ax.set_rmax(20)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line

    # scores_sem = 1.96 * (st.sem(np.array(scores), axis=0))
    #
    # range_idx = range(len(scores_mean))
    # plt.plot(scores_mean, color='b', label='score')
    # plt.fill_between(range_idx, scores_mean - scores_sem,
    #                  scores_mean + scores_sem, color='b', alpha=0.4)
    #
    # plt.plot(range_idx, mouse_positions, color='r', label='mouse angle')

    # plt.ylabel('Anomaly score', fontsize=16)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Frame count (hop=20)', fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.legend()
    plt.title(f'Anomaly_scores{network}_hop{hop}', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/Anomaly_scores_video_data_polar'+network+'_hop'+str(hop)+'.png')
    # plt.show()


def mean_anomaly_score_vs_angle(net=''):
    hops = [15, 20, 25, 30] # TODO: This choice matters.
    if net == 'FC':
        net = '_FC'
    total_scores = []
    total_positions = []
    for hop in hops:
        scores = np.load('results/anomaly_scores' + net + '_hop' + str(hop) + '.npy')
        scores_mean = np.mean(np.array(scores), axis=0)
        total_scores += list(scores_mean)

        mouse_positions = np.load('results/mouse_positions' + net + '_hop' + str(hop) + '.npy')
        total_positions += list(mouse_positions)


    total_scores = np.array(total_scores)
    total_positions = np.array(total_positions)
    total_positions[total_positions < 0] =  total_positions[total_positions < 0] + 2 * np.pi # TODO: This choice matters.

    n_bins = 15 # TODO: This choice matters.
    bins = np.arange(0,2*np.pi,2*np.pi/n_bins)
    bined_positions = np.digitize(total_positions, bins)
    bin_centers = bins+np.diff(np.concatenate((bins,np.array([2*np.pi]))))/2
    mean_anomalies = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        mean_anomalies[i] = np.nanmean(total_scores[np.where(bined_positions == i+1)[0]])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(bin_centers, mean_anomalies,'.', ms=12)
    ax.grid(True)
    ax.set_rlim(0,np.max(mean_anomalies)+0.1*np.max(mean_anomalies))
    plt.tight_layout()
    plt.show()
    pass

if __name__ == '__main__':

    mean_anomaly_score_vs_angle()


    # hops = [5,10,15,20,25,30]
    # nets = ['FC','']
    # for hop in hops:
    #     for net in nets:
    #         print(hop, net)
    #         plot_average_anomaly_polar(hop=hop, network=net)
    #         plot_average_anomaly(hop=hop, network=net)

    # plot_video(200, save_video=False, hop=20)
