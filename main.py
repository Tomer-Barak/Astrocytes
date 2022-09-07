import numpy as np
import matplotlib.pyplot as plt
import torch
import plots, data, networks, training
import time


def create_nets(HP):
    z_net = networks.old_Z(HP).cuda()
    t_net = networks.T(HP).cuda()

    return z_net, t_net


def anomaly_detection(HP):
    vid, mouse_position = data.generate_video_data(HP)

    z_net, t_net = create_nets(HP)

    mse_loss = torch.nn.MSELoss()

    anomaly_scores = []
    mouse_positions = []

    if HP['vid_length'] == 'full':
        N = len(vid) - HP['K']
    else:
        N = HP['vid_length']

    for i in range(N):  # len(vid) - HP['K']):
        start = time.time()

        xs = vid[i:i + HP['K']].float().cuda()
        sixth_image = torch.unsqueeze(vid[i + HP['K']], dim=0).float().cuda()

        if HP['reset_nets']:
            z_net, t_net = create_nets(HP)

        z_net, t_net, loss_train_vector, e_i_ip = training.optimization(xs, z_net, t_net, HP, sixth_image)

        if HP['RN']:
            z_net(xs[-1])
            cat_input = torch.cat((z_net(xs[-1]), z_net(sixth_image)), dim=1)
            label = torch.tensor(1.)
            e_K_c = torch.square((t_net(cat_input) - label)).item()
        else:
            e_K_c = mse_loss(t_net(z_net(xs[-1])), z_net(sixth_image)).item()

        # if np.std(e_i_ip) == 0 and (np.mean(e_i_ip) - e_K_c) == 0:
        #     anomaly_scores.append(0.0)
        # elif np.std(e_i_ip) == 0 and (np.mean(e_i_ip) - e_K_c) != 0:
        #     anomaly_scores.append(5.0)
        # else:
        anomaly_scores.append((e_K_c - np.mean(e_i_ip)) / np.std(e_i_ip))
        mouse_positions.append(mouse_position[i])

        # print(f"i={i}/{N}, Score={np.round(anomaly_scores[-1], 2)},"
        #       f" Time={np.round(time.time() - start, 2)}, e_K_c={np.round(e_K_c, 4)}, e_i_ip={np.round(np.mean(e_i_ip), 4)}")
    #
    # plt.plot(anomaly_scores)
    # plt.show()

    return anomaly_scores, mouse_positions


def multiple_anomaly_tests():
    HP = {'lr': 4e-4, 'Z_dim': 1, 'grid_size': 100, 'K': 5, 'hop': 20, 'RN': False, 'plot_representations': False,
          'gather_errors': True, 'epochs': 1, 'optim': 'RMSprop', 'reset_nets': True, 'iterations': 20,
          'vid_length': 200}

    print(HP)
    scores = []
    for iteration in range(HP['iterations']):
        time_start = time.time()
        anomaly_scores, mouse_position = anomaly_detection(HP)
        scores.append(anomaly_scores)
        np.save('results/anomaly_scores.npy', scores)
        np.save('results/mouse_positions.npy', mouse_position)
        print(f'Iteration {iteration}, time={time.time()-time_start}')


if __name__ == '__main__':
    multiple_anomaly_tests()
