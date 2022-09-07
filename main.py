import numpy as np
import matplotlib.pyplot as plt

import plots, data, networks


def create_nets(HP):

    z_net = networks.old_Z(HP).cuda()
    t_net = networks.T(HP).cuda()

    return z_net, t_net


def main(HP):
    bad_ROIs, df_F, events_above_min, include_frames, quad_data_norm, scaled_centroids = data.load_data()

    # plots.plot_video(bad_ROIs, df_F, events_above_min, include_frames, quad_data_norm, scaled_centroids,
    #                  save_video=False)

    z_net, t_net = create_nets(HP)


if __name__ == '__main__':
    HP = {'lr': 4e-4, 'Z_dim':1,}

    main(HP)
