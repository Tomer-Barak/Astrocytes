import numpy as np
import matplotlib.pyplot as plt

import plots, data, networks


def create_nets(HP):

    z_net = networks.old_Z(HP).cuda()
    t_net = networks.T(HP).cuda()

    return z_net, t_net


def main(HP):
    video, mouse_position = data.generate_video_data(HP)

    z_net, t_net = create_nets(HP)


if __name__ == '__main__':
    HP = {'lr': 4e-4, 'Z_dim':1, 'grid_size':100}

    main(HP)
