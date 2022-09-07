import torch
import numpy as np
import torch.optim as optim
import itertools as it
import torch.nn as nn
import plots
import matplotlib.pyplot as plt


def CPCLoss(e_t_0, e0, e1):
    all_states = torch.cat((e0, e1[-1].unsqueeze(0)), dim=0)

    loss = 0
    for i in range(len(e_t_0)):
        # almost_all_states = torch.cat([all_states[:i+1], all_states[i + 2:]])
        # loss += -torch.log(torch.exp(-torch.norm(e_t_0[i] - e1[i]) ** 2) / torch.sum(
        #     torch.exp(-torch.norm(e_t_0[i] - all_states) ** 2)))
        loss += -torch.log(torch.exp(-(torch.norm(e_t_0[i] - e1[i]) ** 2)) / torch.sum(
            torch.exp(-(torch.norm(e_t_0[i] - all_states, dim=1) ** 2))))
    loss /= len(e_t_0)

    return loss


def optimization(xs, e_net, t_net, HP, sixth_image):
    #  Create the networks and optimizers
    if HP['optim'] == 'RMSprop':
        optimizer = optim.RMSprop(filter(lambda h: h.requires_grad,
                                         it.chain(e_net.parameters(), t_net.parameters())), lr=HP['lr'])
    elif HP['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda h: h.requires_grad,
                                     it.chain(e_net.parameters(), t_net.parameters())), lr=HP['lr'])
    elif HP['optim'] == 'Adam':
        optimizer = optim.Adam(filter(lambda h: h.requires_grad,
                                      it.chain(e_net.parameters(), t_net.parameters())), lr=HP['lr'])


    mse_loss = nn.MSELoss()

    loss_train_vector = []
    e_i_ip = []
    e_i_j = []

    for j in range(HP['epochs']):

        e_net.train()
        t_net.train()

        optimizer.zero_grad()

        e0 = e_net(xs[:-1])
        e1 = e_net(xs[1:])

        if HP['RN']:

            gnet_inputs = torch.cat((e0, torch.unsqueeze(e1[-1], dim=0)), dim=0)
            loss = torch.tensor(0)
            for ginput_i in range(len(gnet_inputs)):
                for ginput_j in range(len(gnet_inputs)):
                    cat_input = torch.cat((gnet_inputs[ginput_i], gnet_inputs[ginput_j]), dim=0).unsqueeze(dim=0)
                    if ginput_j == ginput_i + 1:
                        label = torch.tensor(1.)
                    else:
                        label = torch.tensor(0.)
                    loss = loss + torch.square((t_net(cat_input) - label))
            loss = loss / (len(gnet_inputs) ** 2)

        if not HP['RN']:
            e_t_0 = t_net(e0)
            loss = CPCLoss(e_t_0, e0, e1)

        loss.backward()
        optimizer.step()

        loss_train_vector.append(loss.item())

        if HP['plot_representations']:
            plots.plot_representations(xs, e_net, t_net, j, HP, sixth_image)
            plt.pause(0.5)
            plt.clf()

    if HP['gather_errors']:

        e0 = e_net(xs[:-1])
        e1 = e_net(xs[1:])

        if HP['RN']:
            gnet_inputs = torch.cat((e0, torch.unsqueeze(e1[-1], dim=0)), dim=0)
            for ginput_i in range(len(gnet_inputs)-1):
                cat_input = torch.cat((gnet_inputs[ginput_i], gnet_inputs[ginput_i+1]), dim=0).unsqueeze(dim=0)
                label = torch.tensor(1.)
                e_i_ip.append(torch.square((t_net(cat_input) - label)).item())

        else:
            e_t_0 = t_net(e0)
            for et0_, e1_ in zip(e_t_0, e1):
                e_i_ip.append(mse_loss(et0_, e1_).item())


    loss_train_vector = np.array(loss_train_vector)

    return e_net, t_net, loss_train_vector, e_i_ip



if __name__ == '__main__':
    pass
