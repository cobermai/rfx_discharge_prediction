import numpy as np
import matplotlib.pyplot as plt
'''
plots a ramp-up with analog in and all channels specified in add_channels
ramp_up_lst = list of tuples of form [(ramp_up_id, reconstruction_loss), ...]
expand_drawing = how much before and after the min/max of the ramp up should be plotted
'''
def plot_pairs(analog_in_all, ec_d, min_max_pairs,tc_id, ramp_up_lst, expand_drawing=5000):
    analog_in_comb = analog_in_all[tc_id]
    cut_tc1tc2(tc_id, analog_in_comb)
    min_max_pair = min_max_pairs[tc_id]
    data_all = []
    add_channels = ['IM_MIS', 'IP_MIS', 'ITR_MIS', 'SCINT_MIS']

    for channel in add_channels:
        temp = []
        for ec in ec_d[tc_id]:
            temp.append(ec["ANALOG_IN"][channel][0])
        data_all.append(np.concatenate(temp, axis=0))

    for pair in ramp_up_lst:
        a_id = pair[0]
        start = min_max_pair[a_id][0]
        end = min_max_pair[a_id][1]

        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("{id} (loss: {val})".format(id=a_id, val=pair[1]), fontsize=18, y=0.95)
        plt.rc('font', size=15)
        num_plots = len(add_channels)+1
        ax = plt.subplot(num_plots, 1, 1)

        ax.set_title("Analog IN")
        diff = end-start
        plt.plot(np.arange(- expand_drawing, diff + expand_drawing, 1), analog_in_comb[start - expand_drawing:end + expand_drawing],  linewidth=4)
        plt.plot(0, analog_in_comb[start], "x", color="blue", mew=5, ms=10)
        plt.plot(diff, (analog_in_comb[end]), "x", color="red", mew=5, ms=10)
        plt.ylabel("kV")
        plt.xlabel("total seconds since ramp-up start")
        for count, channel in enumerate(add_channels):
            ax = plt.subplot(num_plots, 1, 2+count)
            ax.set_title("{name}".format(name=channel))

            data = data_all[count]
            plt.plot(np.arange(- expand_drawing, diff + expand_drawing, 1),
                     data[(start - expand_drawing):(end + expand_drawing)],  linewidth=4)
            plt.plot(0, data[start], "x", color="blue", mew=5, ms=10)
            plt.plot(diff, (data[end]), "x", color="red", mew=5, ms=10)

            # labels for axis
            if channel == 'IM_MIS' or channel == 'IP_MIS':
                plt.ylabel(r'$\mu A$')
            elif channel == 'ITR_MIS':
                plt.ylabel('mbar')
            elif channel == 'SCINT_MIS':
                plt.ylabel(r'$\mu SV/h$')
            plt.xlabel("total seconds since ramp-up start")
        plt.savefig("plots/tc{tc_id}/{id}.png".format(tc_id=tc_id, id=a_id))


'''
plots the analog_in signal, the reconstruction loss and beginning and end of ramp-ups.
Reconstruction loss is scaled on the highest value of analog_in
'''
def plot_analog_in_with_rloss(analog_in_all, min_max_pairs, anamoly_lst, r_loss, tc_id):
    file_name = "analog_in_rloss"
    min_max_pair = min_max_pairs[tc_id]
    analog_in = analog_in_all[tc_id]
    analog_in = cut_tc1tc2(tc_id, analog_in)
    plt.figure(figsize=(25, 10))
    plt.plot(analog_in, label="analog in")

    for a_id in anamoly_lst:
        start = min_max_pair[a_id][0]
        end = min_max_pair[a_id][1]
        plt.plot(start, (analog_in[start]), "x", color="blue", mew=5, ms=10)
        plt.plot(end, analog_in[end], "x", color="red", mew=5, ms=10)

        plt.annotate("l:{:.2f}".format((float(r_loss.values[a_id][0]))), (end, analog_in[end]), xytext=(end, analog_in[end] + 25),
                             textcoords='data')


    id_r_loss = []
    for v in r_loss.index:
        id_r_loss.append(min_max_pair[v][0]+(int)((min_max_pair[v][1]-min_max_pair[v][0])/2))

    #rescale loss
    r_loss = ((r_loss - np.min(r_loss)) / (np.max(r_loss) - np.min(r_loss)))*np.max(analog_in)

    plt.plot(id_r_loss,r_loss.values, label = "reconstruction loss", linewidth=3, color='orange')
    plt.ylabel("kV")
    plt.xlabel("seconds")
    plt.legend()
    plt.savefig("{fname}_{id}_tc3_1.png".format(fname = file_name, id = 2))

'''
counts values above thresh_val and plots the length of the plateau (=number of measurements above thresh_val)
'''
def plot_plateau_len(analog_in_all, min_max_pairs_all, thresh_val=0.985):
    plateau_lengths = []
    for tc_id in range(4):
        print(tc_id)
        tc_p_len = []
        analog_in_comb = analog_in_all[tc_id]
        analog_in_comb = cut_tc1tc2(tc_id, analog_in_comb)
        min_max_pairs = min_max_pairs_all[tc_id]

        for c, pair in enumerate(min_max_pairs):
            start = pair[0]
            end = pair[1]
            ramp_up = analog_in_comb[start:end]
            thresh = np.max(ramp_up)*thresh_val
            p_len = np.sum(ramp_up >= thresh)
            tc_p_len.append(p_len)

        plateau_lengths.append(tc_p_len)
    plt.figure(figsize=(9, 16))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Plateau lengths", fontsize=18, y=0.95)
    ax1 = plt.subplot(4, 1, 1)
    ax1.set_title("tc1")
    bins_tc1 = plt.plot(plateau_lengths[0])
    ax2 = plt.subplot(4, 1, 2)
    ax2.set_title("tc2")
    bins_tc2 = plt.plot(plateau_lengths[1])
    ax3 = plt.subplot(4, 1, 3)
    ax3.set_title("tc3")
    bins_tc3 = plt.plot(plateau_lengths[2])
    ax4 = plt.subplot(4, 1, 4)
    ax4.set_title("tc4")
    bins_tc4 = plt.plot(plateau_lengths[3])
    xlim = (min(ax1.get_xlim()[0], ax2.get_xlim()[0], ax3.get_xlim()[0], ax4.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1], ax3.get_xlim()[1], ax4.get_xlim()[1]))
    ylim = (min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0], ax4.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1], ax4.get_ylim()[1]))

    print(xlim, ylim)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 15}

    plt.rc('font', **font)

    plt.setp(ax1, xlabel="ramp-up #", ylabel="plateau length")
    plt.setp(ax2, xlabel="ramp-up #", ylabel="plateau length")
    plt.setp(ax3, xlabel="ramp-up #", ylabel="plateau length")
    plt.setp(ax4, xlabel="ramp-up #", ylabel="plateau length")

    plt.setp(ax1, xlim=xlim, ylim=ylim)
    plt.setp(ax2, xlim=xlim, ylim=ylim)
    plt.setp(ax3, xlim=xlim, ylim=ylim)
    plt.setp(ax4, xlim=xlim, ylim=ylim)
    plt.savefig("plots/plateau/plateau_len_plot.png")


    total_min = min(np.min(plateau_lengths[0]), np.min(plateau_lengths[1]), np.min(plateau_lengths[2]), np.min(plateau_lengths[3]))
    total_max = max(np.max(plateau_lengths[0]), np.max(plateau_lengths[1]), np.max(plateau_lengths[2]),
                    np.max(plateau_lengths[3]))

    binwidth = abs(total_max-total_min)//150
    bins = np.arange(total_min, total_max + binwidth, binwidth)
    plt.figure(figsize=(9, 16))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Plateau lengths", fontsize=18, y=0.95)
    ax1 = plt.subplot(4, 1, 1)
    ax1.set_title("tc1")
    bins_tc1 = plt.hist(plateau_lengths[0], bins=bins)
    ax2 = plt.subplot(4, 1, 2)
    ax2.set_title("tc2")
    bins_tc2 = plt.hist(plateau_lengths[1], bins=bins)
    ax3 = plt.subplot(4, 1, 3)
    ax3.set_title("tc3")
    bins_tc3 = plt.hist(plateau_lengths[2], bins=bins)
    ax4 = plt.subplot(4, 1, 4)
    ax4.set_title("tc4")
    bins_tc4 = plt.hist(plateau_lengths[3], bins=bins)
    xlim = (min(ax1.get_xlim()[0], ax2.get_xlim()[0], ax3.get_xlim()[0], ax4.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1], ax3.get_xlim()[1], ax4.get_xlim()[1]))
    ylim = (min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0], ax4.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1], ax4.get_ylim()[1]))

    print(xlim, ylim)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 15}

    plt.rc('font', **font)

    plt.setp(ax1, yscale="log", xscale="log", ylabel="# of ramp-ups", xlabel="plateau length")
    plt.setp(ax2, yscale="log", xscale="log", ylabel="# of ramp-ups", xlabel="plateau length")
    plt.setp(ax3, yscale="log", xscale="log", ylabel="# of ramp-ups", xlabel="plateau length")
    plt.setp(ax4, yscale="log", xscale="log", ylabel="# of ramp-ups", xlabel="plateau length")

    plt.setp(ax1, xlim=xlim, ylim=ylim)
    plt.setp(ax2, xlim=xlim, ylim=ylim)
    plt.setp(ax3, xlim=xlim, ylim=ylim)
    plt.setp(ax4, xlim=xlim, ylim=ylim)
    plt.savefig("plots/plateau/plateau_hist.png")

def plot_len(min_max_pairs_all):
    ramp_up_lengths = []
    for tc_id in range(4):
        tc_ramp_up_len = []
        min_max_pairs = min_max_pairs_all[tc_id]

        for pair in min_max_pairs:
            start = pair[0]
            end = pair[1]
            tc_ramp_up_len.append(end-start)

        ramp_up_lengths.append(tc_ramp_up_len)
    plt.figure(figsize=(9, 16))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Ramp-up lengths", fontsize=18, y=0.95)
    ax1 = plt.subplot(4, 1, 1)
    ax1.set_title("tc1")
    #bins_tc1 = plt.plot(ramp_up_lengths[0])
    plt.hist(ramp_up_lengths[0], 100)
    ax2 = plt.subplot(4, 1, 2)
    ax2.set_title("tc2")
    #bins_tc2 = plt.plot(ramp_up_lengths[1])
    plt.hist(ramp_up_lengths[1], 100)

    ax3 = plt.subplot(4, 1, 3)
    ax3.set_title("tc3")
    #bins_tc3 = plt.plot(ramp_up_lengths[2])
    plt.hist(ramp_up_lengths[2], 100)

    ax4 = plt.subplot(4, 1, 4)
    ax4.set_title("tc4")
    #bins_tc4 = plt.plot(ramp_up_lengths[3])
    plt.hist(ramp_up_lengths[3], 100)

    xlim = (min(ax1.get_xlim()[0], ax2.get_xlim()[0], ax3.get_xlim()[0], ax4.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1], ax3.get_xlim()[1], ax4.get_xlim()[1]))
    ylim = (min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0], ax4.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1], ax4.get_ylim()[1]))

    print(xlim, ylim)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    plt.rc('font', **font)

    plt.setp(ax1,yscale="log", xscale="log", xlim=xlim, ylim=ylim, ylabel="# of ramp-ups", xlabel="length")
    plt.setp(ax2,yscale="log", xscale="log", xlim=xlim, ylim=ylim, ylabel="# of ramp-ups", xlabel="length")
    plt.setp(ax3,yscale="log", xscale="log", xlim=xlim, ylim=ylim, ylabel="# of ramp-ups", xlabel="length")
    plt.setp(ax4,yscale="log", xscale="log", xlim=xlim, ylim=ylim, ylabel="# of ramp-ups", xlabel="length")
    plt.savefig("plots/len_histo_plt_neu.png")

'''
helper function to cut testcase 2/3 to conform more to the paper (testcase in code numbered from 0-3, in paper 1-4)
'''
def cut_tc1tc2(tc_id, a_in):
    if tc_id == 1:
        a_in = np.append(a_in[:19932690], a_in[20550000:37837690])
    if tc_id == 2:
        a_in = a_in[:-650000]
    return a_in

