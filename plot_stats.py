import numpy as np
import matplotlib.pyplot as plt

import pickle, os, sys


def usage():
    print()
    print('Usage: {:s} <stats_file>'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <stats_file>>            URL to a file containing statistic data of force learn')
    print()
    print('Example:')
    print('{:s} ../putEMG/force_learn_stats.bin'.format(os.path.basename(__file__)))
    exit(1)


def stats_using_key_partial_match(tuple_key, dictionary):
    rmse = []
    std = []
    for key, value in dictionary.items():
        if all(k1 == k2 or k2 is None for k1, k2 in zip(key, tuple_key)):
            rmse += value['rmse']
            std += value['std']
    return rmse, std


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) != 2:
        print('Illegal number of parameters')
        usage()

    stats_url = os.path.abspath(sys.argv[1])
    if not os.path.isfile(stats_url):
        print('File containing statistics does not exist - {:s}'.format(stats_url))
        usage()

    results = pickle.load(open(stats_url, "rb"))

    result_elements = list(zip(*results.keys()))
    regressors = list(set(result_elements[0]))
    feature_sets = list(set(result_elements[1]))
    trajectories = list(set(result_elements[2]))

    figs = list()
    axes = list()

    for trajectory in trajectories:
        fig, ax = plt.subplots(num=trajectory)

        figs.append(fig)
        axes.append(ax)

        bar_width = 0.25
        bar_spacer = 0.04

        for r_index, reg in enumerate(regressors):

            rmse_means = []
            std_means = []

            for f_set in feature_sets:
                rmse_v, std_v = stats_using_key_partial_match((reg, f_set, trajectory), results)

                rmse_means.append(np.mean(rmse_v))
                std_means.append(np.mean(std_v))

            print('rmse', rmse_means)
            print('std', std_means)

            ax.bar(np.arange(len(feature_sets)) + (bar_width+bar_spacer) * r_index,  rmse_means, bar_width,
                   label=reg)
            ax.errorbar(np.arange(len(feature_sets)) + (bar_width+bar_spacer) * r_index, rmse_means,
                        fmt='ko', ecolor='k', lw=2, capsize=5,
                        yerr=np.array(std_means))

    for f, a in zip(figs, axes):
        a.legend()

        a.set_xticks(np.arange((len(feature_sets))) + ((bar_width + bar_spacer) * (len(regressors) - 1)) / 2)
        a.set_xticklabels(feature_sets)

        a.set_ylim([0, 4])
        a.set_yticks(np.arange(5))

    plt.show()
