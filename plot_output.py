import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import gridspec

from typing import Dict, List

import click
from tqdm import tqdm

import pickle
import os
import glob


@click.command()
@click.argument('result_folder', type=click.Path(exists=True))
@click.option('-r', '--regressor', 'regressor', type=str, required=True)
@click.option('-t', '--trajectory', 'trajectory', type=str, required=True)
@click.option('-f', '--feature-set', 'feature_set', type=str, required=True)
def cls(result_folder, regressor, trajectory, feature_set):
    """Displays the results of force learn with a given combination of trajectory, regressor and feature set"""

    results_url = os.path.abspath(result_folder)
    all_files = [f for f in sorted(glob.glob(os.path.join(results_url, "*.bin")))]

    results: List[Dict[str, any]] = list()

    for file_url in tqdm(all_files, desc='Reading force learn data to memory'):
        data = pickle.load(open(file_url, "rb"))

        for d in data['results']:
            if d['reg'] == regressor and d['feature_set'] == feature_set and  d['trajectory'] == trajectory:
                results.append({})
                results[-1]['y_pred'] = d['y_pred'].reshape(d['y_true'].shape)
                results[-1]['y_true'] = d['y_true']
                results[-1]['error'] = d['y_true'] - results[-1]['y_pred']
                results[-1]['split'] = d['split']
                results[-1]['id'] = os.path.splitext(os.path.basename(file_url))[0][-13:]
                results[-1]['rmse'] = np.sqrt(np.mean(np.power(results[-1]['error'], 2)))
                results[-1]['std'] = np.std(results[-1]['error'])

        del data

    if len(results) > 0:
        fig = plt.figure('Regressor: ' + regressor + ', Feature set: ' + feature_set + ', Trajectory: ' + trajectory,
                         figsize=(15, 9))

        gs = gridspec.GridSpec(3, 1, height_ratios=[7, 2, 0.2])

        main_ax = plt.subplot(gs[0])
        error_ax = plt.subplot(gs[1], sharex=main_ax)
        slider_ax = plt.subplot(gs[2])

        main_plot = main_ax.plot(np.hstack((results[0]['y_pred'], results[0]['y_true'])))
        error_plot = error_ax.plot(results[0]['error'])

        res_slider = Slider(slider_ax, 'IDs/Splits:', 0, len(results)-1, valinit=0, valstep=1)

        output_size = results[0]['y_pred'].shape[1]

        main_ax.set_ylim(-0.2, 0.7)
        error_ax.set_ylim(-1, 1)

        if output_size == 1:
            main_ax.legend([trajectory + ' PREDICTION', trajectory + ' TRUE'])
            error_ax.legend([trajectory + ' ERROR'])
        else:
            legend = [str(i) + ' PREDICTION' for i in range(output_size)]
            legend += [str(i) + ' TRUE' for i in range(output_size)]
            main_ax.legend(legend, loc='upper right')

            legend = [str(i) + ' ERROR' for i in range(output_size)]
            error_ax.legend(legend, loc='upper right')

        def set_title(id, split, rmse, std):
            fig.suptitle(id + " - SPLIT " + str(split) + " ------ RMSE: " +
                         str(round(rmse, 3)) + " STDE: " + str(round(std, 3)))

        set_title(results[0]['id'], results[0]['split'], results[0]['rmse'], results[0]['std'])

        fig.subplots_adjust(0.08, 0.02, 0.95, 0.95)

        def update(res):
            for i in range(output_size):
                error_plot[i].set_data([np.arange(len(results[int(res)]['error'])),
                                        results[int(res)]['error'][:, i]])

                main_plot[i].set_data([np.arange(len(results[int(res)]['y_pred'])),
                                       results[int(res)]['y_pred'][:, i]])

                main_plot[i + output_size].set_data([np.arange(len(results[int(res)]['y_true'])),
                                                     results[int(res)]['y_true'][:, i]])

            set_title(results[int(res)]['id'], results[int(res)]['split'],
                      results[int(res)]['rmse'], results[int(res)]['std'])

            fig.canvas.draw_idle()

        def key_press_event(event):
            if event.key == 'left':
                if res_slider.val > 0:
                    res_slider.set_val(res_slider.val - 1)
            elif event.key == 'right':
                if res_slider.val < res_slider.valmax:
                    res_slider.set_val(res_slider.val + 1)

        res_slider.on_changed(update)
        key_presser = fig.canvas.mpl_connect('key_press_event', key_press_event)

        plt.show()
    else:
        print()
        print('No results with matching trajectory, regressor and feature set were found')


if __name__ == '__main__':
    cls()