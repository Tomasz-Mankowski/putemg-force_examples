import numpy as np
from tqdm import tqdm

from typing import Dict, Tuple

import glob, pickle, os, sys


def usage():
    print()
    print('Usage: {:s} <result_folder> <output_file>'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <result_folder>          URL to a folder containing force learn classification results')
    print('    <output_file>            URL to a file for saving statistic data')
    print()
    print('Example:')
    print('{:s} ../putEMG/force_learn_results/ ../putEMG/force_learn_stats.bin'.format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) != 3:
        print('Illegal number of parameters')
        usage()

    results_url = os.path.abspath(sys.argv[1])
    if not os.path.isdir(results_url):
        print('Folder with force results does not exist - {:s}'.format(results_url))
        usage()

    output_url = os.path.abspath(sys.argv[2])

    all_files = [f for f in sorted(glob.glob(os.path.join(results_url, "*.bin")))]

    results: Dict[Tuple[str, str, str], Dict[str, any]] = dict()

    for file_url in tqdm(all_files, desc="Processing files"):# all_files:
        # print('Reading result file: {:s} ... '.format(os.path.basename(file_url)))
        data = pickle.load(open(file_url, "rb"))

        for d in data['results']:
            error = d['y_true'] - d['y_pred'].reshape(d['y_true'].shape)
            rmse = np.sqrt(np.mean(np.power(error,2)))
            std = np.std(error)

            key = (d['reg'], d['feature_set'], d['trajectory'])
            results.setdefault(key, {}).setdefault('rmse', []).append(rmse)
            results[key].setdefault('std', []).append(std)

        del data

    pickle.dump(results, open(output_url,"wb"))

