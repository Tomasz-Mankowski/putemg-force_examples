#!/usr/bin/env python3

import os
import sys
import glob

import pandas as pd

from putemg_features import biolab_utilities


def usage():
    print()
    print("Applies denoising filter")
    print()
    print("Usage: {:s} <input_hdf5_folder> <output_hdf5_folder>".format(os.path.basename(__file__)))
    print("     <input_hdf5_folder>:  putEMG HDF5 file folder containing raw experiment data")
    print("     <output_hdf5_folder>: output HDF5 file folder for filtered data")
    print()
    print("Example:")
    print("{:s} ../putEMG/Data-HDF5 ../putEMG/Data-HDF5-filtered".
          format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) != 3:
        print("Invalid parameter count")
        usage()

    putemg_folder = sys.argv[1]
    output_folder = sys.argv[2]

    all_files = [f for f in sorted(glob.glob(os.path.join(putemg_folder, "*.hdf5")))]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in all_files:
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]
        print('Denoising file: {:s}'.format(basename))

        # read raw putEMG data file and run filter
        df: pd.DataFrame = pd.read_hdf(file)
        biolab_utilities.apply_filter(df)

        output_file = filename + '_filtered.hdf5'
        print('Saving to file: {:s}'.format(output_file))
        df.to_hdf(os.path.join(output_folder, output_file), 'data', format='table', mode='w', complevel=5)
