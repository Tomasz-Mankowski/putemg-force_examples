#!/usr/bin/env python3

import os
import sys
import glob

import pandas as pd

import putemg_features
from putemg_features import biolab_utilities


def usage():
    print()
    print('Calculates signal features')
    print()
    print('Usage: {:s} <feature_config_xml> <input_hdf5_folder> <output_hdf5_folder>'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <feature_config_xml>     XML file containing feature descriptors')
    print('    <input_hdf5_folder>             putEMG HDF5 folder containing experiment data')
    print("    <output_hdf5_folder>            output HDF5 folder containing calculated feature data")
    print()
    print('Example:')
    print('{:s} features_features.xml '
          '../putEMG/Data-HDF5-filtered '
          '../putEMG/Data-HDF5-filtered-feature'.
          format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) != 4:
        print('Illegal number of parameters')
        usage()

    xml_file_url = os.path.abspath(sys.argv[1])
    if not os.path.isfile(xml_file_url):
        print('XML file with feature descriptors does not exist - {:s}'.format(xml_file_url))
        usage()

    input_folder = os.path.abspath(sys.argv[2])
    if not os.path.isdir(input_folder):
        print('{:s} is not a valid folder'.format(input_folder))
        usage()

    output_folder = os.path.abspath(sys.argv[3])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # list all hdf5 files in given input folder
    all_files = [f for f in sorted(glob.glob(os.path.join(input_folder, "*.hdf5")))]

    for file in all_files:
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]

        print('Calculating features for {:s} file'.format(filename))

        record: pd.DataFrame = pd.read_hdf(os.path.join(input_folder, basename))  # Read HDF5 file into pandas DataFrame

        record_after_force: pd.DataFrame = biolab_utilities.force_features_in_window(record, 256, 32)

        # for filtered data file run feature extraction, use xml with limited feature set
        ft: pd.DataFrame = putemg_features.features_from_xml_on_df(xml_file_url, record_after_force)

        # save extracted features file to designated folder with features_filtered_ prefix
        output_file = filename + '_features.hdf5'
        print('Saving result to {:s} file'.format(output_file))
        ft.to_hdf(os.path.join(output_folder, output_file),
                  'data', format='table', mode='w', complevel=5)
