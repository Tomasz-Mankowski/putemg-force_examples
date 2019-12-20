#!/usr/bin/env python3

import os
import sys
import glob
from typing import List, Dict

import pandas as pd

from putemg_features import biolab_utilities


def usage():
    print()
    print('Usage: {:s} <putEMG_HDF5_feature_folder> <output_folder>'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <putEMG_HDF5_feature_folder>     URL to a folder containing HDF5 files with features')
    print('    <output_folder>                  URL to a output folder - results and intermediate '
          'files will be written here')
    print()
    print('Example:')
    print('{:s} ../putEMG/Data-HDF5-filtered-feature ../putEMG/force_learn_results/'.format(os.path.basename(__file__)))
    exit(1)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()

    if len(sys.argv) < 3:
        print('Illegal number of parameters')
        usage()

    input_folder = os.path.abspath(sys.argv[1])
    result_folder = os.path.abspath(sys.argv[2])

    if not os.path.isdir(input_folder):
        print('{:s} is not a valid folder'.format(input_folder))
        exit(1)

    if not os.path.isdir(result_folder):
        print('{:s} is not a valid folder'.format(result_folder))
        exit(1)

    all_files = [f for f in sorted(glob.glob(os.path.join(input_folder, "*.hdf5"))) if not ("bias" in f or "mvc" in f)]

    # create list of records
    all_feature_records = [biolab_utilities.Record(os.path.basename(f)) for f in all_files]

    # data can be additionally filtered based on subject id
    records_filtered_by_subject = biolab_utilities.record_filter(all_feature_records)
    # records_filtered_by_subject = record_filter(all_feature_records,
    #                                             whitelists={"id": ["01", "02", "03", "04", "07"]})
    # records_filtered_by_subject = pu.record_filter(all_feature_records, whitelists={"id": ["01"]})

    # load feature data to memory
    dfs: Dict[biolab_utilities.Record, pd.DataFrame] = {}
    for r in records_filtered_by_subject:
        print("Reading features for input file: ", r)
        filename = os.path.splitext(r.path)[0]
        dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(input_folder, filename + '.hdf5')))

    # create k-fold validation set, with 3 splits - for each experiment day 3 combination are generated
    # this results in 6 data combination for each subject
    splits_all = biolab_utilities.data_per_id_and_date(records_filtered_by_subject, n_splits=3)

