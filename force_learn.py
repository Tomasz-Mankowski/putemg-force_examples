#!/usr/bin/env python3

import os, sys, glob, pickle, time

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

    # defines feature sets to be used in force learn
    feature_sets = {
        "ZC": ["ZC"],
        "WAMP": ["WAMP"],
        "RMS": ["RMS"],
        "AAC": ["AAC"],
        "MNF": ["MNF"],
        "MNP": ["MNP"],
        "PKF": ["PKF"],
        "TimeDomain": ["ZC", "WAMP", "RMS", "AAC"],
        "FreqDomain": ["PKF", "MNF", "MNP"],
    }

    # defines the feature of force measurement to be used
    force_feature = "MEAN"

    # define fingers to be estimated in shallow learn
    trajectories = {
        "Index": [2],
        "Middle": [3],
        "Thumb": [1],
        "Ring+Small": [4],
        # "All": [1, 2, 3, 4],
    }

    # defines regressors and its options to be used in force learn
    regressors = {
        "LR": {
            "predictor": "LR",
            "args": {"fit_intercept": True, "normalize": False, "copy_X": True, "n_jobs": None}},
        "MLPR": {
            "predictor": "MLPR",
            "args": {"hidden_layer_sizes": (200, ), "activation": "logistic", "solver": "adam",
                     "alpha": 0.0001, "batch_size": "auto", "learning_rate": "constant", "learning_rate_init": 0.001,
                     "power_t": 0.5, "max_iter": 200, "shuffle": True, "random_state": None, "tol": 0.0001,
                     "verbose": False, "warm_start": False, "momentum": 0.9, "nesterovs_momentum": True,
                     "early_stopping": False, "validation_fraction": 0.1, "beta_1": 0.9, "beta_2": 0.999,
                     "epsilon": 1e-08, "n_iter_no_change": 10}},
        "SVR": {
           "predictor": "SVR",
           "args": {"kernel": "rbf", "degree": 3, "gamma": "scale", "coef0": 0.0, "tol": 0.001,
                    "C": 1.0, "epsilon": 0.1, "shrinking": True, "cache_size": 200, "verbose": False, "max_iter": -1}},
    }

    # defines channels configurations for which classification will be run
    channel_range = {"begin": 9, "end": 16}
    # "24chn": {"begin": 1, "end": 24},
    # "8chn_1band": {"begin": 1, "end": 8},
    # "8chn_2band": {"begin": 9, "end": 16},
    # "8chn_3band": {"begin": 17, "end": 24}

    # create k-fold validation set, with 3 splits - for each experiment day 3 combination are generated
    # this results in 6 data combination for each subject
    n_splits = 3

    print('Starting to learn how to Force...')

    all_files = [f for f in sorted(glob.glob(os.path.join(input_folder, "*.hdf5"))) if not ("bias" in f or "mvc" in f)]

    # create list of records
    all_feature_records = [biolab_utilities.Record(os.path.basename(f)) for f in all_files]

    # select unique ids list in order to load data for only a single subject (done due to Out of memory problems)
    unique_ids = sorted(set([record.id for record in all_feature_records]))

    for single_id in unique_ids:
        # Filter based on subject id
        records_filtered_by_subject = biolab_utilities.record_filter(all_feature_records,
                                                                     whitelists={"id": [single_id]})
        # load feature data to memory
        dfs: Dict[biolab_utilities.Record, pd.DataFrame] = {}
        for r in records_filtered_by_subject:
            print("\tReading features for input file: ", r)
            filename = os.path.splitext(r.path)[0]
            dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(input_folder, filename + '.hdf5')))

        # Create splits
        splits_all = biolab_utilities.data_per_id_and_date(records_filtered_by_subject, n_splits=n_splits)

        # for each experiment (single subject, single day)
        for id_, id_splits in splits_all.items():
            output: Dict[str, any] = dict()

            output["trajectories"] = trajectories
            output["regressors"] = regressors
            output["feature_sets"] = feature_sets
            output["id"] = id_
            output["results"]: List[Dict[str, any]] = list()

            print('\tTrial ID: {:s}'.format(id_), flush=True)

            # for each finger trajectory
            for trajectory_name, trajectory in trajectories.items():
                print('\t\tFinger trajectory: {:s}'.format(trajectory_name), flush=True)

                # for split in k-fold validation of each day of each subject
                for i_s, s in enumerate(id_splits):
                    print('\t\t\tSplit: {:d}'.format(i_s), flush=True)

                    # for each feature set
                    for feature_set_name, features in feature_sets.items():
                        print('\t\t\t\tFeature set: {:s} -'.format(feature_set_name), end='', flush=True)

                        data = biolab_utilities.prepare_force_data(dfs, s, features, force_feature, trajectory)

                        # strip columns to include only selected channels, eg. only one band
                        band_columns = [c for c in data['train']['input'].columns if
                                        (channel_range["begin"] <= int(c[c.rindex('_') + 1:]) <= channel_range["end"])]

                        train_x = data['train']['input'][band_columns]
                        train_y = data['train']['output']

                        test_x = data['test']['input'][band_columns]
                        test_y_true = data['test']['output']

                        for reg_id, reg_settings in regressors.items():
                            print(' {:s}'.format(reg_id), end='', flush=True)

                            start = time.time()
                            # prepare regressor pipeline
                            # fit the regressor to train data
                            pipeline = biolab_utilities.prepare_pipeline(train_x, train_y,
                                                                         predictor=reg_settings["predictor"],
                                                                         norm_per_feature=False,
                                                                         **reg_settings["args"])
                            elapsed_fit = time.time() - start

                            start = time.time()
                            # run regressor on test data
                            test_y_pred = pipeline.predict(test_x)
                            elapsed_predict= time.time() - start

                            print(' (fit: {:.1f}s pred: {:.1f}s)'.format(elapsed_fit, elapsed_predict), end='', flush=True)

                            # save classification results to output structure
                            output["results"].append({"split": i_s, "reg": reg_id, "trajectory": trajectory_name,
                                                      "force_feature": force_feature, "feature_set": feature_set_name,
                                                      "y_true": test_y_true.values.astype(float),
                                                      "y_pred": test_y_pred})
                        print()

            # Dump regression results to file
            filename = "force-classification-result_" + id_.replace("/", "_") + ".bin"
            print('\tWriting trial file: {:s}'.format(filename), flush=True)
            print()
            pickle.dump(output, open(os.path.join(result_folder, filename), "wb"))

            # Free memory of each output, writing next data to a new file
            del output
        # Free memory of input feature data, new data for next subject will be loaded
        del dfs
