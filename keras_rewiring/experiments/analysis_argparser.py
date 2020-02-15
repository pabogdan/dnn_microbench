import argparse

DEFAULT_FIGURE_DIR = 'experiment_results/'

analysis_parser = argparse.ArgumentParser(
    description='Analyse a cerebellar simulation run on SpiNNaker or  NEST.',
    formatter_class=argparse.RawTextHelpFormatter)

analysis_parser.add_argument('-i', '--input', type=str, nargs="*",
                             help="name(s) of the csv/npz/hdf5 archive storing "
                                  "the results from running the simulations",
                             dest='input')

analysis_parser.add_argument('--results_dir', type=str,
                             help='directory into which to save figures and '
                                  'other results',
                             default=DEFAULT_FIGURE_DIR)

analysis_parser.add_argument('-c', '--compare',
                             action="store_true",
                             help='if this flag is present all of the different'
                                  ' results will be compared against '
                                  'each other',
                             dest='compare')

analysis_args = analysis_parser.parse_args()
