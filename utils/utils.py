import json
import torch
import glob
import os
import pprint
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def setup(args):
    """
    Args:
        args: Command line arguments

    Returns: Configuration dictionary, device, logging directory and checkpoint directory

    """
    # Create configs, logs and checkpoints folder
    if not os.path.exists("configs/"):
        os.mkdir("configs/")
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    if not os.path.exists("checkpoints/"):
        os.mkdir("checkpoints/")

    config_path = 'configs/' + args.config_filename + '.json'
    # Get config
    with open(config_path) as json_config:
        config = json.load(json_config)

    config["name"] = args.config_filename

    # Set folder for logging
    log_dir = config["log_dir"] + '/' + config["name"]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Set folder for checkpoints
    checkpoints_dir = config["checkpoint_dir"] + '/' + config["name"]
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    config["checkpoint_dir"] = checkpoints_dir
    config["log_dir"] = log_dir

    # Check for cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device state: ', device)

    return config, device, log_dir, checkpoints_dir


# REF: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
def tflog2pandas(path: str) -> pd.DataFrame:
    """
    Convert single tensorflow log file to pandas DataFrame
    Args:
        path: path to tensorflow log file

    Returns: converted dataframe

    """
    default_size_guidance = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, default_size_guidance)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
    return runlog_data


def many_logs2pandas(event_paths):
    """
    Convert multiple tensorflow log file to pandas DataFrame
    Args:
        event_paths: path to tensorflow log directory

    Returns: converted dataframe

    """
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def get_events_data(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: str, to_watch: str):
    """
    This script exctracts variables from all logs from tensorflow event
    files ("event*"), writes them to Pandas and finally stores them a csv-file
    or pickle-file including all (readable) runs of the logging directory.

    Args:
        logdir_or_logfile: Path to a log directory or file
        write_pkl: Whether to write to pickle or not
        write_csv: Whether to write to csv or not
        out_dir: Output directory for csv/pickle
        to_watch: One of extensions: '_lr_acc' or '_loss_train' or '_loss_val'.
        Adds extension to csv/pickle name, depending on the event file read

    """
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        event_paths = glob.glob(os.path.join(logdir_or_logfile, "event*"))
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        pp.pprint("Found tensorflow logs to process:")
        pp.pprint(event_paths)
        all_logs = many_logs2pandas(event_paths)
        pp.pprint("Head of created dataframe")
        pp.pprint(all_logs.head())

        os.makedirs(out_dir, exist_ok=True)
        if write_csv:
            print("saving to csv file")
            out_file = os.path.join(out_dir, f"all_training_logs_in_one_file{to_watch}.csv")
            print(out_file)
            all_logs.to_csv(out_file, index=False)
        if write_pkl:
            print("saving to pickle file")
            out_file = os.path.join(out_dir, f"all_training_logs_in_one_file{to_watch}.pkl")
            print(out_file)
            all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")
