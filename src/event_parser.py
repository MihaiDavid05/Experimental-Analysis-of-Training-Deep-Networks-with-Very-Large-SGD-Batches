import argparse
from utils.utils import get_events_data


def get_args():
    """
    Function used for parsing the command line arguments

    Returns: Command line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('log_config_name', type=str, help='Config folder under logs.')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    log_path = f"logs/{args.log_config_name}"
    get_events_data(log_path, False, True, log_path, to_watch='_lr_acc')
    get_events_data(log_path + "/Loss_train", False, True, log_path, to_watch='_loss_train')
    get_events_data(log_path + "/Loss_val", False, True, log_path, to_watch='_loss_val')