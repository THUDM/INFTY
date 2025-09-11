import os
import argparse

from utils.toolkit import load_json, load_yaml
from trainer import train
from infty import plot as infty_plot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inftyopt', type=str, default='ogd', help='select the optimizer')
    parser.add_argument('--config', type=str, default='./exps/ogd.json', help='experiment config file path')
    parser.add_argument('--ckp_dir', type=str, default='./ckp', help='checkpoint directory')
    return parser.parse_args()

def main():
    cli_args = parse_args()
    optimizer_name = cli_args.inftyopt.lower()

    # Select the corresponding yaml path according to the optimizer name
    base_oprimizers = ['base']
    flat_landscape_optimizers = ['sam', 'gsam', 'looksam', 'gam', 'c_flat', 'c_flat_plus']
    gradient_bans_optimizers = ['zo_sgd', 'zo_sgd_sign', 'zo_sgd_conserve', 'zo_adam', 'zo_adam_sign', 'zo_adam_conserve', 'forward_grad']
    gradient_conflicts_optimizers = ['pcgrad', 'gradvac', 'cagrad', 'unigrad_fs', 'ogd']


    config_path = None
    if optimizer_name in base_oprimizers:
        pass
    elif optimizer_name in flat_landscape_optimizers:
        optimizer_yaml_path = "../infty_configs/flat_landscape"
        if f"{optimizer_name}.yaml" in os.listdir(optimizer_yaml_path):
            config_path = os.path.join(optimizer_yaml_path, f"{optimizer_name}.yaml")
            print(f"Loading optimizer config: {config_path}")
    elif optimizer_name in gradient_conflicts_optimizers:
        optimizer_yaml_path = "../infty_configs/gradient_conflicts"
        if f"{optimizer_name}.yaml" in os.listdir(optimizer_yaml_path):
            config_path = os.path.join(optimizer_yaml_path, f"{optimizer_name}.yaml")
            print(f"Loading optimizer config: {config_path}")
    elif optimizer_name in gradient_bans_optimizers:
        optimizer_yaml_path = "../infty_configs/gradient_bans"
        config_path = os.path.join(optimizer_yaml_path, "zeroflow.yaml")
        print(f"Loading optimizer config: {config_path}")
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    args = load_json(cli_args.config)
    args['ckp_dir'] = cli_args.ckp_dir
    if not os.path.exists(args['ckp_dir']):
        os.makedirs(args['ckp_dir'])
    if config_path is not None:
        optimizer_args = load_yaml(config_path)
        args.update(optimizer_args)

    args['inftyopt'] = optimizer_name
    train(args)
    # infty_plot.visualize_trajectory(args['inftyopt'])

if __name__ == '__main__':
    main()
