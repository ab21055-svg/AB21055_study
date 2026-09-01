"""円だけを学習する実行スクリプト。

"""

import argparse
from training_common import train_entity_model


def main():
    parser = argparse.ArgumentParser(description="Train circle model")
    parser.add_argument("--data-root", dest="data_root", help="Dataset root directory (overrides DATA_ROOT)")
    parser.add_argument("--dataset-suffix", dest="dataset_suffix", default="_n1", help="Dataset suffix to use (default: _n1)")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)

    args = parser.parse_args()

    train_entity_model(
        entity_type="circle",
        dataset_suffix=args.dataset_suffix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
