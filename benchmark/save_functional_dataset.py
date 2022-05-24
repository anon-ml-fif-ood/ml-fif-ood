import json
import sys
from collections import defaultdict
from .parser import Arguments, parser

import torch
import torch.utils.data
from tqdm import tqdm

import eval.results as res
import utils.helpers as ood_utils
import vision.nn as models
from scores import scores

print(f"{torch.cuda.device_count()} GPU(s) detected")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


def main():

    """load metadata"""
    with open(res.get_metadata_path(config), "r") as f:
        metadata = json.load(f)
        print(
            "Metadata",
            metadata,
            metadata["features_names"],
            type(metadata["features_names"]),
        )

    """add logits or softmax to features dict"""
    features_names = ["all", "linear"]

    if args.features_names is not None:
        features_names = args.features_names
        metadata["features_names"] = features_names

    print("Features to consider:", features_names)
    config.update(features_names=features_names)

    """load some data"""
    train_targets = torch.load(res.get_train_targets_path(config), map_location="cpu")
    train_logits = torch.load(res.get_train_logits_path(config), map_location="cpu")
    in_targets = torch.load(res.get_in_targets_path(config), map_location="cpu")
    in_logits = torch.load(res.get_in_logits_path(config), map_location="cpu")
    train_pred = torch.argmax(train_logits, axis=1)
    in_pred = torch.argmax(in_logits, dim=1)
    n_classes = models.get_num_classes(args.nn_name)
    all_in_features = torch.load(res.get_in_features_path(config), map_location="cpu")

    detector = {}
    in_scores = defaultdict(list)
    with torch.no_grad():
        for f in tqdm(metadata["features_names"], "Feature"):
            train_scores = []
            """load features"""
            train_features = torch.load(
                res.get_train_features_path(config, f), map_location="cpu"
            )
            in_features = all_in_features[f]

            detector[f] = scores.get_lite_score_by_name(config.method)

            """loaders"""
            train_features_dataset = torch.utils.data.TensorDataset(
                train_features, train_pred
            )
            train_loader = torch.utils.data.DataLoader(
                train_features_dataset,
                batch_size=len(train_features_dataset),
                shuffle=False,
            )
            train_loader_infer = torch.utils.data.DataLoader(
                train_features_dataset, batch_size=args.batch_size, shuffle=False
            )

            in_features_dataset = torch.utils.data.TensorDataset(in_features, in_pred)
            in_loader = torch.utils.data.DataLoader(
                in_features_dataset, batch_size=args.batch_size, shuffle=False
            )

            """calculate and reduce feature scores"""

            # fit the detector (not actually batched)
            for (feature, _) in tqdm(train_loader, "Fitting detector"):
                if f == "linear":
                    detector[f].fit_logits(feature.to(DEVICE), labels=train_targets)
                else:
                    detector[f].fit(feature.to(DEVICE), labels=train_targets)

            # calculate train scores
            for (feature, target) in tqdm(train_loader_infer, "Train scores"):
                feat = feature.to(DEVICE)
                if f == "linear":
                    fwd = detector[f].forward_logits(feat, pred=target)
                else:
                    fwd = detector[f].forward(feat, pred=target)
                train_scores.append(fwd.cpu())

            train_scores = torch.vstack(train_scores)
            train_scores = detector[f].reduce(
                train_scores, mode=config.reduce, pred=train_pred, n_classes=n_classes
            )
            if not args.dont_save:
                torch.save(train_scores, res.get_train_scores_path_feature(config, f))
            del train_scores

            # calculate in-distribution scores
            for (feature, target) in tqdm(in_loader, "In scores"):
                feat = feature.to(DEVICE)
                if f == "linear":
                    fwd = detector[f].forward_logits(feat, pred=target)
                else:
                    fwd = detector[f].forward(feat, pred=target)
                in_scores[f].append(fwd.cpu())

            in_scores[f] = torch.vstack(in_scores[f])
            in_scores[f] = detector[f].reduce(
                in_scores[f], mode=config.reduce, pred=in_pred, n_classes=n_classes
            )

        """save in scores"""
        if not args.dont_save:
            torch.save(in_scores, res.get_in_scores_path(config))
        in_scores = ood_utils.dict2numpy(in_scores)

        """OOD scores"""
        for out_dataset_name in tqdm(args.ood_datasets, "Out datasets"):
            try:
                out_dataset_name = out_dataset_name.upper()
                config.update(out_dataset=out_dataset_name)

                all_out_features = torch.load(
                    res.get_out_features_path(config), map_location="cpu"
                )
                out_logits = torch.load(
                    res.get_out_logits_path(config), map_location="cpu"
                )
                out_targets = torch.tensor([-1] * len(out_logits)).cpu()
                out_pred = torch.argmax(out_logits, dim=1)

                out_scores = defaultdict(list)
                for f in tqdm(metadata["features_names"], "Feature"):
                    """features"""
                    out_features = all_out_features[f]
                    """loaders"""
                    out_features_dataset = torch.utils.data.TensorDataset(
                        out_features, out_pred
                    )
                    out_loader = torch.utils.data.DataLoader(
                        out_features_dataset, batch_size=args.batch_size, shuffle=False
                    )

                    """score"""
                    # calculate out-of-distribution scores
                    for (feature, target) in tqdm(out_loader, "Out scores"):
                        feat = feature.to(DEVICE)
                        if f == "linear":
                            fwd = detector[f].forward_logits(feat, pred=target)
                        else:
                            fwd = detector[f].forward(feat, pred=target)
                        out_scores[f].append(fwd.cpu())

                    out_scores[f] = torch.vstack(out_scores[f])
                    out_scores[f] = detector[f].reduce(
                        out_scores[f],
                        mode=config.reduce,
                        pred=out_pred,
                        n_classes=n_classes,
                    )

                """save out scores"""
                if not args.dont_save:
                    torch.save(out_scores, res.get_out_scores_path(config))

            except Exception as e:
                print(out_dataset_name, e)
                # raise e
                continue


if __name__ == "__main__":
    args = Arguments(parser.parse_args())
    in_dataset_name = models.get_in_dataset_name(args.nn_name)
    if in_dataset_name.upper() == args.ood_dataset.upper():
        print("In dataset is equal to Out dataset.")
        sys.exit(1)

    if args.ood_datasets is not None and len(args.ood_datasets) > 0:
        out_datasets = args.ood_datasets
    else:
        out_datasets = [args.ood_dataset]

    config = res.Config(
        nn_name=args.nn_name,
        in_dataset=in_dataset_name,
        # out_dataset=args.ood_dataset,
        method=args.detector,
        reduce=args.reduce,
        checkpoint_number=1,
    )
    print(config)
    main()
