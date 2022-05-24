import json
import os
from .parser import Arguments, parser

import torch
import torch.utils.data
from tqdm import tqdm

import eval.metrics as m
import eval.results as res
import utils.helpers as ood_utils
import vision.nn as models
from scores import scores

from aggregations import aggs

parser.add_argument(
    "--nseeds",
    type=int,
    default=1,
)


def main():
    args = Arguments(parser.parse_args())
    in_dataset_name = models.get_in_dataset_name(args.nn_name)
    print(args.nn_name, in_dataset_name)

    config = res.Config().from_kwargs(
        nn_name=args.nn_name,
        in_dataset=in_dataset_name,
        method=args.detector,
        reduce="none",
        checkpoint_number=1,
    )

    detector = scores.get_lite_score_by_name(config.method)
    n_classes = models.get_num_classes(args.nn_name)
    with open(res.get_metadata_path(config), "r") as f:
        metadata = json.load(f)

    config.update(features_names=metadata["features_names"])
    if args.features_names is not None and len(args.features_names) > 0:
        features_names = [
            f for f in args.features_names if f in metadata["features_names"]
        ]
        config.update(features_names=features_names)

    """load features"""
    train_logits = torch.load(res.get_train_logits_path(config), map_location="cpu")
    in_logits = torch.load(res.get_in_logits_path(config), map_location="cpu")
    train_pred = torch.argmax(train_logits, axis=1)
    train_target = torch.load(res.get_train_targets_path(config), map_location="cpu")
    in_pred = torch.argmax(in_logits, axis=1)
    train_scores = {
        f: detector.reduce(
            torch.load(
                res.get_train_scores_path_feature(config, f), map_location="cpu"
            ),
            mode=args.reduce,
            pred=train_pred,
            n_classes=n_classes,
        )
        for f in config.features_names
        if f in metadata["features_names"]
    }
    in_scores = {
        f: detector.reduce(v, mode=args.reduce, pred=in_pred, n_classes=n_classes)
        for f, v in torch.load(
            res.get_in_scores_path(config), map_location="cpu"
        ).items()
        if f in config.features_names and f in metadata["features_names"]
    }
    print(in_scores.keys())

    train_scores = ood_utils.dict2numpy(train_scores)
    in_scores = ood_utils.dict2numpy(in_scores)

    penult_in_scores = in_scores[:, -1].reshape(-1, 1)
    torch.save(
        torch.from_numpy(penult_in_scores), res.get_agg_in_scores_path(config, "penult")
    )

    agg = aggs.get_aggregation(args.aggregation)
    print(agg)
    for seed in range(args.nseeds):
        agg.fit(train_scores, pred=train_target.numpy())

        agg_in_scores = agg.forward(in_scores, pred=in_pred.numpy())
        torch.save(
            torch.from_numpy(agg_in_scores),
            res.get_agg_in_scores_path(config, args.aggregation),
        )

        for out_dataset in tqdm(args.ood_datasets):
            out_dataset = out_dataset.upper()
            print(out_dataset)
            config.update(out_dataset=out_dataset)
            out_logits = torch.load(res.get_out_logits_path(config), map_location="cpu")
            out_pred = torch.argmax(out_logits, axis=1)
            out_scores = {
                f: detector.reduce(
                    v, mode=args.reduce, pred=out_pred, n_classes=n_classes
                )
                for f, v in torch.load(
                    res.get_out_scores_path(config), map_location="cpu"
                ).items()
                if f in config.features_names and f in metadata["features_names"]
            }
            out_scores = ood_utils.dict2numpy(out_scores)

            assert train_scores.shape[1] == in_scores.shape[1] == out_scores.shape[1]

            penult_out_scores = out_scores[:, -1].reshape(-1, 1)
            torch.save(
                torch.from_numpy(penult_out_scores),
                res.get_agg_out_scores_path(config, "penult"),
            )
            agg_out_scores = agg.forward(out_scores, pred=out_pred.numpy())
            print("In scores shape", agg_in_scores.shape, agg_in_scores.mean())
            print("Out scores shape", agg_out_scores.shape, agg_out_scores.mean())
            torch.save(
                torch.from_numpy(agg_out_scores),
                res.get_agg_out_scores_path(config, args.aggregation),
            )

            """evaluate"""
            config.update(reduce=args.reduce)
            results_path = os.path.join("results", "results.csv")
            config.update(aggregation=args.aggregation)
            results = m.compute_detection_metrics(agg_in_scores, agg_out_scores)
            config.update(**results)
            """save results"""
            res.append_results_to_file(vars(config), results_path)
            print(
                "Results:",
                "TNR",
                results["tnr_at_0.95_tpr"] * 100,
                "AUROC",
                results["auroc"] * 100,
                "THR",
                results["thr"],
            ),

            """evaluate"""
            config.update(aggregation="minus_" + args.aggregation)
            results = m.compute_detection_metrics(-agg_in_scores, -agg_out_scores)
            config.update(**results)
            """save results"""
            res.append_results_to_file(vars(config), results_path)
            print(
                "Results:",
                "TNR",
                results["tnr_at_0.95_tpr"] * 100,
                "AUROC",
                results["auroc"] * 100,
                "THR",
                results["thr"],
            ),
            config.update(reduce="none")


if __name__ == "__main__":
    main()
