from collections import defaultdict
import json
import os

import torch
import torch.utils.data
from tqdm import tqdm

import vision.data as data
import vision.nn as models


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from .parser import parser

args = parser.parse_args()


def feature_nodes_extraction_loop(feature_extractor, dataloader, gpu, reduce=True):
    features = defaultdict(list)
    targets = []
    with torch.no_grad():
        for (data, target) in tqdm(dataloader, "Extracting features"):
            if gpu is not None:
                data = data.to(gpu)
            feature = feature_extractor(data)
            # reduce features
            if reduce:
                for k, v in feature.items():
                    features[k].append(
                        v.reshape(v.shape[0], v.shape[1], -1).mean(-1).cpu()
                    )
            else:
                for k, v in feature.items():
                    features[k].append(v.cpu())
        targets.append(target.detach().cpu().reshape(-1, 1))

    features = {k: torch.cat(v, dim=0) for k, v in features.items()}
    targets = torch.vstack(targets).reshape(-1)
    return features, targets


def main(args):
    """load dataset"""
    in_dataset_name = models.get_in_dataset_name(args.model)
    print(in_dataset_name, args.dataset, args.model)
    if in_dataset_name.upper() != args.dataset.upper() and args.train:
        return
    transform = data.get_dataset_transformation_by_name(in_dataset_name, train=False)
    dataset = data.get_dataset(args.dataset, transform=transform, train=args.train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    """load model"""
    feature_extractor = models.get_feature_extractor_by_name(
        args.model, pre_trained=True, linear_nodes=True, seed=args.seed
    )
    feature_extractor.eval()
    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor = torch.nn.DataParallel(feature_extractor)

    """save features"""
    features, targets = feature_nodes_extraction_loop(
        feature_extractor=feature_extractor,
        dataloader=loader,
        gpu=DEVICE,
    )
    print("FINISHED EXTRACTING FEATURES")

    ks = list(features.keys())
    metadata = {"features_names": ks}
    destination_path = os.path.join(args.save_root, args.model, args.dataset)
    os.makedirs(destination_path, exist_ok=True)
    with open(os.path.join(args.save_root, args.model, "metadata.json"), "w") as fp:
        json.dump(metadata, fp)

    if args.train:
        filename = f"feature_{args.seed}" if args.seed is not None else "feature"
        filename += "_train"
        for k in list(features.keys()):
            destination_path = os.path.join(args.save_root, args.model, args.dataset, k)
            os.makedirs(destination_path, exist_ok=True)
            torch.save(features[k], os.path.join(destination_path, f"{filename}.pt"))
    else:
        filename = (
            f"all_features_nodes_{args.seed}"
            if args.seed is not None
            else "all_features_nodes"
        )
        destination_path = os.path.join(args.save_root, args.model, args.dataset)
        os.makedirs(destination_path, exist_ok=True)
        torch.save(features, os.path.join(destination_path, f"{filename}.pt"))
        print("FEATURES SAVED TO", destination_path, f"{filename}.pt")

    if "classifier" in list(features.keys()):
        filename = f"logits_{args.seed}" if args.seed is not None else "logits"
        filename += "_train" if args.train else ""
        destination_path = os.path.join(args.save_root, args.model, args.dataset)
        os.makedirs(destination_path, exist_ok=True)
        torch.save(
            features["classifier"], os.path.join(destination_path, f"{filename}.pt")
        )
    filename = "targets_train" if args.train else "targets"
    torch.save(targets, os.path.join(destination_path, f"{filename}.pt"))


if __name__ == "__main__":
    main(args)
