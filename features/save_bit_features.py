import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm

import vision.bit.resnetv2 as resnetv2
from vision import data, nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from .parser import parser

PRE_TRAINED_DIR = os.path.expanduser(
    os.environ.get("PRE_TRAINED_DIR", "vision/pre_trained")
)

args = parser.parse_args()


def get_model_type(model_name: str):
    model_name = model_name.lower()
    if "bitsr101_" in model_name:
        return "BiT-S-R101x1"
    if "bitmr101_" in model_name:
        return "BiT-M-R101x1"


def main():
    # PREPARE
    in_dataset = args.model.lower().split("_")[1]
    if in_dataset == "imagenet":
        in_dataset = "ilsvrc2012"
    print("in dataset", in_dataset)

    # LOAD MODEL
    model_type = get_model_type(args.model)
    num_classes = data.get_dataset_n_classes_by_name(in_dataset)
    print("num classes", num_classes)
    model = resnetv2.KNOWN_MODELS[model_type](head_size=num_classes)
    old_w = model.head.conv.weight.tolist()
    file_list = glob.glob(f"{PRE_TRAINED_DIR}/*{model_type}*.npz")
    filepath = file_list[0]
    w = np.load(filepath)
    model.load_from(w)
    new_w = model.head.conv.weight.tolist()
    assert old_w != new_w, "model is not changed"
    model = model.to(DEVICE)
    model.eval()
    # LOAD DATASET
    print(nn.number_of_parameters(model), "params")

    dataset_name = args.dataset
    print(dataset_name)
    img_size = 480
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = data.get_dataset(dataset_name, transform=transform_test, train=args.train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    features = defaultdict(list)
    targets = []
    with torch.no_grad():
        for (batch, target) in tqdm(loader, "Extracting features"):
            batch = batch.to(DEVICE)
            targets.append(target.reshape(-1, 1))

            root = model.root(batch)
            features["root"].append(
                root.flatten(-2).mean(-1).cpu()
            )  # select class embedding token

            hidden_states = root
            for layer_number, layer_block in enumerate(
                [
                    model.body.block1,
                    model.body.block2,
                    model.body.block3,
                    model.body.block4,
                ]
            ):
                hidden_states = layer_block(hidden_states)
                features["res_layer_{}".format(layer_number)].append(
                    hidden_states.flatten(-2).mean(-1).cpu()
                )  # select class embedding token
            encoded = model.before_head(hidden_states)
            features["encoder_output"].append(encoded.flatten(-2).mean(-1).cpu())
            logits = model.head(encoded)
            features["linear"].append(logits.squeeze().cpu())

    ks = list(features.keys())
    metadata = {"features_names": ks}

    destination_path = os.path.join(args.save_root, args.model, dataset_name)
    os.makedirs(destination_path, exist_ok=True)
    if args.train:
        filename = f"feature_{args.seed}"
        filename += "_train"

        for k in ks:
            destination_path = os.path.join(args.save_root, args.model, dataset_name, k)
            os.makedirs(destination_path, exist_ok=True)
            torch.save(
                torch.cat(features[k], dim=0),
                os.path.join(destination_path, filename + ".pt"),
            )
            print(f"FEATURES SAVED to {filename}")
            if k != "linear":
                del features[k]
            logits = torch.cat(features["linear"], dim=0)
    else:
        features = {k: torch.cat(v, dim=0) for k, v in features.items()}
        filename = f"all_features_nodes_{args.seed}"

        destination_path = os.path.join(args.save_root, args.model, dataset_name)
        os.makedirs(destination_path, exist_ok=True)
        torch.save(features, os.path.join(destination_path, filename + ".pt"))
        print(f"FEATURES SAVED to {filename}")
        logits = features["linear"]

    with open(os.path.join(args.save_root, args.model, "metadata.json"), "w") as fp:
        json.dump(metadata, fp)
    destination_path = os.path.join(args.save_root, args.model, dataset_name)
    filename = f"logits_{args.seed}"
    suffix = ""
    if args.train:
        suffix = "_train"
    print(logits.shape)
    torch.save(logits, os.path.join(destination_path, filename + suffix + ".pt"))
    targets = torch.vstack(targets).reshape(-1)
    torch.save(targets, os.path.join(destination_path, "targets" + suffix + ".pt"))
    print("LOGITS and TARGETS SAVED")

    acc = (logits.argmax(dim=1) == targets).float().mean()
    print(args.model, args.dataset, "seed", args.seed, f"ACCURACY: {acc}")


if __name__ == "__main__":
    main()
