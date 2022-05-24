import os
import sys

import eval.metrics as m
import eval.results as res
import numpy as np
import torch
import torch.utils.data
import vision.nn as models
from torch.autograd import Variable
from tqdm import tqdm
from vision import data

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from .parser import Arguments, parser

parser.add_argument(
    "--score",
    type=str.lower,
    choices=["odin", "energy", "gradnorm"],
    default="odin",
)


def iterate_data_odin(data_loader, model, epsilon, temper):
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    confs = []
    for b, (x, y) in enumerate(tqdm(data_loader, "Odin")):
        x = Variable(x.to(DEVICE), requires_grad=True)
        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper
        if epsilon > 0:
            labels = Variable(torch.LongTensor(maxIndexTemp).to(DEVICE))
            loss = criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(x.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            x = torch.add(x.data, -epsilon, gradient)
            outputs = model(Variable(x))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs / temper

        outputs = outputs.data.cpu()
        outputs = outputs.numpy()
        outputs = outputs - np.max(outputs, axis=1, keepdims=True)
        outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)

        confs.extend(np.max(outputs, axis=1))

    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(tqdm(data_loader, "Energy")):
        with torch.no_grad():
            x = x.to(DEVICE)
            # compute output, measure accuracy and record loss.
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            conf = temper * torch.logsumexp(outputs / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_gradnorm(
    data_loader, model, temperature, num_classes, model_type="bit"
):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(DEVICE)
    for b, (x, y) in enumerate(tqdm(data_loader, "GradNorm")):
        inputs = Variable(x.to(DEVICE), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        targets = torch.ones((inputs.shape[0], num_classes)).to(DEVICE)
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()
        if model_type == "bit":
            layer_grad = model.head.conv.weight.grad.data
        elif model_type == "dn":
            layer_grad = model.classifier.weight.grad.data
        else:
            layer_grad = model.head.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


def run_eval(model, in_loader, out_loader, config: res.Config):
    model.eval()

    if args.score == "odin":
        in_scores = iterate_data_odin(in_loader, model, config.eps, config.temperature)
        out_scores = iterate_data_odin(
            out_loader, model, config.eps, config.temperature
        )
    elif args.score == "energy":
        in_scores = iterate_data_energy(in_loader, model, config.temperature)
        out_scores = iterate_data_energy(out_loader, model, config.temperature)
    elif args.score == "gradnorm":
        if "bit" in config.nn_name.lower():
            model_type = "bit"
        elif "denseneot" in config.nn_name.lower():
            model_type = "dn"
        else:
            model_type = "vit"

        in_scores = iterate_data_gradnorm(
            in_loader, model, args.temperature, num_classes, model_type=model_type
        )
        out_scores = iterate_data_gradnorm(
            out_loader, model, args.temperature, num_classes, model_type=model_type
        )

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    results_path = os.path.join("results", "results_baselines.csv")
    results = m.compute_detection_metrics(in_examples, out_examples)
    config.update(**results)
    res.append_results_to_file(vars(config), results_path)
    print(config)


def main():
    transform_test = models.get_data_transform(args.nn_name, train=False)
    in_dataset = data.get_dataset(
        in_dataset_name, transform=transform_test, train=False
    )
    in_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch_size, shuffle=False
    )
    out_dataset = data.get_dataset(
        args.ood_dataset, transform=transform_test, train=False
    )
    out_loader = torch.utils.data.DataLoader(
        out_dataset, batch_size=args.batch_size, shuffle=False
    )
    print("Datasets loaded.")

    model_path = models.get_model_path(args.nn_name)
    model = models.get_pre_trained_model_by_name(
        args.nn_name, path=model_path, pre_trained=True
    )
    model = model.to(DEVICE)
    print("Model loaded.")
    run_eval(model, in_loader, out_loader, args)


if __name__ == "__main__":
    args = Arguments(parser.parse_args())
    in_dataset_name = models.get_in_dataset_name(args.nn_name)
    if in_dataset_name.upper() == args.ood_dataset.upper():
        print("In dataset is equal to Out dataset.")
        sys.exit(1)
    num_classes = models.get_num_classes(args.nn_name)

    config = res.Config(
        nn_name=args.nn_name,
        in_dataset=in_dataset_name,
        out_dataset=args.ood_dataset,
        method=args.score,
        checkpoint_number=args.checkpoint_number,
    )
    print(config)
    main()
