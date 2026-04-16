import torch
from pprint import pprint

from bcos.experiments.utils import Experiment
from bcos.models.standard_models import I3D
from bcos.training.bcosify_trainer import BcosifyTrainer
# Adjust these imports to your project
from evaluate import load_model_and_config


def print_model_architecture(model, name, save_path=None):
    print("\n" + "="*80)
    print(f"{name}")
    print("="*80)
    print(model)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(str(model))


def print_named_modules(model, name):
    print("\n" + "="*80)
    print(f"{name} NAMED MODULES")
    print("="*80)

    for module_name, module in model.named_modules():
        print(f"{module_name}: {type(module).__name__}")


def main():

    exp = Experiment("UCF101", "standard", "i3d")
    standard_config = exp.config.copy()
    ###########################################
    # LOAD STANDARD MODEL
    ###########################################
    standard_model = I3D(True)

    ###########################################
    # LOAD BCOS MODEL
    ###########################################
    bcos_model = BcosifyTrainer(
        "UCF101",
        "bcosification",
        "i3d"
    )

    ###########################################
    # PRINT ARCHITECTURES
    ###########################################
    print_model_architecture(
        standard_model,
        "STANDARD I3D ARCHITECTURE",
        save_path="standard_i3d.txt"
    )

    print_model_architecture(
        bcos_model,
        "BCOSIFIED I3D ARCHITECTURE",
        save_path="bcos_i3d.txt"
    )

    ###########################################
    # PRINT NAMED MODULES FOR DEBUGGING
    ###########################################
    print_named_modules(standard_model, "STANDARD I3D")
    print_named_modules(bcos_model, "BCOS I3D")


if __name__ == "__main__":
    main()