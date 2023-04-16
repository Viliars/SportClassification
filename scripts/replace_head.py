import click
import random
import torch
from utils import utils_option as option
from models.select_model import define_Model

@click.command()
@click.option(
    "--opt", default="options/nextvit_small.yml", help="Path to option YAML file."
)
def main(opt):
    opt = option.parse(opt, is_train=False)
    
    num_classes = opt["net"]["num_classes"]
    opt["net"]["num_classes"] = 1000
    model = define_Model(opt)

    checkpoint = torch.load("checkpoints/nextvit_large_in1k6m_224.pth", map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.replace_head(num_classes)

    torch.save(model.state_dict(), "checkpoints/large30.pth")

if __name__ == "__main__":
    main()
