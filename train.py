import click
from utils import utils_option as option

@click.command()
@click.option('--opt', default='options/nextvit_small.yml', help='Path to option YAML file.')
def main(opt):
    opt = option.parse(opt, is_train=True)


if __name__ == '__main__':
    main()
