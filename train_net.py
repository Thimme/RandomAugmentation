from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser
import datasets

def setup(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    return cfg


def main(args):
    pass


def add_arguments():
    parser = default_argument_parser()
    parser.add_argument("--dataroot", type=str, help="Specify the folder to all datasets")
    return parser


if __name__ == "__main__":
    args = add_arguments().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)

    data = DatasetCatalog.get("midjourney_fog")
    meta = MetadataCatalog.get("midjourney_fog")
