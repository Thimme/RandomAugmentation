from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, DefaultTrainer
from detectron2.data import DatasetMapper, transforms as T
from detectron2 import model_zoo
from randaug.engine import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from randaug.data import datasets


def setup(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/faster_rcnn.yaml")
    cfg.eval_output = "./evaluation"
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 10 # magnitude of transforms
    return cfg


def main(args):
    sampler = TransformSampler(cfg, [])

    for augmentation in sampler.grid_search():
        trainer = RandTrainer(cfg, augmentation=augmentation) 
        trainer.resume_or_load(resume=args.resume)
        trainer.train()


def add_arguments():
    parser = default_argument_parser()
    parser.add_argument("--dataroot", type=str, help="Specify the folder to all datasets")
    return parser


if __name__ == "__main__":
    args = add_arguments().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)
    main(args)



