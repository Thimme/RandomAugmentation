from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch
from randaug.engine import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from randaug.data import datasets


def setup(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/faster_rcnn.yaml")
    cfg.eval_output = "./evaluation"
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 5 # magnitude of transforms
    return cfg


def main(args):
    cfg = setup(args)
    sampler = TransformSampler(cfg)

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
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



