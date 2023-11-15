from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch
from randaug.engine import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from randaug.data import datasets
from detectron2 import model_zoo


def setup(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/faster_rcnn.yaml")
    cfg.eval_output = "./evaluation"
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 3 # magnitude of transforms
    cfg.box_postprocessing = False
    return cfg


def main(args):
    cfg = setup(args)
    sampler = TransformSampler(cfg, epochs=args.epochs)

    for augmentation in sampler.grid_search():
        trainer = RandTrainer(cfg, augmentation=augmentation) 
        trainer.resume_or_load(resume=args.resume)
        trainer.train()


def add_arguments():
    parser = default_argument_parser()
    parser.add_argument("--dataroot", type=str, help="Specify the folder to all datasets")
    parser.add_argument("--epochs", default=0, type=int, help="Type the line number in results.json to continue")
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



