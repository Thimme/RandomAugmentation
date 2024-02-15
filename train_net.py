from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch
from randaug.engine import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from randaug.data import datasets
from randaug.models.detr import add_detr_config
from detectron2 import model_zoo


def setup_frcnn(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/faster_rcnn.yaml")
    cfg.eval_output = "./evaluation"
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 4 # magnitude of transforms
    # cfg.rand_augment = True
    cfg.box_postprocessing = False
    cfg.network = 'frcnn'
    return cfg

def setup_detr(args):
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/detr.yaml")
    cfg.eval_output = "./evaluation"
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 0 # magnitude of transforms
    #cfg.rand_augment = True
    cfg.box_postprocessing = False
    cfg.network = 'detr'
    return cfg

def setup_yolo(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.eval_output = "./evaluation"
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 0 # magnitude of transforms
    #cfg.rand_augment = True
    cfg.box_postprocessing = False
    cfg.network = 'yolo'
    return cfg

# run static augmentation via grid-search
def main(args):
    cfg = setup_frcnn(args)
    sampler = TransformSampler(cfg, epochs=args.epochs)

    for rand_M_value in range(0, 5):
        cfg.rand_M = rand_M_value

        for augmentation in sampler.grid_search():
            trainer = RandTrainer(cfg, augmentation=augmentation) 
            trainer.resume_or_load(resume=args.resume)
            trainer.train()


# run random augmentation algorithm over all augmentations
# def main(args):
#     cfg = setup_frcnn(args)
    
#     # Iterate over a range of values for rand_N and rand_M to configure and train the RandTrainer
#     for rand_N_value in range(1, 4):  # Looping from 1 to 3 for rand_N
#         for rand_M_value in range(0, 5):  # Looping from 0 to 4 for rand_M
#             # Set the configuration parameters
#             cfg.rand_N = rand_N_value
#             cfg.rand_M = rand_M_value

#             trainer = RandTrainer(cfg, augmentation=None)
#             trainer.resume_or_load(resume=args.resume)
#             trainer.train()


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



