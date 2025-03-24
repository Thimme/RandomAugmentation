from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from randaug.engine import RandTrainer
from randaug.engine.transform_sampler import TransformSampler, dropout_transforms, image_transforms, weather_transforms, geometric_transforms, gan_transforms, diffusion_transforms
from randaug.data import datasets
from randaug.models.detr import add_detr_config
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results
import torch.multiprocessing as mp
from itertools import product

def setup_frcnn(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/fasterrcnn.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'fasterrcnn'
    cfg.save_image = False
    return cfg

def setup_frcnn_r101(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/fasterrcnn_r101.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'fasterrcnn_r101'
    cfg.save_image = False
    return cfg

def setup_frcnn_r101_dc5(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/fasterrcnn_r101_dc5.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'fasterrcnn_r101_dc5'
    cfg.save_image = False
    return cfg

def setup_detr(args):
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/detr.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'detr'
    cfg.save_image = False
    return cfg

def setup_detr_r101(args):
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/detr_r101.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'detr_r101'
    cfg.save_image = False
    return cfg

def setup_detr_r101_dc5(args):
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/detr_r101_dc5.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'detr_r101_dc5'
    cfg.save_image = False
    return cfg

def setup_retinanet(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/retinanet.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'retinanet'
    cfg.save_image = False
    return cfg

def setup_retinanet_r101(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/retinanet_r101.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'retinanet_r101'
    cfg.save_image = False
    return cfg

def main(args):
    random_experiment(args)
    
def add_arguments():
    parser = default_argument_parser()
    parser.add_argument("--dataroot", type=str, help="Specify the folder to all datasets")
    parser.add_argument("--epochs", default=0, type=int, help="Type the line number in results.json to continue")
    parser.add_argument("--experiment", default="default", type=str, help="Specify the experiment to be executed")
    parser.add_argument("--experiment_name", type=str, help="Specify the name for the experiment to be saved")
    parser.add_argument("--bbox", action='store_true', help="Specify if bounding box estimation is to be used")
    parser.add_argument("--cutout", action='store_true', help="Specify if cutout is to be used")
    parser.add_argument("--frozen_backbone", action='store_true', help="Specify if backbone is frozen")
    parser.add_argument("--progressive", action='store_true', help="Specify if training is progressive")
    parser.add_argument("--iterations", type=int, default=3, help="Specify the number of itations")
    parser.add_argument("--weather", type=str, default="diverse", help="Specify an extra tag to separate datasets")
    parser.add_argument("--magnitude", type=int, default="0", help="Specify the applied magnitude")
    parser.add_argument("--magnitude-fixed", action='store_true', help="Specify if the magnitude is fixed")
    parser.add_argument("--no_deep", action='store_false', help="Specify if deep augmentations are used")
    parser.add_argument("--no_overlay", action='store_false', help="Specify if weather overlays are used")
    parser.add_argument("--no_standard", action='store_false', help="Specify if standard augmentations are used")

    return parser

def custom_setup_backbone_augmentation(cfg):
    if 'detr' in cfg.network:
        cfg.progressive = True
    elif 'fasterrcnn' in cfg.network: 
        cfg.progressive = False
    elif 'retinanet' in cfg.network:
        cfg.progressive = False
        cfg.cutout_postprocessing = True
    return cfg

def custom_setup_cyclegan_cut(cfg):
    if 'detr' in cfg.network:
        cfg.cutout_postprocessing = False
    return cfg

def experiment(args):
    setup_funcs = [setup_detr, setup_retinanet]
    transforms = diffusion_transforms
    magnitudes = [9,6]

    for _ in range(args.iterations):
        for setup_func, aug, magnitude in product(setup_funcs, transforms, magnitudes):
            cfg = setup_func(args)
            cfg.box_postprocessing = args.bbox
            cfg.cutout_postprocessing = args.cutout
            cfg.frozen_backbone = args.frozen_backbone
            cfg.progressive = args.progressive
            cfg.weather = args.weather
            cfg.SOLVER.MAX_ITER = 3000
            cfg.TEST.EVAL_PERIOD = 0
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.experiment = args.experiment
            cfg.aug_prob = 1.0
            cfg.magnitude = magnitude
            cfg.experiment_name = aug
            cfg.magnitude_fixed = args.magnitude_fixed
            
            # if args.experiment_name == 'experiment_backbone_augmentation':
            #     cfg = custom_setup_backbone_augmentation(cfg)
            # elif args.experiment_name == 'experiment_backbone_cyclegan':
            #     cfg = custom_setup_cyclegan_cut(cfg)

            default_setup(cfg, args)


            sampler = TransformSampler(cfg, epochs=args.epochs)

            for augmentation in sampler.augmentation(augmentation=aug, magnitude=magnitude):
                trainer = RandTrainer(cfg, augmentation=augmentation) 
                trainer.resume_or_load(resume=args.resume)
                trainer.train()


def random_experiment(args):
    setup_funcs = [setup_detr, setup_frcnn, setup_retinanet]
    transforms = image_transforms # geometric_transforms + weather_transforms + dropout_transforms
    magnitudes = [9, 6, 3]

    for _ in range(args.iterations):
        for setup_func, aug, magnitude in product(setup_funcs, transforms, magnitudes):
            cfg = setup_func(args)
            cfg.box_postprocessing = args.bbox
            cfg.cutout_postprocessing = args.cutout
            cfg.frozen_backbone = args.frozen_backbone
            cfg.progressive = args.progressive
            cfg.weather = args.weather
            cfg.SOLVER.MAX_ITER = 3000
            cfg.TEST.EVAL_PERIOD = 0
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.experiment = args.experiment
            cfg.aug_prob = 1.0
            cfg.magnitude = magnitude
            cfg.experiment_name = aug
            cfg.magnitude_fixed = args.magnitude_fixed
            cfg.use_deep = args.no_deep
            cfg.use_overlay = args.no_overlay
            cfg.use_standard = args.no_standard
            
            # if args.experiment_name == 'experiment_backbone_augmentation':
            #     cfg = custom_setup_backbone_augmentation(cfg)
            # elif args.experiment_name == 'experiment_backbone_cyclegan':
            #     cfg = custom_setup_cyclegan_cut(cfg)

            default_setup(cfg, args)


            sampler = TransformSampler(cfg, epochs=args.epochs)

            for augmentation in sampler.augmentation(augmentation=aug, magnitude=magnitude):
                trainer = RandTrainer(cfg, augmentation=None)
                trainer.resume_or_load(resume=args.resume)
                trainer.train()


def progressive_augmentation(args):
    setup_funcs = [setup_frcnn, setup_detr, setup_retinanet]
    iterations = 3

    for _ in range(iterations):
        for setup_func in setup_funcs:
            cfg = setup_func(args)
            cfg.box_postprocessing = args.bbox
            cfg.cutout_postprocessing = args.cutout
            cfg.frozen_backbone = False
            default_setup(cfg, args)

            # Set the configuration parameters
            cfg.experiment = args.experiment
            cfg.aug_prob = 1.0
            cfg.rand_N = 1
            cfg.rand_M = 0

            sampler = TransformSampler(cfg, epochs=args.epochs)

            for augmentation in sampler.experiment(experiment=cfg.experiment):
                trainer = RandTrainer(cfg, augmentation=augmentation) 
                trainer.resume_or_load(resume=args.resume)
                trainer.train()


def diverse_augmentation(args):
    setup_funcs = [setup_frcnn, setup_frcnn, setup_detr, setup_retinanet, setup_frcnn, setup_detr, setup_retinanet]
    iterations = 1

    for _ in range(iterations):
        for setup_func in setup_funcs:
            cfg = setup_func(args)
            cfg.box_postprocessing = args.bbox
            cfg.cutout_postprocessing = args.cutout
            cfg.frozen_backbone = False
            default_setup(cfg, args)

            # Set the configuration parameters
            cfg.experiment = args.experiment
            cfg.aug_prob = 1.0
            cfg.rand_N = 1
            cfg.rand_M = 0

            sampler = TransformSampler(cfg, epochs=args.epochs)

            for augmentation in sampler.experiment(experiment=cfg.experiment):
                trainer = RandTrainer(cfg, augmentation=augmentation) 
                trainer.resume_or_load(resume=args.resume)
                trainer.train()
                

def no_augmentation(args):
    setup_funcs = [setup_frcnn, setup_detr, setup_retinanet]
    iterations = 1

    for _ in range(iterations):
        for setup_func in setup_funcs:
            cfg = setup_func(args)
            cfg.aug_prob = 1.0
            cfg.rand_N = 2 # number of transforms
            cfg.rand_M = 2 # magnitude of transforms

            cfg.box_postprocessing = False
            cfg.frozen_backbone = False
            default_setup(cfg, args)
            
            sampler = TransformSampler(cfg, epochs=args.epochs)

            for augmentation in sampler.no_augmentation():
                trainer = RandTrainer(cfg, augmentation=augmentation)
                trainer.resume_or_load(resume=args.resume)
                trainer.train()


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
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
