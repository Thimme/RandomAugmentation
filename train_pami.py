from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from randaug.engine import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from randaug.data import datasets
from randaug.models.detr import add_detr_config
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results
import torch.multiprocessing as mp

def setup_frcnn(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/fasterrcnn.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'fasterrcnn'
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

def setup_retinanet(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/retinanet.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'retinanet'
    cfg.save_image = False
    return cfg

setup_dict = {
    "experiment_035_snow": [setup_retinanet]
}

def main(args):
    evaluate_experiment(args)

def add_arguments():
    parser = default_argument_parser()
    parser.add_argument("--dataroot", type=str, help="Specify the folder to all datasets")
    parser.add_argument("--epochs", default=0, type=int, help="Type the line number in results.json to continue")
    parser.add_argument("--experiment", type=str, help="Specify the experiment to be executed")
    return parser

# EVAL 

def evaluate_experiment(args):
    setup_funcs = [setup_frcnn, setup_detr, setup_retinanet]
    iterations = 3

    for _ in range(iterations):
        for setup_func in setup_funcs:
            cfg = setup_func(args)
            cfg.box_postprocessing = True
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

def evaluate_experiment_dict(args):
    for setup_func in setup_dict[args.experiment]:
        cfg = setup_func(args)
        cfg.box_postprocessing = True
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
            default_setup(cfg, args)
            
            sampler = TransformSampler(cfg, epochs=args.epochs)

            for augmentation in sampler.no_augmentation():
                trainer = RandTrainer(cfg, augmentation=augmentation)
                trainer.resume_or_load(resume=args.resume)
                trainer.train()
                sampler = TransformSampler(cfg, epochs=args.epochs)
            

if __name__ == "__main__":
    mp.set_start_method('spawn')
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
