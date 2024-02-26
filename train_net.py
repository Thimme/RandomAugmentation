from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from randaug.engine import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from randaug.data import datasets
from randaug.models.detr import add_detr_config
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results


def setup_frcnn(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/fasterrcnn.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'frcnn'
    return cfg

def setup_detr(args):
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/detr.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'detr'
    return cfg

def setup_retinanet(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/retinanet.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'retinanet'
    return cfg

def randaug(cfg, args):    
    # Iterate over a range of values for rand_N and rand_M to configure and train the RandTrainer
    for rand_N_value in range(1, 4):  # Looping from 1 to 3 for rand_N
        for rand_M_value in range(0, 5):  # Looping from 0 to 4 for rand_M
            # Set the configuration parameters
            cfg.rand_N = rand_N_value
            cfg.rand_M = rand_M_value

            trainer = RandTrainer(cfg, augmentation=None)
            trainer.resume_or_load(resume=args.resume)
            trainer.train()

def grid_search(cfg, args):
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 0 # magnitude of transforms
    
    sampler = TransformSampler(cfg, epochs=args.epochs)

    for augmentation in sampler.diffusion_search():
        trainer = RandTrainer(cfg, augmentation=augmentation) 
        trainer.resume_or_load(resume=args.resume)
        trainer.train()

def inference(cfg, args):
    model = RandTrainer.build_model(cfg)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    res = RandTrainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, res)
    return res

def main(args):
        setup_funcs = [setup_frcnn, setup_detr, setup_retinanet]
        #cfg = setup_frcnn(args)
        #cfg = setup_detr(args)
        #cfg = setup_retinanet(args)

        for setup_func in setup_funcs:
            cfg = setup_func(args)
            #cfg.SOLVER.BASE_LR = lr
            cfg.box_postprocessing = False
            default_setup(cfg, args)

            randaug(cfg, args)
            # grid_search(cfg, args)
            # inference(cfg, args)

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




# EVAL 
    

def finetune():
    lrs = [1e-3, 1e-4, 1e-5]
    setup_funcs = [setup_frcnn, setup_detr, setup_retinanet]
    for lr in lrs:
        for setup_func in setup_funcs:
            cfg = setup_func(args)
            #cfg = setup_detr(args)
            #cfg = setup_retinanet(args)

            cfg.SOLVER.BASE_LR = lr
            cfg.box_postprocessing = False
            default_setup(cfg, args)

            randaug(cfg, args)
            # grid_search(cfg, args)
            # inference(cfg, args)