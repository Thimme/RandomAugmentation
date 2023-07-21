from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, DefaultTrainer
from detectron2.data import DatasetMapper, transforms as T
from detectron2 import model_zoo
from randaug.engine import RandTrainer
from randaug.engine.rand_trainer import build_rand_train_aug
from detectron2.utils.visualizer import Visualizer
from randaug.data import datasets
import cv2


def setup(args):
    cfg = get_cfg()
    cfg.dataroot = args.dataroot
    cfg.merge_from_file("configs/faster_rcnn.yaml")
    cfg.eval_output = "./evaluation"
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

    # d = DatasetCatalog.get("midjourney_fog")[0]
    # img = cv2.imread(d["file_name"])
    # augs = T.AugmentationList(build_rand_train_aug(cfg))
    # input = T.AugInput(img)
    # transform = augs(input)

    # visualizer = Visualizer(input.image, metadata=MetadataCatalog.get("midjourney_fog"), scale=0.5)
    # out = visualizer.draw_dataset_dict(input.boxes)
    # cv2.imwrite("aug.jpg", out.get_image())

    trainer = RandTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


