__author__ = 'ltnghia'

from pycocotools.coco import COCO
from cocoeval import COCOeval


if __name__ == '__main__':
    gt_file = ''
    result_file = ''
    tasks = ['bbox', 'segm']

    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(result_file)

    for task in tasks:
        print("Evaluate", task)
        cocoEval = COCOeval(cocoGt, cocoDt, task)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
