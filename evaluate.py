import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

	parser = argparse.ArgumentParser(description='Simple script for evaluating a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	#parser.add_argument('--coco_path', help='Path to COCO directory')
	#parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--model', help='pretrained model')

	parser = parser.parse_args(args)

	if parser.dataset == 'csv':

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on COCO,')

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
			exit()
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv), exiting.')


	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	# Load the model		
	retinanet = torch.load(parser.model)
	print(retinanet)
	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	print('Evaluating dataset')

	retinanet.eval()
	mAP = csv_eval.evaluate(dataset_val, retinanet)
		


if __name__ == '__main__':
	main()
