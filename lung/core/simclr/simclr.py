import logging
import shutil
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml


torch.manual_seed(0)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
	if not os.path.exists(model_checkpoints_folder):
		os.makedirs(model_checkpoints_folder)
	with open(os.path.join(model_checkpoints_folder, 'dump_config.yml'), 'w') as outfile:
		yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


class SimCLR(object):

	def __init__(self, model, optimizer, scheduler, **kw):
		self.args = kw

		self.model = model.to(self.args['device'])
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.log_dir = os.path.join(
			os.path.dirname(__file__), self.args['log_dir'])
		if not os.path.isdir(self.log_dir):
			os.makedirs(self.log_dir)
		# self.writer = SummaryWriter()

		if 'resume' in self.args.keys() and self.args['resume']:
			assert 'resume_model_path' in self.args.keys()
			checkpoint = torch.load(
				self.args['resume_model_path'], map_location=self.args['device'])
			state_dict = checkpoint['state_dict']
			model.load_state_dict(state_dict)

		logging.basicConfig(
			filename=os.path.join(self.log_dir, 'training.log'), level=logging.DEBUG)

		self.criterion = torch.nn.CrossEntropyLoss().to(self.args['device'])
		self.evals = {}

	def _push_eval(self, ekey, eidx, eval):
		if ekey not in self.evals.keys():
			self.evals[ekey] = ([], [])
		self.evals[ekey][0].append(eidx)
		self.evals[ekey][1].append(eval)

	def _get_last_eval(self, ekey):
		if ekey not in self.evals.keys():
			return -1
		return self.evals[ekey][1][-1]

	def get_eval(self, ekey):
		return self.evals[ekey]

	def info_nce_loss(self, features):

		labels = torch.cat([torch.arange(self.args['batch_size'])
						   for i in range(self.args['n_views'])], dim=0)
		labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
		labels = labels.to(self.args['device'])

		features = F.normalize(features, dim=1)

		similarity_matrix = torch.matmul(features, features.T)
		# assert similarity_matrix.shape == (
		#     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
		# assert similarity_matrix.shape == labels.shape

		# discard the main diagonal from both: labels and similarities matrix
		mask = torch.eye(labels.shape[0], dtype=torch.bool).to(
			self.args['device'])
		labels = labels[~mask].view(labels.shape[0], -1)
		similarity_matrix = similarity_matrix[~mask].view(
			similarity_matrix.shape[0], -1)
		# assert similarity_matrix.shape == labels.shape

		# select and combine multiple positives
		positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

		# select only the negatives the negatives
		negatives = similarity_matrix[~labels.bool()].view(
			similarity_matrix.shape[0], -1)

		logits = torch.cat([positives, negatives], dim=1)
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
			self.args['device'])

		logits = logits / self.args['temperature']
		return logits, labels

	def train(self, train_loader):

		scaler = GradScaler(enabled=self.args['fp16_precision'])

		# save config file
		save_config_file(self.log_dir, self.args)

		n_iter = 0
		logging.info(
			f"Start SimCLR training for {self.args['epochs']} epochs.")
		logging.info(f"Training with gpu: {self.args['enable_cuda']}.")

		for epoch_counter in range(self.args['epochs']):
			cus_bar_format = '[Epoch:{:4d}]'.format(epoch_counter)
			cus_bar_format += '{l_bar}{bar}{r_bar}'
			cus_bar_format += f"Step:{n_iter}\t Loss:{self._get_last_eval('loss'):.4f}\t acc/total:{self._get_last_eval('acc/total'):.2f} acc/top1:{self._get_last_eval('acc/top1'):.2f}\t acc/top5:{self._get_last_eval('acc/top5'):.2f}\t learning_rate:{self.scheduler.get_lr()[0]}"
			for images in tqdm(train_loader, bar_format=cus_bar_format):
				images = torch.cat(images, dim=0)

				images = images.to(self.args['device'])

				with autocast(enabled=self.args['fp16_precision']):
					features = self.model(images)
					logits, labels = self.info_nce_loss(features)
					loss = self.criterion(logits, labels)

				self.optimizer.zero_grad()

				scaler.scale(loss).backward()

				scaler.step(self.optimizer)
				scaler.update()

				if n_iter % self.args['log_every_n_steps'] == 0:
					top1, top5 = accuracy(logits, labels, topk=(1, 5))
					logging.debug(
						f"Step: {n_iter}\tLoss: {loss}\t acc/top1: {top1[0]}\t acc/top5: {top5[0]}\t learning_rate: {self.scheduler.get_lr()[0]}")

					self._push_eval('loss', int(n_iter), float(loss.cpu()))
					self._push_eval('acc/top1', int(n_iter),
									float(top1[0].cpu()))
					self._push_eval('acc/top5', int(n_iter),
									float(top5[0].cpu()))
					self._push_eval('learning_rate', int(
						n_iter), float(self.scheduler.get_lr()[0]))

				n_iter += 1

			# warmup for the first 10 epochs
			if epoch_counter >= 10:
				self.scheduler.step()
			logging.debug(
				f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0].cpu()}")
			self._push_eval('acc/total', int(n_iter), top1[0])

		logging.info("Training has finished.")
		# save model checkpoints
		checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(
			self.args['epochs'])
		save_checkpoint({
			'epoch': self.args['epochs'],
			'arch': self.args['arch'],
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
		}, is_best=False, filename=os.path.join(self.log_dir, checkpoint_name))
		logging.info(
			f"Model checkpoint and metadata has been saved at {self.log_dir}.")

		for name, vdict in self.evals.items():
			name_ = name.replace('/', '-')
			filename = os.path.join(self.log_dir, f'{name_}.csv')
			with open(filename, mode='w') as fobj:
				for i, v in zip(vdict[0], vdict[1]):
					fobj.write(f'{i:d},{v:f}\n')
		logging.info(f"Model evals has been saved at {self.log_dir}.")
