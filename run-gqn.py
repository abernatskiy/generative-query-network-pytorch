#!/usr/bin/env python3

'''
run-gqn.py

Script to train the a GQN on the Shepard-Metzler dataset
in accordance to the hyperparameter settings described in
the supplementary materials of the paper.
'''
import selectgpu
selectgpu.selectGPU(2)

import random
import math
from argparse import ArgumentParser
from pathlib import Path

# Torch
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# TensorboardX
from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, Timer
from ignite.metrics import RunningAverage

from gqn import GenerativeQueryNetwork, partition, Annealer
from asteroids import Asteroids
#from shepardmetzler import ShepardMetzler

device = torch.device('cuda:0') # selectgpu ensures that cuda:0 can be used

# Random seeding
random.seed(99)
torch.manual_seed(99)
torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
	parser = ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
	parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs run (default: 200)')
	parser.add_argument('--batch_size', type=int, default=1, help='multiple of batch size (default: 1)')
	parser.add_argument('--data_dir', type=str, help='location of data', default='train')
	parser.add_argument('--log_dir', type=str, help='location of logging', default='logs')
	parser.add_argument('--fraction', type=float, help='how much of the data to use', default=1.0)
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
	parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
	parser.add_argument('--resume_from_checkpoint', type=str, help='location of a checkpoint to resume training from (default: None)', default=None)
	args = parser.parse_args()

	rootDir = Path(args.data_dir)

	# Create model and optimizer
#	model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)
	model = GenerativeQueryNetwork(x_dim=1, v_dim=5, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=5 * 10 ** (-5))

	# Rate annealing schemes
	sigma_scheme = Annealer(2.0, 0.7, 2 * 10 ** 5)
	mu_scheme = Annealer(5 * 10 ** (-4), 5 * 10 ** (-6), 1.6 * 10 ** 5)

	# Load the dataset
	#train_dataset = ShepardMetzler(root_dir=args.data_dir, fraction=args.fraction)
	#valid_dataset = ShepardMetzler(root_dir=args.data_dir, fraction=args.fraction, train=False)
	train_dataset = Asteroids(rootDir, True, 20)
	valid_dataset = Asteroids(rootDir, False, 2)

	kwargs = {'num_workers': args.workers, 'pin_memory': True}
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
	valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

	def step(engine, batch):
		model.train()

		x, v = batch
		x, v = x.to(device), v.to(device)

		x, v, x_q, v_q = partition(x, v)

		# Reconstruction, representation and divergence
		x_mu, _, kl = model(x, v, x_q, v_q)

		# Log likelihood
		sigma = next(sigma_scheme)
		ll = Normal(x_mu, sigma).log_prob(x_q)

		likelihood     = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
		kl_divergence  = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

		# Evidence lower bound
		elbo = likelihood - kl_divergence
		loss = -elbo
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		with torch.no_grad():
			# Anneal learning rate
			mu = next(mu_scheme)
			i = engine.state.iteration
			for group in optimizer.param_groups:
				group['lr'] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

		print(f'e{engine.state.epoch},i{engine.state.iteration}/{engine.state.epoch_length}: elbo {elbo.item()}, kl {kl_divergence.item()}, sigma {sigma}, mu {mu}')

		return {'elbo': elbo.item(), 'kl': kl_divergence.item(), 'sigma': sigma, 'mu': mu}

	# Trainer and metrics
	trainer = Engine(step)
	metric_names = ['elbo', 'kl', 'sigma', 'mu']
	RunningAverage(output_transform=lambda x: x['elbo']).attach(trainer, 'elbo')
	RunningAverage(output_transform=lambda x: x['kl']).attach(trainer, 'kl')
	RunningAverage(output_transform=lambda x: x['sigma']).attach(trainer, 'sigma')
	RunningAverage(output_transform=lambda x: x['mu']).attach(trainer, 'mu')
#	ProgressBar().attach(trainer, metric_names=metric_names)

	# Model checkpointing
	to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'sigma_annealer': sigma_scheme, 'mu_annealer': mu_scheme}
	checkpoint_handler = Checkpoint(to_save, DiskSaver('./checkpoints', create_dir=True, require_empty=False), n_saved=None)

	trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED(every=1), handler=checkpoint_handler)
#	trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED(every=1000), handler=checkpoint_handler)

	@trainer.on(Events.ITERATION_COMPLETED)
	def log_metrics(engine):
		for key, value in engine.state.metrics.items():
			writer.add_scalar('training/{}'.format(key), value, engine.state.iteration)

	@trainer.on(Events.EPOCH_COMPLETED)
	def save_images(engine):
		with torch.no_grad():
			x, v = engine.state.batch
			x, v = x.to(device), v.to(device)
			x, v, x_q, v_q = partition(x, v)

			x_mu, r, _ = model(x, v, x_q, v_q)

			r = r.view(-1, 1, 16, 16)

			# Send to CPU
			x_mu = x_mu.detach().cpu().float()
			r = r.detach().cpu().float()

			writer.add_image('representation', make_grid(r), engine.state.epoch)
			writer.add_image('reconstruction', make_grid(x_mu), engine.state.epoch)

	@trainer.on(Events.EPOCH_COMPLETED)
	def validate(engine):
		model.eval()
		with torch.no_grad():
			x, v = next(iter(valid_loader))
			x, v = x.to(device), v.to(device)
			x, v, x_q, v_q = partition(x, v)

			# Reconstruction, representation and divergence
			x_mu, _, kl = model(x, v, x_q, v_q)

			# Validate at last sigma
			ll = Normal(x_mu, sigma_scheme.recent).log_prob(x_q)

			likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
			kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

			# Evidence lower bound
			elbo = likelihood - kl_divergence

			writer.add_scalar('validation/elbo', elbo.item(), engine.state.epoch)
			writer.add_scalar('validation/kl', kl_divergence.item(), engine.state.epoch)

	@trainer.on(Events.EXCEPTION_RAISED)
	def handle_exception(engine, e):
		writer.close()
		engine.terminate()
		if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
			import warnings
			warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
			checkpoint_handler(engine)
		else: raise e

	timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
	             pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

	# Tensorbard writer
	writer = SummaryWriter(log_dir=args.log_dir)

	if args.resume_from_checkpoint:
		print(f'Resuming training at {args.data_dir} using checkpoint {args.resume_from_checkpoint}')
		Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(Path(args.resume_from_checkpoint)))
	else:
		print(f'Training from scratch at {args.data_dir}')

	trainer.run(train_loader, args.n_epochs)
	writer.close()
