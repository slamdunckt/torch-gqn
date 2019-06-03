import argparse
import datetime
import math
import os
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from gqn_dataset import GQNDataset, Scene, transform_viewpoint, sample_batch
from scheduler import AnnealingStepLR
from model import GQN
from model_attention import GQNAttention

def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']

        #model.load_state_dict(checkpoint['state_dict'])  # simple original usage
                                                          # but we used nn.parallel...
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        '''
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        '''

        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network Implementation')
    parser.add_argument('--gradient_steps', type=int, default=1*10**6, \
                        help='number of gradient steps to run (default: 1 million)')
    parser.add_argument('--batch_size', type=int, default=36, \
                        help='size of batch (default: 36)')
    parser.add_argument('--dataset', type=str, default='Mazes', \
                        help='dataset (dafault: Mazes)')
    parser.add_argument('--train_data_dir', type=str, default="../dataset/mazes-torch/train", \
                        help='location of train data')
    parser.add_argument('--test_data_dir', type=str, default="../dataset/mazes-torch/test", \
                        help='location of test data')
    parser.add_argument('--root_log_dir', type=str, default="../logs", \
                        help='root location of log')
    parser.add_argument('--log_dir', type=str, default='GQN', \
                        help='log directory (default: GQN)')
    parser.add_argument('--log_interval', type=int, default=100, \
                        help='interval number of steps for logging')
    parser.add_argument('--save_interval', type=int, default=50000, \
                        help='interval number of steps for saveing models')
    parser.add_argument('--workers', type=int, default=0, \
                        help='number of data loading workers')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0], \
                        help='list of CUDA devices (default: [0])')
    parser.add_argument('--representation', type=str, default='pool', \
                        help='representation network (default: pool)')
    parser.add_argument('--layers', type=int, default=12, \
                        help='number of generative layers (default: 12)')
    parser.add_argument('--shared_core', type=bool, default=False, \
                        help='whether to share the weights of the cores across generation steps (default: False)')
    parser.add_argument('--seed', type=int, default=None, \
                        help='random seed (default: None')
    parser.add_argument('--attention', type=bool, default=False, \
                        help='apply attention to model (default: False)')
    parser.add_argument('--resize', type=bool, default=False, \
                        help='resizing images while training (default: False)')
    parser.add_argument('--image_size', type=int, default=64, \
                        help='image resizing width (default: 64)')
    parser.add_argument('--chkpt', type=int, default=0, \
                        help='checkpoint to resume training')
    args = parser.parse_args()

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"

    # Check-point
    chkpt = args.chkpt
    
    # Seed
    if args.seed!=None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Dataset directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # Number of workers to load data
    num_workers = args.workers

    # Log
    log_interval_num = args.log_interval
    save_interval_num = args.save_interval
    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'models'))
    os.mkdir(os.path.join(log_dir,'runs'))

    # TensorBoardX
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    # Dataset
    if args.resize:
        train_dataset = GQNDataset(root_dir=train_data_dir, target_transform=transform_viewpoint, image_size=args.image_size)
        test_dataset = GQNDataset(root_dir=test_data_dir, target_transform=transform_viewpoint, image_size=args.image_size)
    else:
        train_dataset = GQNDataset(root_dir=train_data_dir, target_transform=transform_viewpoint)
        test_dataset = GQNDataset(root_dir=test_data_dir, target_transform=transform_viewpoint)
    D = args.dataset

    # Pixel standard-deviation
    sigma_i, sigma_f = 2.0, 0.7
    sigma = sigma_i

    # Number of scenes over which each weight update is computed
    B = args.batch_size
    
    # Number of generative layers
    L =args.layers

    # Maximum number of training steps
    S_max = args.gradient_steps

    # Define model
    if args.attention:
        model = GQNAttention(L=L, shared_core=args.shared_core).to(device)
    else:
        model = GQN(representation=args.representation, L=L, shared_core=args.shared_core).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)

    # Update model if checkpoint was given
    if chkpt > 0:
        filename = log_dir + "/models/model-{}.pt".format(chkpt)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, filename)
    
    if len(args.device_ids)>1:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True, **kwargs)

    train_iter = iter(train_loader)

    #TODO: save and restore the sampled test data when resume training from checkpoint
    x_data_test, v_data_test = next(iter(test_loader))

    # Training Iterations
    #for t in tqdm(range(S_max)):
    for t in trange(chkpt+1, S_max, total=S_max, initial=chkpt+1):
        try:
            x_data, v_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_data, v_data = next(train_iter)

        x_data = x_data.to(device)
        v_data = v_data.to(device)
        x, v, x_q, v_q = sample_batch(x_data, v_data, D)
        elbo = model(x, v, v_q, x_q, sigma)
        
        # Logs
        writer.add_scalar('train_loss', -elbo.mean(), t)
             
        with torch.no_grad():
            # Write logs to TensorBoard
            if t % log_interval_num == 0:
                x_data_test = x_data_test.to(device)
                v_data_test = v_data_test.to(device)

                x_test, v_test, x_q_test, v_q_test = sample_batch(x_data_test, v_data_test, D, M=3, seed=0)
                elbo_test = model(x_test, v_test, v_q_test, x_q_test, sigma)
                
                if len(args.device_ids)>1:
                    kl_test = model.module.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                    x_q_rec_test = model.module.reconstruct(x_test, v_test, v_q_test, x_q_test)
                    x_q_hat_test = model.module.generate(x_test, v_test, v_q_test)
                else:
                    kl_test = model.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                    x_q_rec_test = model.reconstruct(x_test, v_test, v_q_test, x_q_test)
                    x_q_hat_test = model.generate(x_test, v_test, v_q_test)

                writer.add_scalar('test_loss', -elbo_test.mean(), t)
                writer.add_scalar('test_kl', kl_test.mean(), t)
                writer.add_image('test_ground_truth', make_grid(x_q_test, 6, pad_value=1), t)
                writer.add_image('test_reconstruction', make_grid(x_q_rec_test, 6, pad_value=1), t)
                writer.add_image('test_generation', make_grid(x_q_hat_test, 6, pad_value=1), t)

            if t % save_interval_num == 0:
                state = {'epoch': t+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), }
                torch.save(state, log_dir + "/models/model-{}.pt".format(t))
                #torch.save(model.state_dict(), log_dir + "/models/model-{}.pt".format(t))

        # Compute empirical ELBO gradients
        (-elbo.mean()).backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update optimizer state
        scheduler.step()

        # Pixel-variance annealing
        sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - t/(2e5)), sigma_f)
        
    torch.save(model.state_dict(), log_dir + "/models/model-final.pt")  
    writer.close()

