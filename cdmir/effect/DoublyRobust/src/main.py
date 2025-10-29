import argparse
import torch
import time
import utils as utils
import numpy as np
from targetedModel_DoubleBSpline import TargetedModel_DoubleBSpline
from experiment import Experiment


'''
Since the datasets are too large, over 100MB ( maximum file size in openreview), please find datasets 
in https://github.com/songjiang0909/Causal-Inference-on-Networked-Data

Our code is based on https://github.com/songjiang0909/Causal-Inference-on-Networked-Data
Thank them for their code!
'''

"""1. Basic Experimental Configuration (Causal Data and Environment)"""
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=7, help='Use CUDA training.')
parser.add_argument('--seed', type=int, default=24, help='Random seed. RIP KOBE')
parser.add_argument('--dataset', type=str, default='BC')
parser.add_argument('--flipRate', type=float, default=1)
# parser.add_argument('--dataset', type=str, default='Flickr')
# parser.add_argument('--flipRate', type=float, default=0)
# ["BC","Flickr"]
parser.add_argument('--expID', type=int, default=0)

"""
 2. Set dual robust framework hyperparameters
"""
parser.add_argument('--alpha', type=float, default=.5, help='trade-off of p(t|x).')
parser.add_argument('--gamma', type=float, default=1., help='trade-off of p(z|x).')

parser.add_argument('--num_grid', type=int, default=20, help='Number of epochs to train.')  # 10000
parser.add_argument('--epochs', type=int, default=160, help='Number of epochs to train.')  # 8000/(1step+2step)=160

parser.add_argument('--beta', type=float, default=20, help='trade-off of targeted regur in TargetedModel')
parser.add_argument('--tr_knots', type=float, default=0.1, help='trade-off of targeted regur in TargetedModel')

"""
3. Training process configuration (causal model optimization strategy)
"""
parser.add_argument('--lr_1step', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--lr_2step', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--pre_train_step', type=int, default=0, help='momentum for optimizer 1')

parser.add_argument('--pstep', type=int, default=1, help='epoch of training')  # default 1
parser.add_argument('--iter_2step', type=int, default=50, help='epoch of training fluctation param')

parser.add_argument('--weight_decay_tr', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')  # 1e-5
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')  # 1e-5

# iter_1step = pstep

parser.add_argument('--dstep', type=int, default=50, help='epoch of training discriminator')
parser.add_argument('--d_zstep', type=int, default=50, help='epoch of training discriminator_z')

parser.add_argument('--normy', type=int, default=1)

"""
4. Model structure configuration (causal features and network design)
"""
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')  # 32
parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate (1 - keep probability).')
parser.add_argument('--save_intermediate', type=int, default=1,
                    help='Save training curve and imtermediate embeddings')  # default 1

parser.add_argument('--model', type=str, default='TargetedModel_DoubleBSpline', help='Models or baselines')
parser.add_argument('--loss_2step_with_ly', type=int, default=0,
                    help='loss in 2 step contains loss of y, 0 means no, 1 means yes')
parser.add_argument('--loss_2step_with_ltz', type=int, default=0,
                    help='loss in 2 step contains loss of tz, 0 means no, 1 means yes')


parser.add_argument('--alpha_base', type=float, default=0.5, help='trade-off of balance for baselines.')
parser.add_argument('--printDisc', type=int, default=0, help='Print discriminator result for debug usage')
parser.add_argument('--printDisc_z', type=int, default=0, help='Print discriminator_z result for debug usage')
parser.add_argument('--printPred', type=int, default=1, help='Print encoder-predictor result for debug usage')
parser.add_argument('--search', type=int, default=0, help='parameter searching')

# Record experiment start time
startTime = time.time()

# Parse command-line arguments
args = parser.parse_args()

# Print configuration for verification
print(args)

args.cuda = args.cuda and torch.cuda.is_available()

# Set random seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

trainA, trainX, trainT, cfTrainT, POTrain, cfPOTrain, valA, valX, valT, cfValT, POVal, cfPOVal, testA, testX, testT, cfTestT, POTest, cfPOTest, \
    train_t1z1, train_t1z0, train_t0z0, train_t0z7, train_t0z2, val_t1z1, val_t1z0, val_t0z0, val_t0z7, val_t0z2,\
    test_t1z1, test_t1z0, test_t0z0, test_t0z7, test_t0z2 = utils.load_data(args)

args.beta = args.beta * (trainX.shape[0] ** (-0.5))
print(args.beta)

# normalization
# print('dim='+str(trainX.shape[1]))
# for i in range(trainX.shape[1]):
#     mu_trainX, std_trainX = torch.mean(trainX[:,i]), torch.std(trainX[:,i])
#     trainX[:,i] = (trainX[:,i]-mu_trainX)/std_trainX
#     testX[:,i] = (testX[:,i]-mu_trainX)/std_trainX
#     valX[:,i] = (valX[:,i]-mu_trainX)/std_trainX

if args.model == "TargetedModel_DoubleBSpline":
    """
        Initialize dual B-spline causal model

        Model Features:
        1.Handles networked data with graph-structured dependencies
        2.Combines B-spline basis functions for non-linear effect modeling
        3.Estimates both individual treatment effects (ITE) and peer effects
    """
    print(args.num_grid)
    # Call the TargetedModel_DoubleBSpline method to initializes the TNet model"""
    model = TargetedModel_DoubleBSpline(Xshape=trainX.shape[1], hidden=args.hidden, dropout=args.dropout,
                                        num_grid=args.num_grid, tr_knots=args.tr_knots)

# Initialize experiment manager with data and model
exp = Experiment(args, model, trainA, trainX, trainT, cfTrainT, POTrain, cfPOTrain, valA, valX, valT, cfValT, POVal,
                 cfPOVal, testA, testX, testT, cfTestT, POTest, cfPOTest,
                 train_t1z1, train_t1z0, train_t0z0, train_t0z7, train_t0z2, val_t1z1, val_t1z0, val_t0z0, val_t0z7,
                 val_t0z2, test_t1z1, test_t1z0, test_t0z0, test_t0z7, test_t0z2)

"""Train the model"""
try:
    # Execute training pipeline (1-step + 2-step optimization)
    exp.train()
except KeyboardInterrupt:
    # exp.predict()
    pass

try:
    # Run prediction on test set and compute causal metrics
    exp.predict()
except KeyboardInterrupt:
    # exp.predict()
    pass
"""Moel Predicting"""

# if args.model == "NetEsimator" and args.save_intermediate:
#     exp.save_curve()
#     exp.save_embedding()

"""Print the final result"""
print("Time usage:{:.4f} mins".format((time.time() - startTime) / 60))
print("================================Setting again================================")
print("Model:{} Dataset:{}, expID:{}, filpRate:{}, alpha:{}, gamma:{}".format(args.model, args.dataset, args.expID,
                                                                              args.flipRate, args.alpha, args.gamma))
print(args)
print("================================BYE================================")
