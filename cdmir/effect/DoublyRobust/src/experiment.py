import torch
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
import utils as utils
import numpy as np
import os

def criterion(out, y):
    """
        Custom mean squared error (MSE) loss function
        :param out (Tensor): Model predictions
        :param y (Tensor): Ground truth values

        Returns:
            Tensor: MSE between predictions and ground truth
        """
    return ((out - y) ** 2).mean()

class Experiment():

    def __init__(self, args, model, trainA, trainX, trainT, cfTrainT, POTrain, cfPOTrain, valA, valX, valT, cfValT,
                 POVal, cfPOVal, testA, testX, testT, cfTestT, POTest, cfPOTest,
                 train_t1z1, train_t1z0, train_t0z0, train_t0z7, train_t0z2, val_t1z1, val_t1z0, val_t0z0, val_t0z7,
                 val_t0z2, test_t1z1, test_t1z0, test_t0z0, test_t0z7, test_t0z2):
        """
                This class is to use configuration, model, and data initialization experiments

                :param args (Namespace): Experiment configuration (hyperparameters, model type, etc.)
                :param model (nn.Module): Causal model instance (e.g., NetEsimator, TargetedModel_DoubleBSpline)
                :param trainA/valA/testA (Tensor): Training/validation/test adjacency matrices (graph structure)
                :param trainX/valX/testX (Tensor): Training/validation/test node feature matrices
                :param trainT/valT/testT (Tensor): Training/validation/test treatment variables
                :param cfTrainT/cfValT/cfTestT (Tensor): Counterfactual treatment variables
                :param POTrain/POVal/POTest (Tensor): Factual outcomes
                :param cfPOTrain/cfPOVal/cfPOTest (Tensor): Counterfactual outcomes
                :param train_t1z1, ..., test_t0z2 (Tensor): Ground truth causal effects for evaluation
                """
        super(Experiment, self).__init__()

        self.args = args
        self.model = model
        # Configure optimizers based on model type
        if self.args.model == "NetEsimator":
            # Separate optimizers for discriminator, Z-discriminator, and encoder-predictor
            self.optimizerD = optim.Adam([{'params': self.model.discriminator.parameters()}], lr=self.args.lrD,
                                         weight_decay=self.args.weight_decay)
            self.optimizerD_z = optim.Adam([{'params': self.model.discriminator_z.parameters()}], lr=self.args.lrD_z,
                                           weight_decay=self.args.weight_decay)
            self.optimizerP = optim.Adam(
                [{'params': self.model.encoder.parameters()}, {'params': self.model.predictor.parameters()}],
                lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif  self.args.model == "TargetedModelOnlyT"  \
                or self.args.model == 'TargetedModelOnlyT2' or self.args.model == 'TargetedModelOnlyZ' \
                or self.args.model == "TargetedModelWithoutTR":
            self.optimizerT = optim.Adam(self.model.parameters(), lr=self.args.lrT, weight_decay=self.args.weight_decay)
        elif self.args.model == "TargetedModel" or self.args.model == "TargetedModelLogist" :
            self.optimizerT = optim.Adam(self.model.parameters(), lr=self.args.lrTR_TZ, weight_decay=self.args.weight_decay)

        elif self.args.model == "TargetedModel_DoubleBSpline" \
                or self.args.model == 'TargetedModel_DBS_miss':
            self.optimizerT = optim.Adam(self.model.parameter_base(), lr=self.args.lr_1step, weight_decay=self.args.weight_decay)
            self.optimizer2step = optim.Adam(self.model.parameters(), lr=self.args.lr_2step, weight_decay=self.args.weight_decay_tr)


        elif self.args.model == "TargetedModel2Step" or self.args.model == "TargetedModel2StepAblation":
            self.optimizer1step = optim.Adam(self.model.parameters(), lr=self.args.lr_1step, weight_decay=self.args.weight_decay)
            self.optimizer2step = optim.Adam(self.model.parameters(), lr=self.args.lr_2step, weight_decay=self.args.weight_decay)
        else:
            self.optimizerB = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.cuda:
            self.model = self.model.cuda()
        print("================================Model================================")
        print(self.model)

        # -------------------- Data Initialization --------------------
        self.Tensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        self.trainA = trainA
        self.trainX = trainX
        self.trainT = trainT
        # print(torch.sum(trainT))
        self.trainZ = self.compute_z(self.trainT, self.trainA)
        self.cfTrainT = cfTrainT
        self.POTrain = POTrain
        self.cfPOTrain = cfPOTrain
        self.valA = valA
        self.valX = valX
        self.valT = valT
        self.valZ = self.compute_z(self.valT, self.valA)
        self.cfValT = cfValT
        self.POVal = POVal
        self.cfPOVal = cfPOVal
        self.testA = testA
        self.testX = testX
        self.testT = testT
        self.testZ = self.compute_z(self.testT, self.testA)
        self.cfTestT = cfTestT
        self.POTest = POTest
        self.cfPOTest = cfPOTest

        # Ground truth causal effects (different treatment/peer scenarios)
        self.z_1 = 0.7
        self.z_2 = 0.2
        self.train_t1z1 = self.Tensor(train_t1z1)
        self.train_t1z0 = self.Tensor(train_t1z0)
        self.train_t0z0 = self.Tensor(train_t0z0)
        self.train_t0z7 = self.Tensor(train_t0z7)
        self.train_t0z2 = self.Tensor(train_t0z2)
        # Validation/test causal effects (similar structure)
        self.val_t1z1 = self.Tensor(val_t1z1)
        self.val_t1z0 = self.Tensor(val_t1z0)
        self.val_t0z0 = self.Tensor(val_t0z0)
        self.val_t0z7 = self.Tensor(val_t0z7)
        self.val_t0z2 = self.Tensor(val_t0z2)

        self.test_t1z1 = self.Tensor(test_t1z1)
        self.test_t1z0 = self.Tensor(test_t1z0)
        self.test_t0z0 = self.Tensor(test_t0z0)
        self.test_t0z7 = self.Tensor(test_t0z7)
        self.test_t0z2 = self.Tensor(test_t0z2)

        """PO normalization if any"""
        self.YFTrain, self.YCFTrain = utils.PO_normalize(self.args.normy, self.POTrain, self.POTrain, self.cfPOTrain)
        self.YFVal, self.YCFVal = utils.PO_normalize(self.args.normy, self.POTrain, self.POVal, self.cfPOVal)
        self.YFTest, self.YCFTest = utils.PO_normalize(self.args.normy, self.POTrain, self.POTest, self.cfPOTest)

        # -------------------- Loss Functions --------------------
        self.loss = nn.MSELoss(reduction='mean')
        # self.loss = criterion
        self.d_zLoss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.peheLoss = nn.MSELoss(reduction='mean')

        self.alpha = self.Tensor([self.args.alpha])
        self.gamma = self.Tensor([self.args.gamma])
        self.alpha_base = self.Tensor([self.args.alpha_base])
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
            self.loss = self.loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.peheLoss = self.peheLoss.cuda()

        self.lossTrain = []
        self.lossVal = []
        self.lossTest = []
        self.lossCFTrain = []
        self.lossCFVal = []
        self.lossCFTest = []

        self.dissTrain = []
        self.dissVal = []
        self.dissTrainHalf = []
        self.dissValHalf = []
        self.diss_zTrain = []
        self.diss_zVal = []
        self.diss_zTrainHalf = []
        self.diss_zValHalf = []

        self.labelTrain = []
        self.labelVal = []
        self.labelTest = []
        self.labelTrainCF = []
        self.labelValCF = []
        self.labelTestCF = []

        self.predT = []
        self.labelT = []

    def get_peheLoss(self, y1pred, y0pred, y1gt, y0gt):
        """
                Compute PEHE (Precision in Estimating Heterogeneous Effects)
                This function is to measure the root mean squared error
                between predicted and true individual treatment effects (ITE)


                :param y1pred (Tensor): Predicted outcomes under treatment T=1
                :param y0pred (Tensor): Predicted outcomes under treatment T=0
                :param y1gt (Tensor): Ground truth outcomes under T=1
                :param y0gt (Tensor): Ground truth outcomes under T=0

                Returns:
                    Tensor: PEHE value (√MSE)
                """
        pred = y1pred - y0pred
        gt = y1gt - y0gt
        return torch.sqrt(self.peheLoss(pred, gt))

    def get_ateLoss(self, y1pred, y0pred, y1gt, y0gt):
        """
               This function is to measure the absolute difference between predicted and true average treatment effects
               :param y1pred/y0pred (Tensor): Predicted outcomes under T=1/0
               :param y1gt/y0gt (Tensor): Ground truth outcomes under T=1/0

               Returns:
                   Tensor: Absolute ATE error
               """

        pred = y1pred - y0pred
        gt = y1gt - y0gt
        return torch.abs(torch.mean(pred) - torch.mean(gt))

    def compute_z(self, T, A):
        """
                This function is to compute peer effect variable Z (neighbor-averaged treatment)
                :param T (Tensor): Treatment variable (shape [num_nodes])
                :param A (Tensor): Adjacency matrix (shape [num_nodes, num_nodes])
                Returns:
                    Tensor: Z values (neighbor-averaged T, shape [num_nodes])
        """
        # print ("A has identity?: {}".format(not (A[0][0]==0 and A[24][24]==0 and A[8][8]==0)))
        neighbors = torch.sum(A, 1)
        neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        return neighborAverageT

    def train_one_step_discriminator(self, A, X, T):
        """
               This function is to train the discriminator to distinguish true vs. synthetic treatment variables
               :param A (Tensor): Adjacency matrix
               :param X (Tensor): Node features
               :param T (Tensor): True treatment variable

               Returns:
                   (Tensor, Tensor): Discriminator loss and auxiliary loss (vs. 0.5)
               """
        self.model.train()
        self.optimizerD.zero_grad()
        pred_treatmentTrain, _, _, _, _ = self.model(A, X, T)
        discLoss = self.bce_loss(pred_treatmentTrain.reshape(-1), T)
        num = pred_treatmentTrain.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))
        discLoss.backward()
        self.optimizerD.step()

        return discLoss, discLosshalf

    def eval_one_step_discriminator(self, A, X, T):
        """
                Evaluate treatment discriminator performance
                :param A/X/T (Tensor): Input data (adjacency, features, treatment)
                Returns:
                    (Tensor, Tensor, Tensor, Tensor): Discriminator loss, auxiliary loss, predictions, labels
        """
        self.model.eval()
        pred_treatment, _, _, _, _ = self.model(A, X, T)
        discLossWatch = self.bce_loss(pred_treatment.reshape(-1), T)
        num = pred_treatment.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatment.reshape(-1), self.Tensor(target05))

        return discLossWatch, discLosshalf, pred_treatment, T

    def train_discriminator(self, epoch):
        """
               This function is to train treatment discriminator for multiple steps
               :param epoch (int): Current training epoch
        """
        for ds in range(self.args.dstep):
            # Train on training data
            discLoss, discLossTrainhalf = self.train_one_step_discriminator(self.trainA, self.trainX, self.trainT)
            # Evaluate on validation/test data
            discLossVal, discLossValhalf, _, _ = self.eval_one_step_discriminator(self.valA, self.valX, self.valT)
            discLossTest, discLossTesthalf, _, _ = self.eval_one_step_discriminator(self.testA, self.testX, self.testT)
            # Log metrics at the last step
            if ds == self.args.dstep - 1:
                if self.args.printDisc:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                          'dLoss:{:05f}'.format(discLoss),
                          'dLossVal:{:05f}'.format(discLossVal),
                          'dLossTest:{:05f}'.format(discLossTest),
                          'dLoss0.5:{:05f}'.format(discLossTrainhalf),
                          'dLossVal0.5:{:05f}'.format(discLossValhalf),
                          'dLossTest0.5:{:05f}'.format(discLossTesthalf),
                          )
                self.dissTrain.append(discLoss.detach().cpu().numpy())
                self.dissVal.append(discLossVal.detach().cpu().numpy())
                self.dissTrainHalf.append(discLossTrainhalf.detach().cpu().numpy())
                self.dissValHalf.append(discLossValhalf.detach().cpu().numpy())

    def train_fluctuation_param(self, epoch):
        """
                Optimizes parameters that refine causal effect estimates using dual-robust regularization
                :param epoch (int): Current training epoch
        """
        for ds in range(self.args.iter_2step):
            # self.model.zero_grad()
            self.model.train()
            self.optimizer2step.zero_grad()

            A, X, T, Y = self.trainA, self.trainX, self.trainT, self.YFTrain

            g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
            # Compute dual-robust loss (combines outcome prediction and inverse propensity weighting)
            Loss_TR = self.loss(
                Q_hat.reshape(-1) +
                epsilon.reshape(-1) * (1 / (g_T_hat.reshape(-1).detach() * g_Z_hat.reshape(-1).detach() + 1e-6))
                #  - (1 - T) / (  (1 - g_T_hat.reshape(-1)) * g_Z_hat.reshape(-1) + 1e-6))
                , Y)
            loss_train = self.args.beta * Loss_TR
            # Optional auxiliary losses for outcome prediction and propensity scores
            if self.args.loss_2step_with_ly == 1:
                Q_Loss = self.loss(Q_hat.reshape(-1), Y)
                loss_train = loss_train + Q_Loss

            if self.args.loss_2step_with_ltz == 1:
                g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
                g_Z_Loss = - torch.log(g_Z_hat + 1e-6).mean()
                loss_train = loss_train + self.args.alpha * g_T_Loss + self.args.gamma * g_Z_Loss
            # Backpropagation and parameter update
            loss_train.backward()
            self.optimizer2step.step()
            # return discLoss, discLosshalf

            g_T_hat_val, g_Z_hat_val, Q_hat_val, epsilon_val, _, _ = self.model(self.valA, self.valX,
             self.valT)

            pLoss_val = self.loss(
                Q_hat_val.reshape(-1) +
                epsilon_val.reshape(-1) * (1 / (g_T_hat_val.reshape(-1).detach() * g_Z_hat_val.reshape(-1).detach() + 1e-6))
                #  - (1 - T) / (  (1 - g_T_hat.reshape(-1)) * g_Z_hat.reshape(-1) + 1e-6))
                , self.YFVal
            ) * self.args.beta


        individual_effect_train, peer_effect_train, total_effect_train, \
            ate_individual_train, ate_peer_train, ate_total_train \
            = self.compute_effect_pehe(self.trainA, self.trainX, self.train_t1z1, self.train_t1z0,
                                       self.train_t0z7, self.train_t0z2, self.train_t0z0)

        # individual_effect_val, peer_effect_val, total_effect_val,\
        #     ate_individual_val, ate_peer_val, ate_total_val\
        #     = self.compute_effect_pehe(self.valA, self.valX, self.val_t1z1, self.val_t1z0,
        #                                self.val_t0z7, self.val_t0z2, self.val_t0z0)
        individual_effect_te, peer_effect_te, total_effect_te, \
            ate_individual_te, ate_peer_te, ate_total_te \
            = self.compute_effect_pehe(self.testA, self.testX, self.test_t1z1, self.test_t1z0,
                                       self.test_t0z7, self.test_t0z2, self.test_t0z0)

        # Print training metrics if enabled
        if self.args.printPred:
            print('2_Epoch: {:04d}'.format(epoch + 1),
                  'tLossTrain:{:.4f}'.format(loss_train.item()),
                  'tLossVal:{:.4f}'.format(pLoss_val.item()),
                  # 'dLossTrain:{:.4f}'.format(dLoss_train.item()),
                  # 'dLossVal:{:.4f}'.format(dLoss_val.item()),
                  # 'd_zLossTrain:{:.4f}'.format(d_zLoss_train.item()),
                  # 'd_zLossVal:{:.4f}'.format(d_zLoss_val.item()),
                  #
                  # 'CFpLossTrain:{:.4f}'.format(cfPLoss_train.item()),
                  # 'CFpLossVal:{:.4f}'.format(cfPLoss_val.item()),
                  # 'CFdLossTrain:{:.4f}'.format(cfDLoss_train.item()),
                  # 'CFdLossVal:{:.4f}'.format(cfDLoss_val.item()),
                  # 'CFd_zLossTrain:{:.4f}'.format(cfD_zLoss_train.item()),
                  # 'CFd_zLossVal:{:.4f}'.format(cfD_zLoss_val.item()),

                  'iE_train:{:.4f}'.format(individual_effect_train.item()),
                  'PE_train:{:.4f}'.format(peer_effect_train.item()),
                  'TE_train:{:.4f}'.format(total_effect_train.item()),

                  '\t',
                  'iE_te:{:.4f}'.format(individual_effect_te.item()),
                  'PE_te:{:.4f}'.format(peer_effect_te.item()),
                  'TE_te:{:.4f}'.format(total_effect_te.item()),
                  #
                  # 'AiE_train:{:.4f}'.format(ate_individual_train.item()),
                  # 'APE_train:{:.4f}'.format(ate_peer_train.item()),
                  # 'ATE_train:{:.4f}'.format(ate_total_train.item()),
                  'AiE_te:{:.4f}'.format(ate_individual_te.item()),
                  'APE_te:{:.4f}'.format(ate_peer_te.item()),
                  'ATE_te:{:.4f}'.format(ate_total_te.item()),

                  )



    def train_one_step_discriminator_z(self, A, X, T):
        """
                This function is to train the Z-discriminator to distinguish true vs. synthetic peer effect variables (Z).


                :param A (Tensor): Adjacency matrix (shape [num_nodes, num_nodes])
                :param X (Tensor): Node feature matrix (shape [num_nodes, num_feats])
                :param T (Tensor): Treatment variable (shape [num_nodes])

                Returns:
                    (Tensor, Tensor): Z-discriminator loss (scalar), auxiliary loss (vs. random targets, scalar)
                """
        self.model.train()
        self.optimizerD_z.zero_grad()
        # Forward pass to get Z predictions
        _, pred_zTrain, _, _, labelZ = self.model(A, X, T)
        discLoss_z = self.d_zLoss(pred_zTrain.reshape(-1), labelZ)
        # Auxiliary loss to prevent overfitting (predict random targets)
        num = pred_zTrain.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf_z = self.loss(pred_zTrain.reshape(-1), self.Tensor(target))
        # Backpropagation and parameter update
        discLoss_z.backward()
        self.optimizerD_z.step()

        return discLoss_z, discLosshalf_z

    def eval_one_step_discriminator_z(self, A, X, T):
        """
               Evaluate peer effect discriminator performance
               :param A/X/T (Tensor): Input data (adjacency, features, treatment)
               Returns:
                       Z-discriminator loss (scalar),
                       auxiliary loss (vs. random targets, scalar),
                       Z predictions (shape [num_nodes]),
                       true Z labels (shape [num_nodes])
               """
        self.model.eval()
        _, pred_z, _, _, labelZ = self.model(A, X, T)
        discLossWatch = self.d_zLoss(pred_z.reshape(-1), labelZ)
        num = pred_z.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf = self.loss(pred_z.reshape(-1), self.Tensor(target))

        return discLossWatch, discLosshalf, pred_z, labelZ

    def train_discriminator_z(self, epoch):
        """
               This function is to train peer effect discriminator for multiple steps
               :param epoch (int): Current training epoch
               """

        # Iterate over specified discriminator steps
        for dzs in range(self.args.d_zstep):
            # Train on training data
            discLoss_z, discLoss_zTrainRandom = self.train_one_step_discriminator_z(self.trainA, self.trainX,
                                                                                    self.trainT)
            # Evaluate on validation/test data
            discLoss_zVal, discLoss_zValRandom, _, _ = self.eval_one_step_discriminator_z(self.valA, self.valX,
                                                                                          self.valT)
            discLoss_zTest, discLoss_zTestRandom, _, _ = self.eval_one_step_discriminator_z(self.testA, self.testX,
                                                                                            self.testT)
            # Log metrics at the final step of the epoch
            if dzs == self.args.d_zstep - 1:
                if self.args.printDisc_z:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                          'd_zLoss:{:05f}'.format(discLoss_z),
                          'd_zLossVal:{:05f}'.format(discLoss_zVal),
                          'd_zLossTest:{:05f}'.format(discLoss_zTest),
                          'd_zLRanTrain:{:05f}'.format(discLoss_zTrainRandom),
                          'd_zLRanVal:{:05f}'.format(discLoss_zValRandom),
                          'd_zLRanTest:{:05f}'.format(discLoss_zTestRandom),
                          )
                self.diss_zTrain.append(discLoss_z.detach().cpu().numpy())
                self.diss_zVal.append(discLoss_zVal.detach().cpu().numpy())
                self.diss_zTrainHalf.append(discLoss_zTrainRandom.detach().cpu().numpy())
                self.diss_zValHalf.append(discLoss_zValRandom.detach().cpu().numpy())

    def train_one_step_encoder_predictor(self, A, X, T, Y):
        """
                Trains the main model components (encoder, predictor) to estimate outcomes
                and causal effects,with architecture-specific implementations for different models.
                :param A (Tensor): Adjacency matrix (shape [num_nodes, num_nodes])
                :param X (Tensor): Node features (shape [num_nodes, num_feats])
                :param T (Tensor): Treatment variable (shape [num_nodes])
                :param Y (Tensor): Observed outcomes (shape [num_nodes])

                Returns:
                    (Tensor, Tensor, Tensor, Tensor):
                        Total loss (scalar),
                        outcome prediction loss (scalar),
                        discriminator loss (scalar),
                        Z-discriminator loss (scalar)
                """
        if self.args.model == "NetEsimator":
            # NetEsimator-specific training (adversarial components)
            self.model.zero_grad()
            self.model.train()
            self.optimizerP.zero_grad()
            pred_treatmentTrain, pred_zTrain, pred_outcomeTrain, _, _ = self.model(A, X, T)
            pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
            num = pred_treatmentTrain.shape[0]
            target05 = [0.5 for _ in range(num)]
            # Discriminator losses (auxiliary tasks to prevent mode collapse)
            dLoss = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))
            num = pred_zTrain.shape[0]
            target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
            d_zLoss = self.d_zLoss(pred_zTrain.reshape(-1), target)
            # Total loss with regularization
            loss_train = pLoss + dLoss * self.alpha + d_zLoss * self.gamma
            loss_train.backward()
            self.optimizerP.step()

        elif self.args.model == 'TargetedModel_DoubleBSpline':
            # Double B-spline model training (dual-robust estimation)
            # self.model.zero_grad()
            self.model.train()
            self.optimizerT.zero_grad()
            g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
            # Base losses: outcome prediction + propensity scores
            Q_Loss = self.loss(Q_hat.reshape(-1), Y)
            g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
            g_Z_Loss = - torch.log(g_Z_hat + 1e-6).mean()
            Loss_base = Q_Loss + self.args.alpha * g_T_Loss + self.args.gamma * g_Z_Loss
            # Dual-robust loss
            Loss_TR = self.loss(
                Q_hat.reshape(-1) +
                epsilon.reshape(-1) * (1 / (g_T_hat.detach().reshape(-1) * g_Z_hat.detach().reshape(-1) + 1e-6) )
                        #  - (1 - T) / (  (1 - g_T_hat.reshape(-1)) * g_Z_hat.reshape(-1) + 1e-6))
                ,Y)

            loss_train = Loss_base + self.args.beta * Loss_TR
            loss_train.backward()
            self.optimizerT.step()
            pLoss, dLoss, d_zLoss = Q_Loss, g_T_Loss, g_Z_Loss

        else:
            self.model.zero_grad()
            self.model.train()
            self.optimizerB.zero_grad()
            _, _, pred_outcomeTrain, rep, _ = self.model(A, X, T)
            pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
            if self.args.model in set(["TARNet", "TARNet_INTERFERENCE"]):
                loss_train = pLoss
                dLoss = self.Tensor([0])
            else:
                rep_t1, rep_t0 = rep[(T > 0).nonzero()], rep[(T < 1).nonzero()]
                dLoss, _ = utils.wasserstein(rep_t1, rep_t0, cuda=self.args.cuda)
                loss_train = pLoss + self.alpha_base * dLoss
            d_zLoss = self.Tensor([-1])
            loss_train.backward()
            self.optimizerB.step()

        return loss_train, pLoss, dLoss, d_zLoss

    def eval_one_step_encoder_predictor(self, A, X, T, Y):
        """
                Evaluate encoder-predictor module performance
                :param A/X/T/Y (Tensor): Input data (adjacency, features, treatment, outcomes)

                Returns:
                    (Tensor, Tensor, Tensor, Tensor):
                        Total loss (scalar),
                        outcome prediction loss (scalar),
                        discriminator loss (scalar),
                        Z-discriminator loss (scalar)
                """
        self.model.eval()
        # 根据模型类型差异化计算损失
        if self.args.model == "NetEsimator":
            pred_treatment, pred_z, pred_outcome, _, _ = self.model(A, X, T)
            # 1. 结果预测损失（MSE）
            pLossV = self.loss(pred_outcome.reshape(-1), Y)
            # 2. 治疗判别器损失（强制预测分布均衡，防止过拟合）
            num = pred_treatment.shape[0]
            target05 = [0.5 for _ in range(num)]
            dLossV = self.loss(pred_treatment.reshape(-1), self.Tensor(target05))
            num = pred_z.shape[0]
            # 3. Z判别器损失（预测值与随机目标的MSE，防止过拟合）
            target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
            d_zLossV = self.d_zLoss(pred_z.reshape(-1), target)
            # 总损失：结果预测 + 治疗判别器（α加权） + Z判别器（γ加权）
            loss_val = pLossV + dLossV * self.alpha + d_zLossV * self.gamma

        elif self.args.model == 'TargetedModel_DoubleBSpline':
            # 双稳健估计模型（TargetedModel_DoubleBSpline）的评估逻辑
            g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
            # 1. 基础损失（结果预测 + 倾向分数）
            Q_Loss = self.loss(Q_hat.reshape(-1), Y)
            g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
            g_Z_Loss = -torch.log(g_Z_hat + 1e-6).mean()
            Loss_base = Q_Loss + 0.5 * g_T_Loss + 0.5 * g_Z_Loss
            # 2. 双稳健损失（结合结果预测与逆倾向加权，优化因果效应）
            Loss_TR = self.loss(Y, Q_hat.reshape(-1) + epsilon.reshape(-1) / (
                    g_T_hat.reshape(-1) * g_Z_hat.reshape(-1) + 1e-6))
            # 总损失：基础损失 + 双稳健损失（加权1.0）
            loss_val = Loss_base + 1. * Loss_TR
            pLossV, dLossV, d_zLossV = Q_Loss, g_T_Loss, g_Z_Loss

        else:
            # 基线模型（如TARNet等）的评估逻辑
            # 前向传播获取结果预测和节点表示
            _, _, pred_outcome, rep, _ = self.model(A, X, T)
            # 1. 结果预测损失（MSE）
            pLossV = self.loss(pred_outcome.reshape(-1), Y)
            # 2. 表示对齐损失（仅非TARNet模型有效，衡量处理组与对照组表示的分布差异）
            if self.args.model in set(["TARNet", "TARNet_INTERFERENCE"]):
                loss_val = pLossV
                dLossV = self.Tensor([0])
            else:
                # 其他模型计算Wasserstein距离（处理组vs对照组表示）
                rep_t1, rep_t0 = rep[(T > 0).nonzero()], rep[(T < 1).nonzero()]
                dLossV, _ = utils.wasserstein(rep_t1, rep_t0, cuda=self.args.cuda)
                loss_val = self.args.alpha_base * dLossV
                # 总损失：预测损失 + 对齐损失
                loss_val = pLossV + self.args.alpha_base * dLossV
            d_zLossV = self.Tensor([-1])

        return loss_val, pLossV, dLossV, d_zLossV

    def compute_effect_pehe(self, A, X, gt_t1z1, gt_t1z0, gt_t0z7, gt_t0z2, gt_t0z0):
        """
               Evaluates model performance by comparing predicted vs. ground truth potential outcomes across
               various treatment (T) and peer effect (Z) scenarios.
               :param A (Tensor): Adjacency matrix of the network (shape [N, N])
               :param X (Tensor): Node feature matrix (shape [N, D])
               :param gt_t1z1 (Tensor): Ground truth outcome for T=1, Z=1 (shape [N])
               :param gt_t1z0 (Tensor): Ground truth outcome for T=1, Z=0 (shape [N])
               :param gt_t0z7 (Tensor): Ground truth outcome for T=0, Z=z₁ (shape [N])
               :param gt_t0z2 (Tensor): Ground truth outcome for T=0, Z=z₂ (shape [N])
               :param gt_t0z0 (Tensor): Ground truth outcome for T=0, Z=0 (shape [N])

               Returns:
                   tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                       individual_effect: PEHE for individual treatment effect (ITE)
                       peer_effect: PEHE for peer effect (PTE)
                       total_effect: PEHE for total treatment effect (TTE)
                       ate_individual: ATE for individual effect
                       ate_peer: ATE for peer effect
                       ate_total: ATE for total effect
               """
        num = X.shape[0]
        # Create treatment (T) and peer effect (Z) scenario tensors
        z_1s = self.Tensor(np.ones(num))
        z_0s = self.Tensor(np.zeros(num))
        z_07s = self.Tensor(np.zeros(num) + self.z_1)
        z_02s = self.Tensor(np.zeros(num) + self.z_2)
        t_1s = self.Tensor(np.ones(num))
        t_0s = self.Tensor(np.zeros(num))

        # Get model predictions for all treatment-peer combinations
        if self.args.model == 'TargetedModel_DoubleBSpline':
            # Specialized prediction for double B-spline model
            pred_outcome_t1z1 = self.model.infer_potential_outcome(A, X, t_1s, z_1s)
            pred_outcome_t1z0 = self.model.infer_potential_outcome(A, X, t_1s, z_0s)
            pred_outcome_t0z0 = self.model.infer_potential_outcome(A, X, t_0s, z_0s)
            pred_outcome_t0z7 = self.model.infer_potential_outcome(A, X, t_0s, z_07s)
            pred_outcome_t0z2 = self.model.infer_potential_outcome(A, X, t_0s, z_02s)
        else:
            # Generic prediction for other models
            _, _, pred_outcome_t1z1, _, _ = self.model(A, X, t_1s, z_1s)
            _, _, pred_outcome_t1z0, _, _ = self.model(A, X, t_1s, z_0s)
            _, _, pred_outcome_t0z0, _, _ = self.model(A, X, t_0s, z_0s)
            _, _, pred_outcome_t0z7, _, _ = self.model(A, X, t_0s, z_07s)
            _, _, pred_outcome_t0z2, _, _ = self.model(A, X, t_0s, z_02s)

        # Recover normalized potential outcomes to original scale
        pred_outcome_t1z1 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t1z1)
        pred_outcome_t1z0 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t1z0)
        pred_outcome_t0z0 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t0z0)
        pred_outcome_t0z7 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t0z7)
        pred_outcome_t0z2 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t0z2)

        # Compute PEHE (mean squared error) for different effect types
        individual_effect = self.get_peheLoss(pred_outcome_t1z0, pred_outcome_t0z0, gt_t1z0, gt_t0z0)
        peer_effect = self.get_peheLoss(pred_outcome_t0z7, pred_outcome_t0z2, gt_t0z7, gt_t0z2)
        total_effect = self.get_peheLoss(pred_outcome_t1z1, pred_outcome_t0z0, gt_t1z1, gt_t0z0)

        # Compute ATE (average difference) for different effect types
        ate_individual = self.get_ateLoss(pred_outcome_t1z0, pred_outcome_t0z0, gt_t1z0, gt_t0z0)
        ate_peer = self.get_ateLoss(pred_outcome_t0z7, pred_outcome_t0z2, gt_t0z7, gt_t0z2)
        ate_total = self.get_ateLoss(pred_outcome_t1z1, pred_outcome_t0z0, gt_t1z1, gt_t0z0)

        return individual_effect, peer_effect, total_effect, ate_individual, ate_peer, ate_total

    def train_encoder_predictor(self, epoch):
        """
        This function executes the training loop of the core model components (encoder and result predictor),
        Track training/validation losses and evaluate causal effects at the end of the epoch.
        :param Epoch (int): Index of the current training epoch
        """
        for epoch_num in range(self.args.pstep):
            # Train on training data
            loss_train, pLoss_train, dLoss_train, d_zLoss_train = self.train_one_step_encoder_predictor(self.trainA,
                                                                                                        self.trainX,
                                                                                                        self.trainT,
                                                                                                        self.YFTrain)
            # Validate on validation data
            loss_val, pLoss_val, dLoss_val, d_zLoss_val = self.eval_one_step_encoder_predictor(self.valA, self.valX,
                                                                                               self.valT, self.YFVal)
            # Track losses for later analysis
            self.lossTrain.append(loss_train.cpu().detach().numpy())
            self.lossVal.append(loss_val.cpu().detach().numpy())

            """CHECK CF"""
            # cfloss_train, cfPLoss_train, cfDLoss_train, cfD_zLoss_train = self.eval_one_step_encoder_predictor(
            #     self.trainA, self.trainX, self.cfTrainT, self.YCFTrain)
            # cfloss_val, cfPLoss_val, cfDLoss_val, cfD_zLoss_val = self.eval_one_step_encoder_predictor(self.valA,
            #                                                                                            self.valX,
            #                                                                                            self.cfValT,
            #                                                                                            self.YCFVal)
            # self.lossCFTrain.append(cfloss_train.cpu().detach().numpy())
            # self.lossCFVal.append(cfloss_val.cpu().detach().numpy())

        # Compute causal effects on train/test sets
        individual_effect_train, peer_effect_train, total_effect_train,\
            ate_individual_train, ate_peer_train, ate_total_train\
            = self.compute_effect_pehe(self.trainA, self.trainX, self.train_t1z1, self.train_t1z0,
                                       self.train_t0z7, self.train_t0z2, self.train_t0z0)

        # individual_effect_val, peer_effect_val, total_effect_val,\
        #     ate_individual_val, ate_peer_val, ate_total_val\
        #     = self.compute_effect_pehe(self.valA, self.valX, self.val_t1z1, self.val_t1z0,
        #                                self.val_t0z7, self.val_t0z2, self.val_t0z0)
        individual_effect_te, peer_effect_te, total_effect_te,\
            ate_individual_te, ate_peer_te, ate_total_te\
            = self.compute_effect_pehe(self.testA, self.testX, self.test_t1z1, self.test_t1z0,
                                       self.test_t0z7, self.test_t0z2, self.test_t0z0)

        if self.args.printPred:
            print('p_Epoch: {:04d}'.format(epoch + 1),
                  'pLossTrain:{:.4f}'.format(pLoss_train.item()),
                  'pLossVal:{:.4f}'.format(pLoss_val.item()),
                  # 'dLossTrain:{:.4f}'.format(dLoss_train.item()),
                  # 'dLossVal:{:.4f}'.format(dLoss_val.item()),
                  # 'd_zLossTrain:{:.4f}'.format(d_zLoss_train.item()),
                  # 'd_zLossVal:{:.4f}'.format(d_zLoss_val.item()),
                  #
                  # 'CFpLossTrain:{:.4f}'.format(cfPLoss_train.item()),
                  # 'CFpLossVal:{:.4f}'.format(cfPLoss_val.item()),
                  # 'CFdLossTrain:{:.4f}'.format(cfDLoss_train.item()),
                  # 'CFdLossVal:{:.4f}'.format(cfDLoss_val.item()),
                  # 'CFd_zLossTrain:{:.4f}'.format(cfD_zLoss_train.item()),
                  # 'CFd_zLossVal:{:.4f}'.format(cfD_zLoss_val.item()),

                  'iE_train:{:.4f}'.format(individual_effect_train.item()),
                  'PE_train:{:.4f}'.format(peer_effect_train.item()),
                  'TE_train:{:.4f}'.format(total_effect_train.item()),

                  '\t',
                  'iE_te:{:.4f}'.format(individual_effect_te.item()),
                  'PE_te:{:.4f}'.format(peer_effect_te.item()),
                  'TE_te:{:.4f}'.format(total_effect_te.item()),
                  #
                  # 'AiE_train:{:.4f}'.format(ate_individual_train.item()),
                  # 'APE_train:{:.4f}'.format(ate_peer_train.item()),
                  # 'ATE_train:{:.4f}'.format(ate_total_train.item()),
                  'AiE_te:{:.4f}'.format(ate_individual_te.item()),
                  'APE_te:{:.4f}'.format(ate_peer_te.item()),
                  'ATE_te:{:.4f}'.format(ate_total_te.item()),

                  )

    def train(self):
        """
               Orchestrates training workflows based on the specified model type, including:
               1.NetEsimator: Adversarial training with discriminators
               2.TargetedModel_DoubleBSpline: Pre-training + iterative dual-robust training
               3.Baseline models: Standard encoder-predictor training
               """
        print("================================Training Start================================")

        if self.args.model == "NetEsimator":
            print("******************NetEsimator******************")
            for epoch in range(self.args.epochs):
                self.train_discriminator(epoch)
                self.train_discriminator_z(epoch)
                self.train_encoder_predictor(epoch)
        elif self.args.model == "TargetedModel_DoubleBSpline" :
            print("******************" + str(self.args.model) + "******************")

            # Phase 1: Pre-training (fixed steps)
            print("******************" + "train 1 step for specified epochs" + "******************")
            for epoch in range(self.args.pre_train_step):
                # 1 step
                self.train_encoder_predictor(epoch)

            # Phase 2: Iterative training with dual-robust refinement
            print("******************" + "train iteratively for specified epochs" + "******************")
            for epoch in range(self.args.epochs):
                # iterative training step
                self.train_encoder_predictor(epoch)
                # if self.args.beta != 0:
                #     self.train_fluctuation_param(epoch)
                self.train_fluctuation_param(epoch)

        else:
            print("******************" + str(self.args.model) + "******************")
            for epoch in range(self.args.epochs):
                self.train_encoder_predictor(epoch)

    def one_step_predict(self, A, X, T, Y):
        """
                Computes model predictions and evaluation loss for given data, with model-specific
                handling for potential outcome inference.

                :param A (Tensor): Adjacency matrix (shape [N, N])
                :param X (Tensor): Node features (shape [N, D])
                :param T (Tensor): Treatment variable (shape [N])
                :param Y (Tensor): Observed outcomes (shape [N])

                Returns:
                    tuple[Tensor, Tensor, Tensor]:
                        pLoss: Prediction loss (scalar)
                        pred_outcome: Model predictions (shape [N])
                        Y: Recovered ground truth outcomes (shape [N])
                """
        self.model.eval()
        if  self.args.model == 'TargetedModel_DoubleBSpline' :
            # Specialized prediction for double B-spline model
            pred_outcome = self.model.infer_potential_outcome(A, X, T)
            pred_outcome = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome)
            Y = utils.PO_normalize_recover(self.args.normy, self.POTrain, Y)
            # print(pred_outcome.shape)
            # print(Y.shape)
            pLoss = self.loss(pred_outcome.reshape(-1), Y)
        else:
            # Generic prediction for other models
            pred_treatment, _, pred_outcome, _, _ = self.model(A, X, T)
            # Recover normalized outcomes to original scale
            pred_outcome = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome)
            Y = utils.PO_normalize_recover(self.args.normy, self.POTrain, Y)
            # Compute prediction loss
            pLoss = self.loss(pred_outcome.reshape(-1), Y)

        return pLoss, pred_outcome, Y

    def predict(self):
        """
                Execute full prediction pipeline and persist results to disk

                Performs:
                1. Factual and counterfactual outcome prediction on training/validation/testing sets
                2. Causal effect metric computation (PEHE, ATE)
                3. Result serialization to structured pickle files for reproducibility

                Output files include:
                1.Predicted outcomes (factual/counterfactual)
                2.Ground truth outcomes (original scale)
                3.Loss metrics (factual/counterfactual prediction errors)
                4.Causal effect metrics (PEHE, ATE)
                5.Experiment configuration parameters
        """
        print("================================Predicting================================")
        # Factual outcome predictions
        factualLossTrain, pred_train, YFTrainO = self.one_step_predict(self.trainA, self.trainX, self.trainT,
                                                                       self.YFTrain)
        factualLossVal, pred_val, YFValO = self.one_step_predict(self.valA, self.valX, self.valT, self.YFVal)
        factualLossTest, pred_test, YFTestO = self.one_step_predict(self.testA, self.testX, self.testT, self.YFTest)

        # Counterfactual outcome predictions
        cfLossTrain, cfPred_train, YCFTrainO = self.one_step_predict(self.trainA, self.trainX, self.cfTrainT,
                                                                     self.YCFTrain)
        cfLossVal, cfPred_val, YCFValO = self.one_step_predict(self.valA, self.valX, self.cfValT, self.YCFVal)
        cfLossTest, cfPred_test, YCFTestO = self.one_step_predict(self.testA, self.testX, self.cfTestT, self.YCFTest)

        # Compute causal effects across all sets
        individual_effect_train, peer_effect_train, total_effect_train, \
                ate_individual_train, ate_peer_train, ate_total_train \
                = self.compute_effect_pehe(self.trainA, self.trainX, self.train_t1z1, self.train_t1z0,
                                           self.train_t0z7, self.train_t0z2, self.train_t0z0)
        individual_effect_val, peer_effect_val, total_effect_val, \
            ate_individual_val, ate_peer_val, ate_total_val \
            = self.compute_effect_pehe(self.valA, self.valX, self.val_t1z1, self.val_t1z0,
                                        self.val_t0z7, self.val_t0z2, self.val_t0z0)
        individual_effect_test, peer_effect_test, total_effect_test, \
            ate_individual_test, ate_peer_test, ate_total_test \
            = self.compute_effect_pehe(self.testA, self.testX, self.test_t1z1, self.test_t1z0,
                                       self.test_t0z7, self.test_t0z2, self.test_t0z0)
        # Print key causal effect metrics for quick performance inspection
        print(
              # 'F_train:{:.4f}'.format(factualLossTrain.item()),
              # 'F_val:{:.4f}'.format(factualLossVal.item()),
              # 'F_test:{:.4f}'.format(factualLossTest.item()),
              # 'CF_train:{:.4f}'.format(cfLossTrain.item()),
              # 'CF_val:{:.4f}'.format(cfLossVal.item()),
              # 'CF_test:{:.4f}'.format(cfLossTest.item()),

              'IiE_train:{:.4f}'.format(individual_effect_train.item()),
              'IPE_train:{:.4f}'.format(peer_effect_train.item()),
              'ITE_train:{:.4f}'.format(total_effect_train.item()),
              'IiE_val:{:.4f}'.format(individual_effect_val.item()),
              'IPE_val:{:.4f}'.format(peer_effect_val.item()),
              'ITE_val:{:.4f}'.format(total_effect_val.item()),
              'IiE_test:{:.4f}'.format(individual_effect_test.item()),
              'IPE_test:{:.4f}'.format(peer_effect_test.item()),
              'ITE_test:{:.4f}'.format(total_effect_test.item()),
            'ate_ie_train{:.4f}'.format(ate_individual_train.detach().cpu().numpy()),
            'ate_pe_train{:.4f}'.format(ate_peer_train.detach().cpu().numpy()),
            'ate_te_train{:.4f}'.format(ate_total_train.detach().cpu().numpy()),
            'ate_ie_val{:.4f}'.format(ate_individual_val.detach().cpu().numpy()),
            'ate_pe_val{:.4f}'.format(ate_peer_val.detach().cpu().numpy()),
            'ate_te_val{:.4f}'.format(ate_total_val.detach().cpu().numpy()),
            'ate_ie_test{:.4f}'.format(ate_individual_test.detach().cpu().numpy()),
            'ate_pe_test{:.4f}'.format(ate_peer_test.detach().cpu().numpy()),
            'ate_te_test{:.4f}'.format(ate_total_test.detach().cpu().numpy()),
              )
        # Structured data container for all experiment outputs
        data = {
            "pred_train_factual": pred_train,
            "PO_train_factual": YFTrainO,
            "pred_val_factual": pred_val,
            "PO_val_factual": YFValO,
            "pred_test_factual": pred_test,
            "PO_test_factual": YFTestO,

            "pred_train_cf": cfPred_train,
            "PO_train_cf": YCFTrainO,
            "pred_val_cf": cfPred_val,
            "PO_val_cf": YCFValO,
            "pred_test_cf": cfPred_test,
            "PO_test_cf": YCFTestO,

            "factualLossTrain": factualLossTrain.detach().cpu().numpy(),
            "factualLossVal": factualLossVal.detach().cpu().numpy(),
            "factualLossTest": factualLossTest.detach().cpu().numpy(),
            "cfLossTrain": cfLossTrain.detach().cpu().numpy(),
            "cfLossVal": cfLossVal.detach().cpu().numpy(),
            "cfLossTest": cfLossTest.detach().cpu().numpy(),
            "individual_effect_train": individual_effect_train.detach().cpu().numpy(),
            "peer_effect_train": peer_effect_train.detach().cpu().numpy(),
            "total_effect_train": total_effect_train.detach().cpu().numpy(),
            "individual_effect_val": individual_effect_val.detach().cpu().numpy(),
            "peer_effect_val": peer_effect_val.detach().cpu().numpy(),
            "total_effect_val": total_effect_val.detach().cpu().numpy(),
            "individual_effect_test": individual_effect_test.detach().cpu().numpy(),
            "peer_effect_test": peer_effect_test.detach().cpu().numpy(),
            "total_effect_test": total_effect_test.detach().cpu().numpy(),

            "ate_individual_train": ate_individual_train.detach().cpu().numpy(),
            "ate_peer_train": ate_peer_train.detach().cpu().numpy(),
            "ate_total_train": ate_total_train.detach().cpu().numpy(),
            "ate_individual_val": ate_individual_val.detach().cpu().numpy(),
            "ate_peer_val":ate_peer_val.detach().cpu().numpy(),
            "ate_total_val":ate_total_val.detach().cpu().numpy(),
            "ate_individual_test":ate_individual_test.detach().cpu().numpy(),
            "ate_peer_test":ate_peer_test.detach().cpu().numpy(),
            "ate_total_test":ate_total_test.detach().cpu().numpy(),

            "args": str(self.args),
                }

        # Generate unique file path based on experiment parameters for reproducibility
        if self.args.model == "NetEsimator":
            # Path for NetEsimator (adversarial model)
            print("================================Save prediction...================================")
            file = "./results/" + self.args.dataset + "/perf/" + self.args.dataset + "_prediction_expID_" + str(
                    self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
                    self.args.gamma) + "_flipRate_" + str(self.args.flipRate) + ".pkl"

        elif self.args.model == 'TargetedModel_DoubleBSpline':
            # Path for TargetedModel_DoubleBSpline (dual-robust model)
            file = "./results/baselines/" + self.args.model + "/" + self.args.dataset + "/perf/prediction_expID_" + \
                       str(self.args.expID) + \
                       "_alpha_" + str(self.args.alpha) + "_gamma_" + str(self.args.gamma) + "_beta_" + '{:.4f}'.format(self.args.beta)  + \
                       "_flipRate_" + str(self.args.flipRate) + "_numGrid_" + str(self.args.num_grid) + \
                       '_kn_' + str(self.args.tr_knots) + "_lr_1_" + str(self.args.lr_1step) + "_lr_2_" + str(self.args.lr_2step) + ".pkl"

        else:
            # Path for baseline models
            print("================================Save Bseline prediction...================================")
            file = "./results/baselines/" + self.args.model + "/" + self.args.dataset + "/perf/prediction_expID_" + str(
                self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
                self.args.gamma) + "_flipRate_" + str(self.args.flipRate)  + ".pkl"


        # Ensure output directory exists and serialize data
        dir_path = os.path.dirname(file)  # 提取目录路径
        # Create directories recursively if missing
        os.makedirs(dir_path, exist_ok=True)

        with open(file, "wb") as f:
            # print('not save')
            pkl.dump(data, f)
        print("================================Save prediction done!================================")

    def save_curve(self):
        """
               Save training curve data for discriminator losses

               Persists loss trajectories of treatment discriminator and peer effect discriminator
               (training/validation sets) to disk. Used for training dynamics analysis and model debugging.

               Output file contains:
               1.Treatment discriminator losses (training/validation)
               2.Peer effect discriminator losses (training/validation)
               3.Auxiliary losses (vs. random targets)
               """
        print("================================Save curve...================================")
        # Structured data container for loss curves
        data = {"dissTrain": self.dissTrain,
                "dissVal": self.dissVal,
                "dissTrainHalf": self.dissTrainHalf,
                "dissValHalf": self.dissValHalf,
                "diss_zTrain": self.diss_zTrain,
                "diss_zVal": self.diss_zVal,
                "diss_zTrainHalf": self.diss_zTrainHalf,
                "diss_zValHalf": self.diss_zValHalf}

        with open("../results/" + str(self.args.dataset) + "/curve/" + "curve_expID_" + str(
                self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
            self.args.gamma) + "_flipRate_" + str(self.args.flipRate) + ".pkl", "wb") as f:
            pkl.dump(data, f)
        print("================================Save curve done!================================")

    def save_embedding(self):
        """
                Persists low-dimensional representations of nodes (embeddings) along with corresponding
                treatment variables (T) and peer effect variables (Z). Used for downstream analysis of
                representation learning quality and causal mechanism exploration.

                Output file contains:
                1.Node embeddings (training/testing sets)
                2.Treatment variables (T) for training/testing sets
                3.Peer effect variables (Z) for training/testing sets
                """
        print("================================Save embedding...================================")
        # Get embeddings from model (training/testing sets)
        _, _, _, embedsTrain, _ = self.model(self.trainA, self.trainX, self.trainT)
        _, _, _, embedsTest, _ = self.model(self.testA, self.testX, self.testT)
        # Structured data container for embeddings and metadata
        data = {"embedsTrain": embedsTrain.cpu().detach().numpy(), "embedsTest": embedsTest.cpu().detach().numpy(),
                "trainT": self.trainT.cpu().detach().numpy(), "testT": self.testT.cpu().detach().numpy(),
                "trainZ": self.trainZ.cpu().detach().numpy(), "testZ": self.testZ.cpu().detach().numpy()}
        with open("../results/" + str(self.args.dataset) + "/embedding/" + "embeddings_expID_" + str(
                self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
            self.args.gamma) + "_flipRate_" + str(self.args.flipRate) + ".pkl", "wb") as f:
            pkl.dump(data, f)
        print("================================Save embedding done!================================")
