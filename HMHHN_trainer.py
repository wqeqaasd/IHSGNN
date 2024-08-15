import dgl
import torch
import time
import torch as th
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import EarlyStopping
from ..layers.GeneralGNNLayer import Linear,MultiLinearLayer
from ..utils.sampler import get_epoch_samples
buffer_device = th.device("cpu")
KLloss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)


@register_flow("HMHHN_trainer")
class HMHHN_trainer(BaseFlow):


    def __init__(self, args):
        super(HMHHN_trainer, self).__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)
        self.args.category = self.task.dataset.category
        self.category = self.args.category



        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)

        self.mea = Measure_F(self.model.emd_dim, (self.model.gnn_layer-1)*self.model.emd_dim,self.model.latent_dim)
        self.optimizer = th.optim.Adam([{'params': self.model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                                       {'params': self.mea.parameters(), 'lr': args.lr,'weight_decay': args.weight_decay}])
        self.model = self.model.to(self.device)
        self.mea = self.mea.to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.pin_memory = buffer_device != self.device

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()

        self.g = dgl.to_homogeneous(self.hg)
        self.k1 = 1.0
        self.k2 = 1.0
        self.evaluate_interval = 2
        if self.args.mini_batch_flag:
            sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5, 5, 3])
            if self.train_idx is not None:
                self.train_loader = dgl.dataloading.DataLoader(
                                        self.g,self.train_idx.to(self.device), sampler,
                                        batch_size=self.args.batch_size,  # 设置batch_size
                                        device=self.device, shuffle=True
                                    )
            #torch.cat((self.train_idx,self.valid_idx),axis=-1).to(self.device)
            if self.valid_idx is not None:
                self.val_loader = dgl.dataloading.DataLoader(
                    self.g, self.valid_idx.to(self.device), sampler,
                    batch_size=self.args.batch_size,  # 设置batch_size
                    device=self.device, shuffle=True
                )
            if self.args.test_flag:
                self.test_loader = dgl.dataloading.DataLoader(
                    self.g,  self.test_idx.to(self.device), sampler,
                    batch_size=self.args.batch_size,  # 设置batch_size
                    device=self.device, shuffle=True
                )
            if self.args.prediction_flag:
                self.pred_loader = dgl.dataloading.DataLoader(
                    self.g, self.pred_idx.to(self.device), sampler,
                    batch_size=self.args.batch_size,  # 设置batch_size
                    device=self.device, shuffle=True
                )
    def preprocess(self):

        self.pos_edges = self.g.edges()
        self.category = self.task.dataset.category
        #super(HMHHN_trainer, self).preprocess()

        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        k = False
        param_groups=[{'lr':0.05 , 'weight_decay':0.0001 }]#,
        for param_group in param_groups:

            for epoch in epoch_iter:

                if epoch == 70:#acm 100 dblp 70 imdb 30
                    stopper.load_model(self.model)
                    stopper.counter = 0
                    stopper.best_loss = None
                    stopper.patience = 30

                    for param_group in self.optimizer.param_groups:
                       param_group["lr"] = 0.01
                       param_group["weight_decay"] = 0.0001
                if epoch == 105:
                    stopper.best_loss = None
                    for param_group in self.optimizer.param_groups:
                        if k == 1:
                            param_group["lr"] = 0.035
                            param_group["weight_decay"] = 0.00065
                        if k == 0:
                            k = 1
                loss = self._full_train_setp(epoch, 70)
                print('Epoch{}: Loss:{:.4f}'.format(epoch, loss))

                if epoch > 0:
                    early_stop = stopper.loss_step(loss, self.model)
                    if early_stop:
                        print('Early Stop!\tEpoch:' + str(epoch))
                        break


            stopper.load_model(self.model)
            metrics = self._test_step(modes=['valid'])

        return metrics
    def _full_train_setp(self, epoch, self_distillation_start_epoch=10):
        self.model.train()
        self.optimizer.zero_grad()
        self.hg = self.hg.to(self.device)
        h = self.hg.ndata['h']
        h_dict = self.model.feature_proj(h)


        labels = self.task.labels[self.train_idx].to(self.device)

        if epoch >= self_distillation_start_epoch:
            hgnn, hstack = self.model(self.hg, self.g, h_dict, test=True)
            h_dict = self.model.feature_proj(h)
            out_h = self.model(self.hg, self.g, h_dict, torch.cat((self.train_idx,self.valid_idx,self.test_idx),dim=-1).to(self.device), apply_aggregation=True)

            log = self.model.typeMLP(self.model.concatdict(hgnn,out_h,self.k1,self.k2))


            logits = log[self.category][self.train_idx]
            loss = 1.0 * self.some_custom_loss(logits, labels)+1.0*self.distillation_loss(hgnn[self.category][self.train_idx],logits,self.mea,labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()
            with th.no_grad():

                h_dict = self.model.feature_proj(h)
                out_h = self.model(self.hg, self.g, h_dict, self.valid_idx.to(self.device), apply_aggregation=True)
                hgnn, hstack = self.model(self.hg, self.g, h_dict, test=True)
                log = self.model.typeMLP(self.model.concatdict(hgnn,out_h,self.k1,self.k2))#self.model.concatdict(hgnn, out_h))
                labels = self.task.labels[self.valid_idx].to(self.device)
                logitsval = log[self.category][self.valid_idx]

                loss = 1.0 * self.some_custom_loss(
                    logitsval, labels)
                return loss.item()
        else:
                hgnn, hstack = self.model(self.hg, self.g, h_dict, test=True)
                log = self.model.typeMLP1(hgnn)
                logits = log[self.category][self.train_idx]
                #logits = F.log_softmax(logits, 1)
                loss = self.some_custom_loss(logits, labels)#+ self.Orthogonalization(h_khop[self.valid_idx])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(self._test_step(aggr=False,modes=["valid"]))
                #print(f"Epoch {epoch + 1}/{self.max_epoch}, Loss: {loss_all}")
                self.model.eval()
                with th.no_grad():
                    hgnn, hstack = self.model(self.hg, self.g, h_dict, test=True)
                    log = self.model.typeMLP1(hgnn)
                    logitsval = log[self.category][self.valid_idx]
                    labels = self.task.labels[self.valid_idx].to(self.device)
                    #logitsval = F.log_softmax(logitsval, 1)
                    loss = self.some_custom_loss(logitsval, labels)

                return loss.item()

    def some_custom_loss(self,logits, labels):
        k=F.cross_entropy(logits, labels)
        return F.cross_entropy(logits, labels)
    def sim_loss(self, h):
        logitskhop = h[:, self.model.emd_dim:]
        logitskhop = self.model.encoder2(logitskhop)
        logits = self.model.encoder3(h)
        sim1 = torch.sum(1 - F.cosine_similarity(logitskhop, logits))

        return 0.0001*sim1
    def self_distillation_loss(self,logits, logits_teacher, labels, alpha=0.5, temperature=1.0):
        hard_loss = F.cross_entropy(logits, labels)
        soft_loss = KLloss(F.log_softmax(logits, dim=-1),F.log_softmax(logits_teacher.detach(), dim=-1))
        total_loss = alpha*hard_loss + (1-alpha)*soft_loss
        return total_loss

    def distillation_loss(self, h,logits,mea,labels,margin=0.4):
        # 先执行分布评估，再使用KL散度评估分布差异
        logitskhop = h[:,self.model.emd_dim:]
        logitsgnn = h[:,:self.model.emd_dim]
        logitsgnn, logitskhop = mea(logitsgnn,logitskhop)

        soft_loss1 = KLloss(F.log_softmax(logitskhop, dim=-1), F.log_softmax(logits, dim=-1))
        soft_loss2 = KLloss(F.log_softmax(logitsgnn, dim=-1), F.log_softmax(logits, dim=-1))

        return 1.0*soft_loss1
    def recon_loss(self, h,margin=1.0):
        h1 = h[:,self.model.emd_dim:]
        h = h1.reshape(h1.size(0)*(self.model.gnn_layer-1),self.model.emd_dim)
        hidden= self.model.encoder(h)
        H = self.model.decoder(hidden)
        h = h.view(h1.size(0), self.model.gnn_layer-1, self.model.emd_dim)
        H = H.view(h1.size(0), self.model.gnn_layer-1, self.model.emd_dim)
        similarity_matrix = F.cosine_similarity(h.unsqueeze(2), H.unsqueeze(1), dim=3)
        similarity_matrix1 = F.cosine_similarity(h, H, dim=2)
        sum_exp_sim_over_tau = torch.sum(similarity_matrix , dim=1)
        loss = torch.sum(sum_exp_sim_over_tau - 2*similarity_matrix1)

        return loss


    def generateLabel(self,x,blocks):
        Key = [x[key].size(0) for key in x]
        for i in range(1, len(Key)):
            Key[i] = Key[i - 1] + Key[i]
        Label = [key for key in x]
        Label = torch.eye(len(Label), dtype=torch.float32)
        label = []
        for index in blocks[0].srcdata["_ID"]:
            for i in range(len(Key)):
                if index < Key[i]:
                    label.append(Label[i])
                    break
        label = torch.stack(label)
        label=label[blocks[-1].dstdata["_ID"]]
        return label

    def _mini_train_step(self, epoch, self_distillation_start_epoch=10):
        self.model.train()
        self.optimizer.zero_grad()
        self.hg = self.hg.to(self.device)
        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)

        h = self.hg.ndata['h']

        for l, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
            h_dict = self.model.feature_proj(h)
            logits = self.model(self.hg,blocks, h_dict,self.category)[self.category]
            labels = self.task.labels[output_nodes.to("cpu")]
            labels = labels.to(self.device)

            if epoch >= self_distillation_start_epoch:
                g_sample, inverse_indices = dgl.khop_in_subgraph(self.hg,
                                                                 {self.category: self.train_idx.to(self.device)},
                                                                 self.model.gnn_layer, relabel_nodes=True)
                g_sample = dgl.to_homogeneous(g_sample)
                logitsST = self.model(self.hg, g_sample, h_dict, self.category, apply_aggregation=True)
                logitsST = logitsST[self.category]
                loss = 0.5 * self.self_distillation_loss(logitsST, logits, labels, 1.0) + self.some_custom_loss(
                    logits, labels)
            else:
                 loss = self.some_custom_loss(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_all+=loss.item()

        self.model.eval()
        logits = self.model(self.hg, self.g, h_dict, test=True)
        logitsval = logits[self.category][self.valid_idx]
        labels = self.task.labels[self.valid_idx].to(self.device)
        loss = self.some_custom_loss(logitsval, labels)
        return loss.item()
    def loss_calculation(self, homo_h, neg_edges, ns_samples, ns_prediction):
        pairwise_loss = self.cal_node_pairwise_loss(homo_h, self.pos_edges, neg_edges)

        ns_label = th.cat([ns['label'] for ns in ns_samples]).type(th.float32).to(self.args.device)
        BCE_loss = th.nn.BCELoss()
        cla_loss = BCE_loss(ns_prediction, ns_label)
        loss = pairwise_loss + cla_loss * self.args.beta
        return loss

    def loss_calc(self, homo_h, neg_edges):
        pairwise_loss = self.cal_node_pairwise_loss(homo_h, self.pos_edges, neg_edges)
        return pairwise_loss

    def cal_node_pairwise_loss(self, node_emd, edge, neg_edge):
        # cross entropy loss from LINE
        # pos loss
        inner_product = self.cal_inner_product(node_emd, edge)
        pos_loss = - th.mean(F.logsigmoid(inner_product))
        # neg loss
        inner_product = self.cal_inner_product(node_emd, neg_edge)
        neg_loss = - th.mean(F.logsigmoid(-1 * inner_product))
        loss = pos_loss + neg_loss
        return loss


    def _test_step(self, modes,aggr=True, logits=None):
        self.model.eval()
        #loader_tqdm = tqdm(self.train_loader, ncols=120)
        #for l, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
        with th.no_grad():
            # elif self.args.task == 'link_prediction':
            #     metric = self.task.evaluate(logits, 'academic_lp')
            #     return metric
            h = self.hg.ndata['h']
            h_dict = self.model.feature_proj(h)
            #h_dict = self.model.input_feature()
            if aggr:
                out_h = self.model(self.hg, self.g, h_dict,self.test_idx.to(self.device), apply_aggregation=True)
                #node_emb[self.category] =(node_emb[self.category]+h_khop[self.category])/2
                node_emb, outh1 = self.model(self.hg, self.g, h_dict, test=True)
                #node_emb =self.model.encoder(node_emb)
                node_emb = self.model.typeMLP(self.model.concatdict(node_emb,out_h,self.k1,self.k2))
                #node_emb = self.model.typeMLP(out_h)#self.model.concatdict(node_emb, out_h))
            else:
                node_emb,outh1 = self.model(self.hg, self.g, h_dict,test=True)
                node_emb = self.model.typeMLP1(node_emb)
            logits = logits if logits else node_emb[self.category]
            #logits = F.log_softmax(logits, 1)
            with open("DBLP.txt", "r", encoding="utf-8") as input_file:
                lines = input_file.readlines()
                lines = [line.strip().split('\t') for line in lines]
                test_idx = [int(line[0]) for line in lines]#torch.LongTensor([int(line[0]) for line in lines])
                index_sequences = [line[3] for line in lines]
                max_index = max(max(map(int, sequence.split(","))) for sequence in index_sequences)

                # 初始化结果列表
                result_lists = []

                # 将索引序列转换为对应下标为1的序列列表
                for sequence in index_sequences:
                    indices = list(map(int, sequence.split(",")))
                    result = [1 if i in indices else 0 for i in range(max_index + 1)]
                    result_lists.append(result)
                result_lists = (torch.LongTensor(result_lists).cpu().numpy() > 0).astype(int)
                sorted_indices = np.argsort(test_idx)
                result_lists = result_lists[sorted_indices]
                logits1 = (logits[self.test_idx].cpu().numpy() > 0).astype(int)
                f1_dict = self.task.evaluator.f1_node_classification(result_lists, logits1)



            if self.args.task == 'node_classification':
                if self.args.dataset[:4] == 'HGBn':
                    masks = {}
                    for mode in modes:
                        if mode == "train":
                            masks[mode] = self.train_idx
                        elif mode == "valid":
                            masks[mode] = self.valid_idx
                        elif mode == "test":
                            masks[mode] = self.test_idx

                    metric = {key: self.task.evaluate(logits, mode=key) for key in masks}
                else:
                    metric = self.task.downstream_evaluate(logits, 'f1_lr')
            return metric, f1_dict

    def Orthogonalization(self, embeddings):

        loss = 0
        len = int(embeddings.size(1)/self.model.emd_dim)
        for i in range(embeddings.size(0)):
            k=embeddings[i,:].view(len,self.model.emd_dim)
            if torch.isnan(k.any()) or torch.isinf(k.any()):
                print("矩阵包含 NaN 或 Infinity 值")
            frobenius_norm_sq = torch.mm(k, k.t())
            frobenius_norm_sq = torch.nn.functional.normalize(frobenius_norm_sq - torch.eye(len, device=self.device),p=2)

            loss += torch.sum(frobenius_norm_sq ** 2)
        # 返回带有权重的损失
        return loss/embeddings.size(0)

    def disen_loss(self, embeddings):

        embeddings = embeddings[:, self.model.emd_dim:]
        embeddings = embeddings.view(embeddings.size(0),self.model.gnn_layer - 1, self.model.emd_dim)
        loss = 0
        for i in range(embeddings.size(0)):
            k = self.model.encoder1(embeddings[i, :, :])
            if torch.all(k == 0):
                continue
            pearson_corr = torch.corrcoef(k) - torch.eye(embeddings.size(1)).to(self.device)

            loss += torch.sum(0.5*pearson_corr)
        return 0.0001*loss
    def update_S(self, hkhop):

        with torch.no_grad():

            FF1 = hkhop - torch.mean(hkhop, 1, True)
            U, _, T = torch.svd(torch.sum(FF1, dim=1))
            S = torch.mm(U, T.t())
            S = S * (FF1.shape[0]) ** 0.5
        return S

    def loss_matching_recons(self, s, U_batch):
        l = torch.nn.MSELoss(reduction='sum')
        k=s.shape[0]

        u = U_batch.view(U_batch.size(0), 1, U_batch.size(1)).repeat(1,s.size(1),1)
        match_err = l(s, u) / (s.size(0))

        return match_err/500#/200#/900
    def loss_decode(self, rebuild,hnei,horin):
        l = torch.nn.MSELoss(reduction='sum')
        k=rebuild.shape[0]

        # Matching loss
        match_err1 = l(rebuild, hnei.detach())/k#/ (s.size(0))


        return 1.0*match_err1#+0.03*match_err2#/60000#350000
    def loss_independence(self, phi_c_list, psi_p_list):
        # Correlation
        corr = 0
        for i in range(len(phi_c_list)):
            corr += self.compute_corr(phi_c_list[i], psi_p_list[i])
        return corr

    def compute_corr(self, x1, x2):
        # Subtract the mean
        x1_mean = torch.mean(x1, 0, True)
        x1 = x1 - x1_mean
        x2_mean = torch.mean(x2, 0, True)
        x2 = x2 - x2_mean

        # Compute the cross correlation
        sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
        sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
        corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)

        return corr

class Measure_F(torch.nn.Module):
    def __init__(self, view1_dim, view2_dim, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MultiLinearLayer((view1_dim,latent_dim),act=torch.nn.Tanh(),dropout=0.5, has_l2norm=True,has_bn=False)
        self.psi = MultiLinearLayer((view2_dim,view1_dim,latent_dim),act=torch.nn.Tanh(),dropout=0.5, has_l2norm=True,has_bn=False)
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()
        self.grl2 = GradientReversalLayer()

    def forward(self,x1,x2):
        y1 = self.phi(grad_reverse(x1,1))
        y2 = self.psi(grad_reverse(x2,1.0))#dblp1
        return y1,y2

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coeff) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)