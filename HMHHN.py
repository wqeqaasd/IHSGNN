import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
import torch
import math
import time
import dgl.function as fn
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from . import BaseModel, register_model
from ..layers.HeteroLinear import HeteroMLPLayer, HeteroLinearLayer
from ..layers.GeneralGNNLayer import GeneralLayer,Linear


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

@register_model('HMHHN')
class HMHHN(BaseModel):
    r"""
    OUR WORKS
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, 'GCN', 5,project_dim=args.dim_size['project'],
                 emd_dim=args.dim_size['emd'])

    def __init__(self, g, gnn_model,gnn_layer,project_dim, emd_dim):
        super(HMHHN, self).__init__()
        self.gnn_model = gnn_model
        self.norm_emb = True
        self.gnn_layer = gnn_layer
        # dimension of transform: after projecting, after aggregation, after CE_encoder
        self.project_dim = project_dim
        self.emd_dim = emd_dim
        self.device = "cuda"
        self.dblp = False
        # * ================== encoder config==================
        linear_dict1 = {}
        linear_dict2 = {}
        linear_dict3 = {}
        linear_dict4 = {}
        linear_dict5 = {}
        linear_dict6 = {}

        gain = torch.nn.init.calculate_gain('tanh')
        for ntype in g.ntypes:
            '''
              if g.nodes[ntype].data != dict():
                 in_dim = g.nodes[ntype].data['h'].shape[1]
                             if ntype == 'movie':
                in_dim = g.nodes[ntype].data['h'].shape[1]
              '''

            if ntype == 'author':
                in_dim = g.nodes[ntype].data['h'].shape[1]
            else:

                dim = g.num_nodes(ntype)
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                g.nodes[ntype].data['h'] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim]),
                                                                   device=self.device)
                #g.nodes[ntype].data['h'] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(self.device)
                #g.nodes[ntype].data['h'] = torch.sparse_coo_tensor(g.num_nodes(ntype)).to(self.device)
                #torch.nn.init.xavier_uniform_(g.nodes[ntype].data['h'], gain=gain)
                in_dim = g.num_nodes(ntype)

            '''
            g.nodes[ntype].data['h'] = torch.eye(g.num_nodes(ntype)).to(self.device)
            #g.nodes[ntype].data['h'] = torch.zeros(g.num_nodes(ntype),g.num_nodes(ntype)).to(self.device)
            #g.nodes[ntype].data['h'] = torch.cat((g.nodes[ntype].data['h'], torch.zeros(g.num_nodes(ntype),2000-g.num_nodes(ntype)).to(self.device)),dim=1).to(self.device)#acm   1902 dblp 20 imdb 7971
            #torch.nn.init.xavier_uniform_(g.nodes[ntype].data['h'],gain=gain)
            in_dim = g.num_nodes(ntype)
            #in_dim = 2000
            '''


            linear_dict1[ntype] = (in_dim, self.project_dim)

            if self.gnn_layer==2:
                linear_dict2[ntype] = (2 * self.emd_dim, self.emd_dim)
            else:
                linear_dict2[ntype] = ((self.gnn_layer - 1) * self.emd_dim, self.emd_dim)
            linear_dict3[ntype] = (self.gnn_layer * self.emd_dim,self.emd_dim, 4)#len(g.ntypes) hgbn-imdb  5 hgbn-acm 3  dblp 4
            linear_dict4[ntype] = (self.emd_dim, 4)
            linear_dict5[ntype] = ((self.gnn_layer-1) * self.emd_dim,self.emd_dim, 4)

        self.latent_dim = 4
        # * ================== Project feature Layer==================
        self.feature_proj = HeteroLinearLayer(linear_dict1, has_l2norm=False, has_bn=False)

        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4)
        self.layers = torch.nn.ModuleList()
        self.projlayers = torch.nn.ModuleList()
        self.projlayers.append(
            Linear(self.project_dim, self.emd_dim-4, act=torch.nn.Tanh(), dropout=0.5, has_l2norm=False, has_bn=True))
        self.projlayers.append(
            Linear(self.project_dim, self.emd_dim-6, act=torch.nn.Tanh(), dropout=0.5, has_l2norm=False, has_bn=True))
        self.projlayers.append(
            Linear(self.project_dim, self.emd_dim-8, act=torch.nn.Tanh(), dropout=0.5, has_l2norm=False, has_bn=True))
        self.projlayers.append(
            Linear(self.project_dim, self.emd_dim-10, act=torch.nn.Tanh(), dropout=0.5, has_l2norm=False, has_bn=True))
        self.projlayers.append(
            Linear(self.project_dim, self.emd_dim-12, act=torch.nn.Tanh(), dropout=0.5, has_l2norm=False, has_bn=True))

        self.layer = GeneralLayer('sageconv', self.emd_dim, self.emd_dim, dropout=0.5, has_bn=False, act=torch.nn.Tanh(),has_l2norm=False)
        self.PPR = dgl.transforms.PPR(alpha=0.15)
        if self.gnn_model == "GCN":
            if self.gnn_layer == 1:
                self.layers.append(GeneralLayer('sageconv', self.project_dim, self.emd_dim,num_heads=8,dropout=0.5,negative_slope=0.05, has_bn=False,act=torch.nn.Tanh(),has_l2norm=False))#len(g.ntypes) hgbn-imdb  5 hgbn-acm 3  dblp 4
            else:
                if self.gnn_layer == 2:
                    self.layers.append(GeneralLayer('gatconv', self.project_dim, self.emd_dim, num_heads=8,dropout=0.5,negative_slope=0.05, has_bn=False,act=torch.nn.Tanh(),has_l2norm=False))

                if self.gnn_layer >= 3:
                    self.layers.append(GeneralLayer('gatconv', self.project_dim, self.emd_dim,num_heads=4,dropout=0.5,negative_slope=0.05, has_bn=False,act=torch.nn.Tanh(),has_l2norm=False))

                    for t in range(2, self.gnn_layer-1):
                        self.layers.append(GeneralLayer('gatconv', self.emd_dim, self.emd_dim, num_heads=4,dropout=0.5,negative_slope=0.05,
                            has_bn=False, act=torch.nn.Tanh(),has_l2norm=False))

                    self.layers.append(GeneralLayer('gatconv', self.emd_dim, self.emd_dim,num_heads=4,dropout=0.5,negative_slope=0.05,has_bn=False, act=torch.nn.Tanh(),has_l2norm=False))

                    '''self.add_module('gnn1', GraphConv(self.project_dim, self.emd_dim, norm="none", activation=F.relu))
            for t in range(2,self.gnn_layer):
                self.add_module('gnn' + t, GraphConv(self.emd_dim, self.emd_dim, norm="none", activation=F.relu))
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, norm="none", activation=None)'''
        elif self.gnn_model == "GAT":
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, activation=F.relu)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, activation=None)

        self.projMLP = HeteroMLPLayer(linear_dict2, act=F.relu, dropout=0.5,
                                      has_l2norm=False, has_bn=True, final_act=False)
        self.typeMLP = HeteroMLPLayer(linear_dict3, act=torch.nn.Tanh(), dropout=0.5,
                                      has_l2norm=True, has_bn=False, final_act=True)
        self.typeMLP1 = HeteroMLPLayer(linear_dict4, act=torch.nn.Tanh(), dropout=0.5,
                                      has_l2norm=True, has_bn=False, final_act=False)

        self.encoder = Linear(self.emd_dim, int(self.emd_dim/8), act=torch.nn.Tanh(), dropout=0.0, has_l2norm=False)
        self.decoder = Linear(int(self.emd_dim/8), self.emd_dim, act=torch.nn.Tanh(), dropout=0.0, has_l2norm=False)
        self.gate = Linear(self.emd_dim, 1, act=torch.nn.Tanh(), dropout=0.0, has_l2norm=False,has_bn=False)
    def ordered_label_propagation(self, g, seeds):
        labels = g.ndata['label'][seeds]
        for layer in range(g.number_of_layers()):
            neighbors = list(g.successors(seeds))
            if neighbors:
                for neighbor in neighbors:
                    g.ndata['label'][neighbor] = F.cat([g.ndata['label'][neighbor], labels])

    def combine_message(self, edges):
        k=edges.src['h']
        #return {'msg': torch.cat([torch.zeros((edges.src['h'].size(0),2), dtype=torch.float32).to(self.device),edges.src['h']], dim=-1)}
        return {'msg': edges.src['h']}

    def update_label(self, nodes):
        k=torch.cat([nodes.data['label'], nodes.data['h']],dim=-1)
        return {'h': torch.cat([nodes.data['label'], nodes.data['h']],dim=-1)}

    def reduce_func(self, nodes):
        k = torch.prod(nodes.mailbox['msg'], dim=1)
        #k = torch.mean(nodes.mailbox['msg'], dim=1)
        #k = F.sigmoid(k)
        return {'h': k}#/(torch.norm(k, p=2)+1e-8)}


    def forward(self,hg,blocks, h_dict,output_nodes=None,category=None,apply_aggregation=False,test=False):

        with hg.local_scope():
            # * =============== Encode heterogeneous feature ================
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata=['h'])
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            h = g.ndata['h']

            if not test:
                if apply_aggregation:
                    out_h = h#[output_nodes]


                    h_label = self.h2label(g.nodes(), hg.ndata['h'])
                    if self.dblp:
                        pass


                    else:
                        g_adj0=self.khop_adj(g, 1)#.to_sparse()

                        g_adj0 = sp.triu(g_adj0)
                        g_adj0 = dgl.from_scipy(g_adj0).to(self.device)

                    g_khop1 = dgl.khop_out_subgraph(g_adj0, output_nodes, self.gnn_layer - 1, relabel_nodes=False,
                                                   store_ids=False).to(self.device)
                    g_khop1 = dgl.remove_self_loop(g_khop1).to(self.device)

                    g_khop1.ndata['label'] = h_label
                    g_khop1.ndata['h'] = h_label

                    src_ids = torch.tensor([2])
                    dst_ids = torch.tensor([1])

                    for k, (layer, projlayer) in enumerate(zip(self.layers, self.projlayers)):

                        g_khop = dgl.khop_graph(g_adj0, k+1)

                        g_khop = dgl.out_subgraph(g_khop, output_nodes, relabel_nodes=False, store_ids=False).to(self.device)
                        degrees = g_khop.in_degrees() 
                        non_zero_degree_nodes = torch.nonzero(degrees).squeeze()
                        g_khop = dgl.add_reverse_edges(g_khop, readonly=None, copy_ndata=False, copy_edata=False,
                                                       ignore_bipartite=False, exclude_self=True)


                        g_khop = dgl.remove_self_loop(g_khop)

                        g_khop = dgl.add_edges(g_khop,non_zero_degree_nodes, non_zero_degree_nodes, data=None, etype=None)

                        g_khop1.update_all(message_func=self.combine_message,
                                           reduce_func=self.reduce_func, apply_node_func=self.update_label)

                        if k==0:
                            hprj = projlayer(h)
                        else:
                            hprj = projlayer(h)

                        h1 = torch.cat((hprj, g_khop1.ndata['h']), dim=1)
                        hl = layer(g_khop, h1)

                        g_khop = dgl.graph((src_ids, dst_ids))
                        if k==0:
                            out_h = hl
                            out_h1 = hl
                        else:
                            out_h = torch.cat((hl, out_h), dim=1)
                            #out_h1 = hl+out_h1
                    #out_h1 = out_h1/(self.gnn_layer-1)
                    #h_khop = out_h[output_nodes]#.view(out_h[output_nodes].size(0), self.gnn_layer - 1, self.emd_dim)#[output_nodes]
                    #out_h1 = self.h2dict(out_h1, hg.ndata['h'])

                    out_h = self.h2dict(out_h, hg.ndata['h'])

                    return out_h#h_khop#,h_khop
                else:
                    h = h[blocks[0].srcdata["_ID"]]
                    for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                        h = layer(block, h)

                    out_h = {category: h}

                    out_h = self.typeMLP(out_h)

                    return out_h


            else:
                if apply_aggregation:
                    out_h = h
                    g_khop = blocks

                    for k, layer in enumerate(self.layers):
                        g_adj= dgl.khop_adj(g_khop, k+1)
                        g_adj = torch.where(g_adj != 0, torch.tensor(1.0), g_adj)
                        g_adj = sp.coo_matrix(g_adj.numpy())

                        g_khop = dgl.from_scipy(g_adj).to(self.device)

                        hl = layer(g_khop, h)
                        out_h = torch.cat((hl, out_h[g_khop.nodes()]), dim=1)
                        out_h = torch.cat((hl, out_h[g_khop.nodes()]), dim=1)
                    out_h = self.projMLP(out_h, hg.ndata['h'])
                    out_h = self.h2dict(out_h, hg.ndata['h'])
                    out_h = self.typeMLP(out_h)
                else:

                    for l, layer in enumerate(self.layers):
                        #if l==1:
                        #    break
                        h = layer(g, h)#+h
                        if l==0:
                            out_h1 = h
                        else:
                            out_h1 = torch.cat((h, out_h1), dim=1)

                    out_h = self.h2dict(h, hg.ndata['h'])
                    #out_h = self.typeMLP1(out_h)
                    out_h1 = self.h2dict(out_h1, hg.ndata['h'])
                    #h = self.classfication(h)
                return out_h,out_h1

    def forwaod(self, hg, h,apply_aggregation=False):
        with hg.local_scope():
            # * =============== Encode heterogeneous feature ================
            h_dict = self.feature_proj(h)

            emd = self.h2dict(h, h_dict)
            #hg.ndata['h'] = emd
            print(h_dict.shape)
        return emd, h
    def h2dict(self, h, hdict):
        pre = 0
        for i, value in hdict.items():
            hdict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return hdict

    def adddict(self, h, hkhop):
        pre = 0
        for i, value in h.items():
            h[i] = (h[i]+hkhop[i])/2
            #pre += value.shape[0]
        return h
    def concatdict(self, h, hkhop,k1,k2):
        #pre = 0
        #h = k2*h
        #hkhop = k1*hkhop
        for i, value in h.items():
            h[i] = torch.concat((k2*h[i],k1*hkhop[i]),dim=1)
            #pre += value.shape[0]
        return h

    def h2label(self, h, hdict):
        pre = 0
        k = 0
        #t = torch.LongTensor(k)
        for i, value in hdict.items():
            t = torch.tensor(k,dtype=torch.int64).repeat(1,value.shape[0]).to(self.device)
            if k==0:
                h_label = torch.cat((h[pre:value.shape[0]+pre].unsqueeze(0),t),dim=0)
            else:
                h1= torch.cat((h[pre:value.shape[0] + pre].unsqueeze(0), t), dim=0)
                h_label = torch.cat((h_label, h1), dim=1)
            pre += value.shape[0]
            k = k + 1
        h_label = h_label.to(dtype=torch.float32)
        h_label[0] = torch.pi * (h_label[0]-h.size(0)/2) / h_label.size(1)
        h_label[1] = torch.pi * h_label[1] / (k*2)
        h_label[0] = self.positional_encoding_1d(h_label[0].size(0))#torch.sin(h_label[0])
        h_label[1] = torch.cos(h_label[1])
        return torch.transpose(h_label, 0, 1)

    def positional_encoding_1d(self, node_num):
        position = torch.arange(1, node_num + 1).float()
        div_term = torch.exp(torch.arange(0, node_num).float() * -(math.log(10000.0) / node_num))
        encoding = torch.zeros(1, node_num)
        encoding[0, :] = torch.sin(position * div_term)
        return encoding
    def top_k_filter(self, adj, sim_matrix, k=25):
        # Assuming adj is a sparse matrix in CSR format
        top_k_indices = np.argpartition(sim_matrix.data, -k)[-k:]

        filtered_adj = sp.coo_matrix(
            (adj.data[top_k_indices], (adj.row[top_k_indices], adj.col[top_k_indices])),
            shape=adj.shape)
        return filtered_adj

    def khop_adj(self,g, k):
        """Return the matrix of :math:`A^k` where :math:`A` is the adjacency matrix of the graph
        :math:`g`.

        The returned matrix is a 32-bit float dense matrix on CPU. The graph must be homogeneous.

        Examples
        --------
        >>> import dgl
        >>> g = dgl.graph(([0,1,2,3,4,0,1,2,3,4], [0,1,2,3,4,1,2,3,4,0]))
        >>> dgl.khop_adj(g, 1)
        tensor([[1., 1., 0., 0., 0.],
                [0., 1., 1., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 1., 1.],
                [1., 0., 0., 0., 1.]])
        >>> dgl.khop_adj(g, 3)
        tensor([[1., 3., 3., 1., 0.],
                [0., 1., 3., 3., 1.],
                [1., 0., 1., 3., 3.],
                [3., 1., 0., 1., 3.],
                [3., 3., 1., 0., 1.]])
        """
        assert g.is_homogeneous, "only homogeneous graph is supported"
        adj_k = (
                g.adj_external(transpose=False, scipy_fmt=g.formats()["created"][0])
                ** k
        )
        return adj_k#F.tensor(adj_k.todense().astype(np.float32))


