#### EDSR ver 2 at 8/27############
######################### EDSR FINAL ####################################
import math
from option import args
import torch
import torch.nn as nn

# TODO uncomment for original
# from model import common
# from model import common_new as common
# from model import common_2 as common

from model import common_final as common
# from model import common_original as common
import numpy as np

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


def make_model(args, parent=False):
    if args.original:
        return EDSR_original(args)
    else:
        return EDSR(args)


def reshape(a,b):
    return a[np.nonzero(b)].squeeze()
def prune(a, index):
    # a (1,4,2,2)
    # b (4)
    return a[:,np.nonzero(index).squeeze(1),:,:]
class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]

        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        p=args.P
        r=args.R
        t=args.T
        pr=args.PR
        pt=args.PT
        rt=args.RT
        prt=args.PRT
        # nop=args.NOP
        self.op1 = args.op1
        self.op2 = args.op2 
        self.op_last = args.op_last

        # self.head = conv(args.n_colors, int(args.op1.sum()), kernel_size)
        self.head = conv(args.n_colors, n_feats, kernel_size)

 
        skip_init =  args.op1.unsqueeze(0)
        skip_gate = torch.cat((skip_init, args.skip.squeeze()), dim=0)

        resconv2_init =  args.op1.unsqueeze(0)#torch.ones(1,args.n_feats).cuda() #?
        resconv2_gate= torch.cat((resconv2_init, args.resconv2),dim=0)
        
        resconv3_init =  args.op1.unsqueeze(0)# torch.ones(1,args.n_feats).cuda() #?
        resconv3_gate = torch.cat((resconv3_init, args.resconv3),dim=0)
 
        in_gate = skip_gate + resconv3_gate - resconv3_gate*skip_gate.detach()
        # input_gate = torch.cat([torch.ones(1,args.n_feats), in_gate.cpu()],dim=0)
        input_gate = torch.cat([torch.ones(1,args.n_feats).cuda(), in_gate],dim=0)

        # print(input_gate)
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size,i,input_gate[i],args.op1, args.op2,p[i],r[i],t[i],pr[i],pt[i],rt[i],prt[i],skip_gate[i],resconv3_gate[i],skip_gate[i+1],resconv3_gate[i+1],resconv2_gate[i+1], res_scale=args.res_scale
            ) for i in range(n_resblocks)
        ]
        self.body = nn.Sequential(*m_body)
        self.body2 = conv(n_feats, int(args.op2.sum()), kernel_size)
        
        self.tail1 = conv( int(args.op2.sum()), int(args.op_last.sum()), kernel_size)
        # self.tail1 = conv(len(args.op_last), args.n_colors*(4**int(math.log(scale, 2))), kernel_size)

        tail2 = [nn.PixelShuffle(2) for i in range(int(math.log(scale, 2)))] # only covers scale 2, 4
        self.tail2 = nn.Sequential(*tail2)


    def forward(self, x):
        # flops =0

        x = x.cuda() # 3
        x = self.sub_mean(x) # 3
        x1 = self.head(x) # 256

        # flops+=torch.prod(torch.tensor(x1.shape[1:]))*18*x.shape[1]
        x1 = (x1.clone().permute(0,2,3,1)*self.op1).permute(0,3,1,2) # 256 (51 activated)
        # x1 = (x1.clone().permute(0,2,3,1)*self.op1.cpu()).permute(0,3,1,2) # 256 (51 activated)
        
        res = self.body(x1)

        res2 =self.body2(res) # 256 -> 70
        # change to last -> 70 ? last is probably close to or same as 256 anyways
        # flops+=torch.prod(torch.tensor(res2.shape[1:]))*18*res.shape[1]
        ### TODO Fix here?
        # print(res.shape)
        # print(res2.shape)
        x1_ = x1[:,self.op2.nonzero().squeeze(1),:,:] # 70
        res2 += x1_
        '''
        op1_pruned = reshape(self.op2, self.op1) # print(self.op1.sum()) #51
        op2_pruned = reshape(self.op1, self.op2) # print(self.op2.sum()) #70
        
        x1_ = x1[:,op1_pruned.nonzero().squeeze(1),:,:] # 12
        res2_= res2[:,(1-op2_pruned).nonzero().squeeze(1),:,:]
        x1__=torch.cat([x1_,res2_],dim=1)

        res2 += x1__ 
        '''
        out = self.tail1(res2)
        # flops+=torch.prod(torch.tensor(x.shape[1:]))*18*res2.shape[1]

        out=self.tail2(out)
        out=self.add_mean(out)


        # print(flops//10**9)
    
        return out


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))



class EDSR_original(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_original, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        #act = nn.ReLU(True)
        act = nn.ReLU(False)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*m_head)

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        # m_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)
        self.body2 = conv(n_feats, n_feats, kernel_size)

        # define tail module
        # m_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)
        # ]
        self.tail1 = conv(n_feats, args.n_colors*(4**int(math.log(scale, 2))), kernel_size)
        m_tail2 = [nn.PixelShuffle(2) for i in range(int(math.log(scale, 2)))] # only covers scale 2, 4
        # used nn.PixelShuffle instead of common.PixelShuffle because we don't need FLOPs calculation here
        
        self.tail2 = nn.Sequential(*m_tail2)

    def forward(self, x):

        x=x.cuda()
        x = self.sub_mean(x)
       
        x = self.head(x) # (3, n_feats)
        res = self.body(x) # (n_feats, n_feats) *resblocks
  

        res = self.body2(res) # (n_feats, n_feats)
        res += x

        x = self.tail1(res) # (n_feats, 3*16)
        x = self.tail2(x)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
