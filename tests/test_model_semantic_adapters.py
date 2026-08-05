import unittest
import torch
import torch.nn as nn

from spn_quant.adapters import detect_model_name, install_model_semantic_adapter


class BasicBlock(nn.Module):
    def __init__(self, c=4):
        super().__init__(); self.conv1=nn.Conv2d(c,c,1); self.bn1=nn.Identity(); self.relu=nn.ReLU(); self.conv2=nn.Conv2d(c,c,1); self.bn2=nn.Identity(); self.downsample=None
    def forward(self,x): return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))+x)
class Gudi_UpProj_Block(BasicBlock):
    def __init__(self,c=4):
        super().__init__(c); self.sc_conv1=nn.Conv2d(c,c,1); self.sc_bn1=nn.Identity()
    def _up_pooling(self,x,scale): return x
    def forward(self,x):
        y=self._up_pooling(x,2); out=self.relu(self.bn1(self.conv1(y))); out=self.bn2(self.conv2(out)); return self.relu(out+self.sc_bn1(self.sc_conv1(y)))
class Gudi_UpProj_Block_Cat(nn.Module):
    def __init__(self,c=4):
        super().__init__(); self.conv1=nn.Conv2d(c,c,1); self.bn1=nn.Identity(); self.conv1_1=nn.Conv2d(c*2,c,1); self.bn1_1=nn.Identity(); self.conv2=nn.Conv2d(c,c,1); self.bn2=nn.Identity(); self.sc_conv1=nn.Conv2d(c,c,1); self.sc_bn1=nn.Identity(); self.relu=nn.ReLU()
    def _up_pooling(self,x,scale): return x
    def forward(self,x,side):
        y=x; out=self.relu(self.bn1(self.conv1(y))); out=torch.cat((out,side),1); out=self.relu(self.bn1_1(self.conv1_1(out))); out=self.bn2(self.conv2(out)); return self.relu(out+self.sc_bn1(self.sc_conv1(y)))
class Head(nn.Module):
    def __init__(self,cin=4,cout=1): super().__init__(); self.conv1=nn.Conv2d(cin,cout,1)
    def forward(self,x): return self.conv1(x)
class CProp(nn.Module):
    def affinity_normalization(self,g): return g, g.sum(1,keepdim=True)
    def forward(self,g,d,s=None): self.affinity_normalization(g); return torch.relu(d)
class CModel(nn.Module):
    def __init__(self):
        super().__init__(); self.conv1_1=nn.Conv2d(4,4,1); self.layer1=nn.Sequential(BasicBlock()); self.conv2=nn.Conv2d(4,4,1); self.gud_up_proj_layer1=Gudi_UpProj_Block(); self.gud_up_proj_layer2=Gudi_UpProj_Block_Cat(); self.gud_up_proj_layer3=Gudi_UpProj_Block_Cat(); self.gud_up_proj_layer4=Gudi_UpProj_Block_Cat(); self.gud_up_proj_layer5=Head(4,1); self.gud_up_proj_layer6=Head(4,4); self.post_process_layer=CProp()
    def forward(self,x):
        s=x[:,3:4]; a=self.conv2(self.layer1(self.conv1_1(x))); b=self.gud_up_proj_layer1(a); b=self.gud_up_proj_layer2(b,a); b=self.gud_up_proj_layer3(b,a); b=self.gud_up_proj_layer4(b,a); d=self.gud_up_proj_layer5(b); g=self.gud_up_proj_layer6(b); return self.post_process_layer(g,d,s)

class CatBase(nn.Module):
    def _concat(self,a,b,dim=1): return torch.cat((a,b),dim)

class SELayer(nn.Module):
    def __init__(self, c=4):
        super().__init__(); self.fc=nn.Sequential(nn.Linear(c,1),nn.ReLU(),nn.Linear(1,c),nn.Sigmoid())
    def forward(self,x):
        b,c,_,_=x.shape; gate=self.fc(x.mean((2,3))).view(b,c,1,1); return x*gate
class SEBlock(nn.Module):
    def __init__(self,c=4): super().__init__(); self.conv=nn.Conv2d(c,c,1); self.se=SELayer(c)
    def forward(self,x): return self.se(self.conv(x))

class DBase(CatBase):
    def __init__(self):
        super().__init__(); self.conv1_rgb=nn.Conv2d(3,2,1); self.conv1_dep=nn.Conv2d(1,2,1); self.conv2=SEBlock(4); self.conv6=nn.Conv2d(4,4,1); self.dec2=nn.Conv2d(8,4,1); self.gd_dec1_=nn.Conv2d(8,4,1); self.gd_dec0_dyspn_1_1=nn.Conv2d(8,3,1)
    def forward(self,rgb,dep):
        a=self.conv1_rgb(rgb); b=self.conv1_dep(dep); x=torch.cat((a,b),1); x=self.conv2(x); x=self.conv6(x); x=self.dec2(self._concat(x,x)); x=self._concat(x,x); x=self._concat(x,x); x=self._concat(x,x); x=self.gd_dec1_(x[:, :8]); return self.gd_dec0_dyspn_1_1(self._concat(x,x))
class DProp(nn.Module):
    def __init__(self): super().__init__(); self.conv_offset_aff=nn.Conv2d(1,3,1)
    def forward(self,d,g,s,c):
        z=self.conv_offset_aff(g); off=z[:,:2]; aff=torch.softmax(z[:,2:3],1); state=torch.relu(d); return {'pred':state,'list_feat':[state],'offset':off,'aff':aff}
class DModel(nn.Module):
    def __init__(self): super().__init__(); self.base=DBase(); self.dyspn_1_1=DProp()
    def forward(self,rgb,dep):
        guide=self.base(rgb,dep); return self.dyspn_1_1(guide[:,:1],guide[:,:1],dep,guide[:,1:2])

class NProp(nn.Module):
    def __init__(self): super().__init__(); self.conv_offset_aff=nn.Conv2d(2,3,1)
    def forward(self,d,g,c,s,rgb=None):
        z=self.conv_offset_aff(g); return d,[d],z[:,:2],torch.softmax(z[:,2:3],1),torch.tensor(1.)
class NModel(CatBase):
    def __init__(self):
        super().__init__(); self.conv1_rgb=nn.Conv2d(3,2,1); self.conv1_dep=nn.Conv2d(1,2,1); self.conv2=SEBlock(4); self.conv6=nn.Conv2d(4,4,1); self.dec2=nn.Conv2d(8,4,1); self.id_dec0=nn.Conv2d(8,1,1); self.gd_dec0=nn.Conv2d(8,2,1); self.cf_dec0=nn.Sequential(nn.Conv2d(8,1,1),nn.Sigmoid()); self.prop_layer=NProp()
    def forward(self,sample):
        a=self.conv1_rgb(sample['rgb']); b=self.conv1_dep(sample['dep']); x=self.conv2(torch.cat((a,b),1)); x=self.conv6(x); pairs=[]
        for _ in range(7): pairs.append(self._concat(x,x))
        y=self.dec2(pairs[0]); d=self.id_dec0(pairs[1]); g=self.gd_dec0(pairs[2]); c=self.cf_dec0(pairs[3]); out=self.prop_layer(d,g,c,sample['dep'],sample['rgb']); return {'pred':out[0],'pred_init':d,'pred_inter':out[1],'guidance':g,'offset':out[2],'aff':out[3],'confidence':c}

class Attention(nn.Module):
    def __init__(self): super().__init__(); self.q=nn.Linear(4,4)
    def forward(self,x): return self.q(x)
class ChannelGate(nn.Module):
    def __init__(self): super().__init__(); self.sigmoid=nn.Sigmoid()
    def forward(self,x): return self.sigmoid(x.mean((2,3),keepdim=True))
class Former(nn.Module):
    def __init__(self): super().__init__(); self.attn=Attention(); self.ca=ChannelGate()
    def forward(self,x):
        gate=self.ca(x); x=x*gate; b,c,h,w=x.shape; y=x.flatten(2).transpose(1,2); y=self.attn(y); return y.transpose(1,2).reshape(b,c,h,w)
class Backbone(CatBase):
    def __init__(self):
        super().__init__(); self.conv1_rgb=nn.Conv2d(3,2,1); self.conv1_dep=nn.Conv2d(1,2,1); self.conv1=nn.Conv2d(4,4,1); self.former=Former(); self.dec2=nn.Conv2d(8,4,1); self.dep_dec0=nn.Conv2d(8,1,1); self.gd_dec0=nn.Conv2d(8,2,1); self.cf_dec0=nn.Sequential(nn.Conv2d(8,1,1),nn.Sigmoid())
    def forward(self,rgb,dep):
        x=self.conv1(torch.cat((self.conv1_rgb(rgb),self.conv1_dep(dep)),1)); x=self.former(x); pairs=[]
        for _ in range(8): pairs.append(self._concat(x,x))
        y=self.dec2(pairs[0]); return self.dep_dec0(pairs[1]),self.gd_dec0(pairs[2]),self.cf_dec0(pairs[3])
class CFModel(nn.Module):
    def __init__(self): super().__init__(); self.backbone=Backbone(); self.prop_layer=NProp()
    def forward(self,sample):
        d,g,c=self.backbone(sample['rgb'],sample['dep']); d=d+sample['dep']; out=self.prop_layer(d,g,c,sample['dep'],sample['rgb']); return {'pred':out[0],'pred_init':d,'pred_inter':out[1],'guidance':g,'offset':out[2],'aff':out[3],'confidence':[c]}

class Tests(unittest.TestCase):
    def run_adapter(self,model,name,args):
        self.assertEqual(detect_model_name(model),name)
        ad=install_model_semantic_adapter(model,name,strict=True)
        ad.observe(); model(*args); ad.freeze(4); rows=ad.semantic_manifest()
        self.assertTrue(all(r['transform']=='none' for r in rows))
        self.assertTrue(all(r['observed'] for r in rows if int(r.get('meta_required',0))))
        self.assertTrue(any(r['role']=='concat_merge' and r['observed'] for r in rows))
        ad.quantize(); model(*args); ad.close(); return rows
    def test_cspn(self):
        rows=self.run_adapter(CModel(),'cspn',(torch.randn(1,4,3,3),))
        self.assertEqual(sum(r['role']=='concat_merge' and r['observed'] for r in rows),3)
    def test_dyspn(self):
        rows=self.run_adapter(DModel(),'dyspn',(torch.randn(1,3,3,3),torch.rand(1,1,3,3)))
        self.assertTrue(any(r['role']=='se_gate' and r['observed'] for r in rows))
        self.assertTrue(any(r['role']=='affinity_logits' and r['observed'] for r in rows))
    def test_nlspn(self):
        rows=self.run_adapter(NModel(),'nlspn',({'rgb':torch.randn(1,3,3,3),'dep':torch.rand(1,1,3,3)},))
        self.assertTrue(any(r['role']=='offset_logits' and r['observed'] for r in rows))
    def test_completionformer(self):
        rows=self.run_adapter(CFModel(),'completionformer',({'rgb':torch.randn(1,3,3,3),'dep':torch.rand(1,1,3,3)},))
        self.assertTrue(any(r['role']=='channel_attention_gate' and r['observed'] for r in rows))
        self.assertTrue(any(r['role']=='attention_probability' and not int(r['meta_operational']) for r in rows))
    def test_fail_closed(self):
        with self.assertRaises(RuntimeError): install_model_semantic_adapter(nn.Identity(),'cspn',strict=True)

if __name__=='__main__': unittest.main()
