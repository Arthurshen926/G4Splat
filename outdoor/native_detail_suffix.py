"""Independent low-opacity 2D proposals appended to a frozen surface model."""
import torch


class NativeDetailSuffix(torch.nn.Module):
    def __init__(self, base, xyz, normal, rgb, sigma, *, train_base_geometry=False, refine_proposals=False):
        super().__init__()
        self.base=base
        self.train_base_geometry=bool(train_base_geometry)
        self.start=len(base.get_xyz)
        self.active_sh_degree=base.active_sh_degree
        self.max_sh_degree=base.max_sh_degree
        self.use_mip_filter=False
        if xyz.ndim!=2 or xyz.shape[1]!=3 or normal.shape!=xyz.shape or rgb.shape!=xyz.shape or sigma.shape!=(len(xyz),):
            raise ValueError('Aligned surface proposal rows required')
        if len(xyz)==0 or not all(torch.isfinite(v).all() for v in (xyz,normal,rgb,sigma)) or (sigma<=0).any():
            raise ValueError('Finite nonempty proposals with positive scales required')
        normal=torch.nn.functional.normalize(normal,dim=-1)
        if (normal.norm(dim=-1)<.5).any():raise ValueError('Surface proposals require usable normals')
        # Quaternion rotates local z onto the measured world normal.
        quat=torch.stack((1+normal[:,2],-normal[:,1],normal[:,0],torch.zeros_like(normal[:,0])),dim=-1)
        opposite=normal[:,2]<-.999999
        quat[opposite]=quat.new_tensor([0.,1.,0.,0.])
        self.register_buffer('xyz',xyz.clone())
        self.register_buffer('rotation',torch.nn.functional.normalize(quat,dim=-1))
        self.register_buffer('scale',sigma[:,None].expand(-1,2).clone())
        self.dc=torch.nn.Parameter(((rgb-.5)/.28209479177387814)[:,None].clone())
        self.logit=torch.nn.Parameter(xyz.new_full((len(xyz),1),-4.59511985)) # .01
        self.register_buffer('rest',xyz.new_zeros((len(xyz),base.get_features.shape[1]-1,3)))
        self.refine_proposals=bool(refine_proposals)
        self.position_delta=torch.nn.Parameter(torch.zeros_like(xyz),requires_grad=refine_proposals)
        self.scale_delta=torch.nn.Parameter(torch.zeros_like(self.scale),requires_grad=refine_proposals)

    @property
    def proposal_xyz(self):return self.xyz+self.position_delta if self.refine_proposals else self.xyz

    def geometry_groups(self):
        return [{'params':[self.position_delta],'lr':.001},{'params':[self.scale_delta],'lr':.001}] if self.refine_proposals else []

    @property
    def get_xyz(self):return torch.cat((self.base.get_xyz if self.train_base_geometry else self.base.get_xyz.detach(),self.proposal_xyz),0)
    @property
    def get_scaling(self):return torch.cat((self.base.get_scaling if self.train_base_geometry else self.base.get_scaling.detach(),self.scale*torch.exp(.69314718056*torch.tanh(self.scale_delta)) if self.refine_proposals else self.scale),0)
    @property
    def get_rotation(self):return torch.cat((self.base.get_rotation if self.train_base_geometry else self.base.get_rotation.detach(),self.rotation),0)
    @property
    def get_opacity(self):return torch.cat((self.base.get_opacity.detach(),self.logit.sigmoid()),0)
    @property
    def get_features(self):return torch.cat((self.base.get_features.detach(),torch.cat((self.dc,self.rest),1)),0)
