"""Read-only scientific visualization of cached MoGe outputs (CPU only)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

root=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1')
out=root/'audit_660_moge_railings_visual'
out.mkdir(exist_ok=True)
data=np.load(root/'StMarysChurch_moge3_vitl_exact_k/views/seq2__frame00103.npz')
rgb=np.asarray(Image.open('/mnt/pool/sqy/G4Splat_runs/cambridge_outdoor_mainline_v1/prepared/StMarysChurch/dataset_qc_tree_v5/train_all/images/seq2__frame00103.png').convert('RGB'))
depth=data['depth_m']; valid=data['valid_mask'].astype(bool)
assert rgb.shape[:2]==depth.shape
near,far=np.percentile(depth[valid],[2,98])
fig,axs=plt.subplots(2,1,figsize=(16,18))
axs[0].imshow(rgb);axs[0].set_title('660 reference RGB (1920 x 1080)')
im=axs[1].imshow(np.where(valid,depth,np.nan),cmap='turbo_r',vmin=near,vmax=far)
axs[1].set_title('Raw MoGe-3 camera-z depth; predicted metres, NOT calibrated ground truth')
fig.colorbar(im,ax=axs[1],fraction=.025)
for ax in axs:ax.axis('off')
fig.tight_layout();fig.savefig(out/'full.png',dpi=120);plt.close(fig)
y=slice(855,1065);x=slice(60,1170)
d=depth[y,x];v=valid[y,x];lo,hi=np.percentile(d[v],[2,98])
reduced=cv2.resize(np.where(valid,depth,0),(640,360),interpolation=cv2.INTER_AREA)/np.maximum(cv2.resize(valid.astype(np.float32),(640,360),interpolation=cv2.INTER_AREA),1e-8)
up=cv2.resize(reduced,(1920,1080),interpolation=cv2.INTER_NEAREST)
fig,axs=plt.subplots(5,1,figsize=(18,18))
axs[0].imshow(rgb[y,x]);axs[0].set_title('Reference RGB: railing ROI, native resolution crop')
for ax,arr,title in [(axs[1],d,'Raw MoGe-3 depth'),(axs[2],up[y,x],'Valid-area mean 640 x 360 depth, nearest upsample for display')]:
    im=ax.imshow(arr,cmap='turbo_r',vmin=lo,vmax=hi);ax.set_title(title+f' (shared range {lo:.2f} to {hi:.2f} predicted m)')
for ax,key,title in [(axs[3],'normal_direct_camera','Native model normal head'),(axs[4],'normal_depth_exact_k_camera','Postprocessed normal from depth + exact K')]:
    ax.imshow(np.clip(data[key][y,x]*.5+.5,0,1));ax.set_title(title+'; RGB = (camera normal + 1) / 2')
for ax in axs:ax.axis('off')
fig.tight_layout();fig.savefig(out/'railings.png',dpi=120);plt.close(fig)
print(out)
