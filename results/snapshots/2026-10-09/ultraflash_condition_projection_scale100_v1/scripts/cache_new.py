import csv,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; PS=ROOT.parent; sys.path.insert(0,str(PS)); sys.path.insert(0,str(PS/'latent_sr_v2_multivideo')); sys.path.insert(0,'/Huang_group/zyyz/home_dir/.conda/envs/pssr-feature-prior-runtime/lib/python3.10/site-packages')
from common import load_vae,resize_center_crop,sample_nearest_frames,write_video
def main():
 rows=[r for r in csv.DictReader((ROOT/'data/train100_clips.csv').open()) if r['split']=='train' and not (ROOT/'data/cache/train'/f"{r['clip_id']}.pt").exists()]
 vae=load_vae();
 with torch.inference_mode():
  for i,r in enumerate(rows,1):
   frames=[resize_center_crop(x) for x in sample_nearest_frames(r['source_video'],json.loads(r['sample_timestamps']))]; lr=[cv2.resize(x,(288,160),interpolation=cv2.INTER_AREA) for x in frames]
   hr=torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float().div(127.5).sub(1); lo=torch.from_numpy(np.stack(lr)).permute(3,0,1,2).float().div(127.5).sub(1)
   zh=vae.encode(hr.unsqueeze(0).to(torch.bfloat16),device='cuda',tiled=False)[0].half().cpu(); zl=vae.encode(lo.unsqueeze(0).to(torch.bfloat16),device='cuda',tiled=False)[0].half().cpu()
   assert tuple(zh.shape)==(16,5,60,108) and tuple(zl.shape)==(16,5,20,36)
   p=ROOT/'data/cache/train'/f"{r['clip_id']}.pt";p.parent.mkdir(parents=True,exist_ok=True);torch.save({'clip_id':r['clip_id'],'video_id':r['video_id'],'source_video':r['source_video'],'z_lr':zl,'z_hr':zh},p)
   torch.save(torch.from_numpy(np.stack(frames)),ROOT/'data/gt_train'/f"{r['clip_id']}.pt")
   print(f'cache {i}/{len(rows)} {r["clip_id"]}',flush=True)
 print('NEW_CACHE_DONE',len(rows))
if __name__=='__main__':main()
