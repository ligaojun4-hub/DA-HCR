"""Read-only audit of original data/checkpoints; write new audit artifacts only."""
import argparse, gc, hashlib, json, os, platform, shutil, sys, time
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parent.parent
LEGACY=Path('/root/autodl-tmp/robust_hash_project_scaffold/robust_hash_project_scaffold')
SUBMISSION=Path('/root/autodl-tmp/DA_HCR_SUBMISSION_WORKSPACE')
DATA=LEGACY/'data'
os.environ['DA_HCR_DATA_ROOT']=str(DATA)
EXPECTED={
 'caltech256':{'train':18197,'val':2881,'test_query':4351,'test_retrieval':4351},
 'mit_indoor67':{'train':9462,'val':1532,'test_query':2313,'test_retrieval':2313},
 'flowers102':{'train':5044,'val':775,'test_query':1185,'test_retrieval':1185},
}
DIRS={'caltech256':'caltech256_5baseline_exact30','mit_indoor67':'mit_indoor67_5baseline_exact30_with_dsh','flowers102':'flowers102_5baseline_exact30_with_dsh'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def save(path,data):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--hash-images',action='store_true');args=parser.parse_args()
 frozen=ROOT/'frozen_src'
 if not frozen.exists():shutil.copytree(SUBMISSION/'src',frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 sys.path.insert(0,str(frozen))
 import numpy as np, pandas as pd, torch, torchvision, PIL
 from robust_hash.data.datasets import resolve_image_path
 from robust_hash.data.transforms import ACTIVE_PROTOCOL,DEG_PROTOCOLS,build_image_tensor,_build_tensor_pipeline
 from robust_hash.models.hash_model import RobustHashModel
 from PIL import Image
 torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
 report={'stage':'B1','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'running','environment':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'pillow':PIL.__version__,'numpy':np.__version__,'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count()},'active_protocol':ACTIVE_PROTOCOL,'protocol':DEG_PROTOCOLS,'source_files':{},'datasets':{},'checkpoints':[],'errors':[]}
 if torch.cuda.is_available():report['environment']['gpu']=torch.cuda.get_device_name(0)
 for p in sorted(frozen.rglob('*.py')):report['source_files'][str(p.relative_to(frozen))]=sha(p)
 assert ACTIVE_PROTOCOL=='formal_v12_monotonic'
 content_groups=defaultdict(list)
 for ds,expected in EXPECTED.items():
  print('AUDIT DATA',ds,flush=True)
  path=DATA/'splits'/ds/'seed_42'/'split_manifest.csv';df=pd.read_csv(path)
  counts={str(k):int(v) for k,v in df.groupby('split').size().items()}
  assert counts==expected,(ds,counts,expected)
  assert not df.rel_path.duplicated().any(),(ds,'duplicate rel_path')
  assert sorted(df.label.unique().tolist())==list(range(df.label.nunique()))
  roles={k:set(df[df.split==k].rel_path.astype(str)) for k in expected}
  assert all(not(roles[x]&roles[y]) for i,x in enumerate(roles) for y in list(roles)[i+1:])
  labels=set(df[df.split=='test_retrieval'].label);assert all(y in labels for y in df[df.split=='test_query'].label)
  missing=[];canonical=defaultdict(list);sizes=0
  for i,row in df.iterrows():
   try:
    f=Path(resolve_image_path(row,ds));canonical[str(f.resolve())].append(row.rel_path);sizes+=f.stat().st_size
    if args.hash_images:content_groups[sha(f)].append({'dataset':ds,'split':row.split,'rel_path':row.rel_path})
   except FileNotFoundError as e:missing.append(str(e))
   if i and i%5000==0:print(ds,'checked',i,flush=True)
  assert not missing,(ds,missing[:10])
  assert all(len(v)==1 for v in canonical.values()),(ds,'aliasing paths')
  first=df[df.split=='train'].iloc[0];image=resolve_image_path(first,ds)
  clean=build_image_tensor(image,224,'clean');direct=_build_tensor_pipeline(224)(Image.open(image).convert('RGB'))
  assert torch.equal(clean,direct),'clean identity mismatch'
  repeat={s:bool(torch.equal(build_image_tensor(image,224,s),build_image_tensor(image,224,s))) for s in ['weak','medium','strong']}
  assert all(repeat.values())
  report['datasets'][ds]={'manifest':str(path),'sha256':sha(path),'counts':counts,'classes':int(df.label.nunique()),'resolved_all_images':True,'distinct_canonical_paths':True,'image_bytes':sizes,'clean_identity_pass':True,'old_noise_repeatability':repeat,'content_digest_check':bool(args.hash_images)}
  save(ROOT/'artifacts'/'B1_preflight.json',report)
 for ds,folder in DIRS.items():
  for method in ['base','dasupcon']:
   ck=LEGACY/'outputs'/'formal_v12'/folder/('hashnet_'+method)/'run'/'best_val.pt'
   print('AUDIT WEIGHT',ds,method,flush=True)
   payload=torch.load(ck,map_location='cpu',weights_only=False)
   ca=payload.get('args',{});state=payload['model_state_dict']
   assert any(k.startswith('backbone.layer1.') for k in state),'Not ResNet18'
   assert not any(k.startswith('backbone.features.') for k in state),'Fallback CNN'
   assert int(ca.get('hash_bits',64))==64
   assert ca.get('dataset',ds)==ds and int(ca.get('split_seed',42))==42
   assert ca.get('baseline','hashnet')=='hashnet'
   model=RobustHashModel(int(payload['num_classes']),64,'resnet18',False)
   assert hasattr(model.backbone,'layer1') and model.backbone.__class__.__name__=='ResNet'
   model.load_state_dict(state,strict=True)
   config=ck.parent/'config.json';oldmetrics=ck.parent/'test_metrics.json'
   entry={'dataset':ds,'method':method,'path':str(ck),'sha256':sha(ck),'size':ck.stat().st_size,'epoch':payload['epoch'],'args':ca,'checkpoint_metrics':payload.get('metrics',{}),'num_classes':int(payload['num_classes']),'architecture_verified':'torchvision ResNet18','strict_load_pass':True,'config':json.loads(config.read_text()) if config.exists() else None,'archived_test_metrics':json.loads(oldmetrics.read_text()) if oldmetrics.exists() else None}
   report['checkpoints'].append(entry)
   del model,payload,state;gc.collect()
   save(ROOT/'artifacts'/'B1_preflight.json',report)
 if args.hash_images:
  duplicates=[v for v in content_groups.values() if len(v)>1]
  leakage=[v for v in duplicates if len({(x['dataset'],x['split']) for x in v})>1]
  save(ROOT/'artifacts'/'B1_image_duplicates.json',{'exact_content_duplicate_groups':duplicates,'cross_role_or_dataset_groups':leakage,'note':'Byte-identical files only; near duplicates require separate inspection.'})
  report['exact_content_duplicates']={'groups':len(duplicates),'cross_role_or_dataset_groups':len(leakage),'needs_review':bool(leakage)}
 report['status']='technical_audit_complete'
 report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 save(ROOT/'artifacts'/'B1_preflight.json',report)
 print('B1_PREFLIGHT_COMPLETE',report['environment'],flush=True)

if __name__=='__main__':main()
