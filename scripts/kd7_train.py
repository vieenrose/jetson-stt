"""Break the data wall: fine-tune the warm small-bilingual student on ~260h (cached fbank + GT text),
char+bpe tokenizer, SpecAugment, relabel->Traditional. Eval on 500 CV-zh-TW vs warm baseline 0.108 / X-ASR 0.064."""
import os, sys, re, random, glob, argparse
R = "icefall/egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
sys.path[:0] = [R, "icefall"]
import numpy as np, torch, k2
import train as T7, sentencepiece as spm
from opencc import OpenCC
from lhotse import Fbank, FbankConfig
from beam_search import greedy_search_batch

STEPS=int(os.environ.get("STEPS","16000")); BS=int(os.environ.get("BS","14"))
LR=float(os.environ.get("LR","1e-4")); WARM=int(os.environ.get("WARM","500"))
EVAL_EVERY=int(os.environ.get("EVAL_EVERY","2000")); dev="cuda"
random.seed(0); torch.manual_seed(0)
t2s=OpenCC("t2s"); s2twp=OpenCC("s2twp"); CJK=re.compile(r"[㐀-鿿]")

p=argparse.ArgumentParser(); T7.add_model_arguments(p)
a=p.parse_args(["--num-encoder-layers","2,2,2,2,2","--feedforward-dims","768,768,768,768,768",
   "--encoder-dims","256,256,256,256,256","--encoder-unmasked-dims","192,192,192,192,192",
   "--nhead","4,4,4,4,4","--attention-dims","192,192,192,192,192"])
pr=T7.get_params(); pr.update(vars(a)); pr.blank_id=0; pr.vocab_size=6254; pr.context_size=2
pr.decoder_dim=512; pr.joiner_dim=512; pr.use_transducer=True; pr.use_ctc=False
m=T7.get_transducer_model(pr) if hasattr(T7,"get_transducer_model") else T7.get_model(pr)
miss,unexp=m.load_state_dict(torch.load(open("smb_ck.txt").read().strip(),map_location="cpu",weights_only=False)["model"],strict=False)
assert not miss and not unexp; m.to(dev); print(f"warm-started small-bilingual {sum(x.numel() for x in m.parameters())/1e6:.1f}M",flush=True)

sp=spm.SentencePieceProcessor(); sp.load(open("smb_bpe.txt").read().strip())
lines=[l.rstrip("\n") for l in open(open("smb_tok.txt").read().strip(),encoding="utf-8")]
tok2id={l.rsplit(" ",1)[0]:int(l.rsplit(" ",1)[1]) for l in lines if len(l.rsplit(" ",1))==2}
id2surf={}
for tok,i in tok2id.items():
    pre="▁" if tok.startswith("▁") else ""; body=tok[len(pre):]
    id2surf[i]=pre+(s2twp.convert(body) if body and all(CJK.match(c) for c in body) else body)
def detok(ids): return "".join(id2surf.get(i,"") for i in ids if i in id2surf).replace("▁"," ").strip()
def tok_target(text):
    ids=[]; i=0; n=len(text)
    while i<n:
        c=text[i]
        if CJK.match(c):
            tid=tok2id.get("▁"+c, tok2id.get(c));
            if tid is not None: ids.append(tid)
            i+=1
        elif re.match(r"[A-Za-z0-9']",c):
            mm=re.match(r"[A-Za-z0-9']+",text[i:]);
            for pc in sp.encode(mm.group(0).upper(),out_type=str):
                if pc in tok2id: ids.append(tok2id[pc])
            i+=len(mm.group(0))
        else: i+=1
    return ids

F=[]; Y=[]
for sh in sorted(glob.glob("kd7_feats/shard_*.npz")):
    z=np.load(sh,allow_pickle=True)
    for f,t in zip(z["eo"],z["txt"]):
        ids=[i for i in tok_target(str(t)) if 0<i<6254]
        if 0<len(ids)<=80 and f.shape[0]<=1500: F.append(f); Y.append(ids)
print(f"train clips: {len(F)}",flush=True)
ev=np.load("cv_audio_cache_500.npz",allow_pickle=True); EA=list(ev["a"]); ER=list(ev["r"])
_FB=Fbank(FbankConfig(sampling_rate=16000,num_mel_bins=80,snip_edges=False,dither=0.0))
def fbk(a): return torch.from_numpy(np.asarray(_FB.extract(np.ascontiguousarray(a.astype(np.float32)),16000))).float()
def lev(a,b):
    dp=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        q=dp[0];dp[0]=i
        for j,y in enumerate(b,1):
            c=dp[j];dp[j]=min(dp[j-1]+1,dp[j]+1,q+(x!=y));q=c
    return dp[len(b)]
@torch.no_grad()
def cer():
    m.eval(); e=t=eR=tR=0
    for i in range(0,len(EA),16):
        fs=[fbk(a) for a in EA[i:i+16]]; Tm=max(f.shape[0] for f in fs)
        x=torch.zeros(len(fs),Tm,80,device=dev); L=torch.zeros(len(fs),dtype=torch.long,device=dev)
        for j,f in enumerate(fs): x[j,:f.shape[0]]=f.to(dev); L[j]=f.shape[0]
        eo,el=m.encoder(x,L)
        for j,h in enumerate(greedy_search_batch(model=m,encoder_out=eo,encoder_out_lens=el)):
            txt=detok(h); ref=ER[i+j]
            e+=lev([c for c in t2s.convert(txt) if CJK.match(c)],[c for c in t2s.convert(ref) if CJK.match(c)]); t+=len([c for c in t2s.convert(ref) if CJK.match(c)])
            eR+=lev([c for c in txt if CJK.match(c)],[c for c in ref if CJK.match(c)]); tR+=len([c for c in ref if CJK.match(c)])
    m.train(); return e/max(t,1), eR/max(tR,1)
def specaug(x,L):
    for b in range(x.shape[0]):
        for _ in range(2):
            f=random.randint(0,27); f0=random.randint(0,max(0,80-f)); x[b,:,f0:f0+f]=0
        tl=int(L[b])
        for _ in range(2):
            tw=random.randint(0,max(1,tl//10)); t0=random.randint(0,max(0,tl-tw)); x[b,t0:t0+tw,:]=0
    return x
r0,rt0=cer(); print(f"[eval@0] recognition {r0:.4f} | raw-Traditional {rt0:.4f}  (warm baseline; X-ASR 0.064/0.068)",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-2)
def set_lr(s):
    lr=LR*min(1.0,s/WARM)*(max(0.05,(STEPS-s)/STEPS) if s>WARM else 1.0)
    for g in opt.param_groups: g["lr"]=lr
best=r0; m.train()
for step in range(1,STEPS+1):
    set_lr(step); bi=[random.randrange(len(F)) for _ in range(BS)]
    fs=[torch.from_numpy(F[k].astype(np.float32)) for k in bi]; Tm=max(f.shape[0] for f in fs)
    x=torch.zeros(BS,Tm,80); L=torch.zeros(BS,dtype=torch.long)
    for j,f in enumerate(fs): x[j,:f.shape[0]]=f; L[j]=f.shape[0]
    x=specaug(x.to(dev),L); L=L.to(dev); y=k2.RaggedTensor([Y[k] for k in bi]).to(dev)
    try:
        simple,pruned,*_=m(x,L,y,prune_range=5,am_scale=0.0,lm_scale=0.0); loss=0.5*simple+pruned
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    except Exception as ex:
        print("step",step,"skip",str(ex)[:80],flush=True); continue
    if step%200==0: print(f"step {step}/{STEPS} lr={opt.param_groups[0]['lr']:.1e} loss/tok={pruned.item()/max(sum(len(Y[k]) for k in bi),1):.3f}",flush=True)
    if step%EVAL_EVERY==0:
        rc,rtc=cer(); tag="  <-- best" if rc<best else ""
        if rc<best: best=rc; torch.save({"model":m.state_dict()},"ft_runs/kd7_best.pt")
        print(f"  [eval@{step}] recognition {rc:.4f} | raw-Traditional {rtc:.4f}{tag}",flush=True)
print(f"\n=== KD7 (Breeze-relabeled Taiwan): best recognition CER {best:.4f} vs warm 0.108 / X-ASR 0.064 (Taiwan-only, no Mainland) ===",flush=True)
