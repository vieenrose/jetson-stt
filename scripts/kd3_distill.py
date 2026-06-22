"""Distill X-ASR (zh-TW native) INTO the small-bilingual zh-en streaming student (zipformer1, 30.9M).
Warm-start the real checkpoint, relabel tokens -> Traditional (s2twp), fine-tune on X-ASR pseudo-labels.
Output: native Traditional zh-TW + English, no OpenCC, in a model ~5x smaller/faster than X-ASR."""
import os, sys, re, random, argparse
R = "icefall/egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
sys.path[:0] = [R, "icefall"]
import numpy as np, torch, k2, soundfile as sf
import train as T7, sentencepiece as spm
from opencc import OpenCC
from lhotse import Fbank, FbankConfig
from beam_search import greedy_search_batch

STEPS = int(os.environ.get("STEPS", "4000")); BS = int(os.environ.get("BS", "12"))
LR = float(os.environ.get("LR", "1.5e-4")); WARM = int(os.environ.get("WARM", "200"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "800")); dev = "cuda"
random.seed(0); torch.manual_seed(0)
t2s = OpenCC("t2s"); s2twp = OpenCC("s2twp"); CJK = re.compile(r"[㐀-鿿]")

# build + warm-start student
p = argparse.ArgumentParser(); T7.add_model_arguments(p)
a = p.parse_args(["--num-encoder-layers","2,2,2,2,2","--feedforward-dims","768,768,768,768,768",
    "--encoder-dims","256,256,256,256,256","--encoder-unmasked-dims","192,192,192,192,192",
    "--nhead","4,4,4,4,4","--attention-dims","192,192,192,192,192"])
pr = T7.get_params(); pr.update(vars(a))
pr.blank_id=0; pr.vocab_size=6254; pr.context_size=2; pr.decoder_dim=512; pr.joiner_dim=512; pr.use_transducer=True; pr.use_ctc=False
m = T7.get_transducer_model(pr) if hasattr(T7,"get_transducer_model") else T7.get_model(pr)
miss,unexp = m.load_state_dict(torch.load(open("smb_ck.txt").read().strip(),map_location="cpu",weights_only=False)["model"], strict=False)
assert not miss and not unexp, (len(miss),len(unexp))
m.to(dev); print(f"student {sum(x.numel() for x in m.parameters())/1e6:.1f}M, warm-started clean", flush=True)

sp = spm.SentencePieceProcessor(); sp.load(open("smb_bpe.txt").read().strip())
# relabel tokens -> Traditional; build id->Traditional surface for detok
lines=[l.rstrip("\n") for l in open(open("smb_tok.txt").read().strip(),encoding="utf-8")]
id2surf={}
for l in lines:
    pp=l.rsplit(" ",1)
    if len(pp)!=2: continue
    tok,i=pp[0],int(pp[1]); pre="▁" if tok.startswith("▁") else ""; body=tok[len(pre):]
    surf = (pre + (s2twp.convert(body) if body and all(CJK.match(c) for c in body) else body))
    id2surf[i]=surf
def detok(ids): return "".join(id2surf.get(i,"") for i in ids if i in id2surf).replace("▁"," ").strip()
# original (Simplified) char+bpe tokenizer for ENCODING targets: Chinese char -> bare token id; English -> 500-bpe pieces
tok2id={}
for l in lines:
    pp=l.rsplit(" ",1)
    if len(pp)==2: tok2id[pp[0]]=int(pp[1])
def tok_target(text):
    ids=[]; i=0; n=len(text)
    while i<n:
        c=text[i]
        if CJK.match(c):
            tid=tok2id.get("▁"+c, tok2id.get(c))
            if tid is not None: ids.append(tid)
            i+=1
        elif re.match(r"[A-Za-z0-9']",c):
            mm=re.match(r"[A-Za-z0-9']+",text[i:]); w=mm.group(0)
            for pc in sp.encode(w.upper(),out_type=str):
                if pc in tok2id: ids.append(tok2id[pc])
            i+=len(w)
        else: i+=1
    return ids

FB=Fbank(FbankConfig(sampling_rate=16000,num_mel_bins=80,snip_edges=False,dither=0.0))
def fb(a): return torch.from_numpy(np.asarray(FB.extract(np.ascontiguousarray(a.astype(np.float32)),16000))).float()
def fb_paths(paths):
    fs=[fb((lambda d: d.mean(1) if d.ndim>1 else d)(sf.read(w,dtype="float32")[0])) for w in paths]
    L=torch.tensor([f.shape[0] for f in fs]); Tm=int(L.max()); x=torch.zeros(len(fs),Tm,80)
    for i,f in enumerate(fs): x[i,:f.shape[0]]=f
    return x.to(dev),L.to(dev)

# data: X-ASR pseudo-labels (kd_manifest), target = student bpe ids
train=[]
for l in open("kd_manifest.tsv",encoding="utf-8"):
    if "\t" not in l: continue
    w,txt=l.rstrip("\n").split("\t",1); ids=[i for i in tok_target(txt) if 0<i<6254]
    if 0<len(ids)<=80: train.append((w,ids))
print(f"distill pairs: {len(train)}", flush=True)
ev=np.load("cv_audio_cache_500.npz",allow_pickle=True); EA=list(ev["a"]); ER=list(ev["r"])
def lev(a,b):
    dp=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        pr2=dp[0];dp[0]=i
        for j,y in enumerate(b,1):
            c=dp[j];dp[j]=min(dp[j-1]+1,dp[j]+1,pr2+(x!=y));pr2=c
    return dp[len(b)]
@torch.no_grad()
def cer():
    m.eval(); e=t=eR=tR=0
    for i in range(0,len(EA),16):
        fs=[fb(a) for a in EA[i:i+16]]; Tm=max(f.shape[0] for f in fs)
        x=torch.zeros(len(fs),Tm,80,device=dev); L=torch.zeros(len(fs),dtype=torch.long,device=dev)
        for j,f in enumerate(fs): x[j,:f.shape[0]]=f.to(dev); L[j]=f.shape[0]
        eo,el=m.encoder(x,L)
        for j,h in enumerate(greedy_search_batch(model=m,encoder_out=eo,encoder_out_lens=el)):
            txt=detok(h); ref=ER[i+j]
            H=[c for c in t2s.convert(txt) if CJK.match(c)]; Rn=[c for c in t2s.convert(ref) if CJK.match(c)]
            e+=lev(H,Rn); t+=len(Rn)                                  # recognition (neutralized)
            Ht=[c for c in txt if CJK.match(c)]; Rt=[c for c in ref if CJK.match(c)]
            eR+=lev(Ht,Rt); tR+=len(Rt)                               # raw Traditional (no OpenCC)
    m.train(); return e/max(t,1), eR/max(tR,1)
r0,rt0=cer(); print(f"[eval@0] recognition CER {r0:.4f} | raw-Traditional CER {rt0:.4f}  (X-ASR teacher 0.064/0.068)", flush=True)

opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-2)
def set_lr(s):
    lr=LR*min(1.0,s/WARM)*(max(0.05,(STEPS-s)/STEPS) if s>WARM else 1.0)
    for g in opt.param_groups: g["lr"]=lr
best=1e9; m.train()
for step in range(1,STEPS+1):
    set_lr(step); items=[random.choice(train) for _ in range(BS)]
    x,xl=fb_paths([w for w,_ in items]); y=k2.RaggedTensor([ids for _,ids in items]).to(dev)
    try:
        simple,pruned,*_=m(x,xl,y,prune_range=5,am_scale=0.0,lm_scale=0.0); loss=0.5*simple+pruned
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    except Exception as ex:
        print("step",step,"skip",str(ex)[:90],flush=True); continue
    if step%200==0:
        nt=sum(len(i) for _,i in items); print(f"step {step}/{STEPS} lr={opt.param_groups[0]['lr']:.1e} loss/tok={pruned.item()/max(nt,1):.3f}",flush=True)
    if step%EVAL_EVERY==0:
        rc,rtc=cer(); tag="  <-- best" if rc<best else ""
        if rc<best: best=rc; torch.save({"model":m.state_dict()},"ft_runs/kd3_smallbi_best.pt")
        print(f"  [eval@{step}] recognition {rc:.4f} | raw-Traditional {rtc:.4f}{tag}",flush=True)
print(f"\n=== KD3: small-bilingual distilled best recognition CER {best:.4f} vs X-ASR 0.064 | started warm at {r0:.4f} (30.9M, ~5x smaller) ===",flush=True)
