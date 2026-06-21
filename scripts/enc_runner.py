"""Self-contained raw-ONNX streaming encoder runner (no kaldifeat): lhotse fbank -> encoder_out.
Replicates recipe onnx_pretrained-streaming.py state handling exactly."""
import numpy as np, torch, onnxruntime as ort
from lhotse import Fbank, FbankConfig

_FB = None
def fbank(samples):
    global _FB
    if _FB is None:
        _FB = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80,
                                high_freq=-400, snip_edges=False, dither=0.0))
    f = _FB.extract(np.ascontiguousarray(samples.astype(np.float32)), 16000)
    return torch.from_numpy(np.asarray(f)).float()  # (T,80)

class Enc:
    def __init__(self, path, threads=2):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        m = self.sess.get_modelmeta().custom_metadata_map
        il = lambda s: list(map(int, s.split(",")))
        self.nl = il(m["num_encoder_layers"]); self.ed = il(m["encoder_dims"])
        self.ck = il(m["cnn_module_kernels"]); self.lc = il(m["left_context_len"])
        self.qhd = il(m["query_head_dims"]); self.vhd = il(m["value_head_dims"]); self.nh = il(m["num_heads"])
        self.T = int(m["T"]); self.offset = int(m["decode_chunk_len"])
        self.reset()
    def reset(self, bs=1):
        self.states = []
        for i in range(len(self.nl)):
            kd = self.qhd[i]*self.nh[i]; ed = self.ed[i]; nah = 3*ed//4; vd = self.vhd[i]*self.nh[i]; clp = self.ck[i]//2
            for _ in range(self.nl[i]):
                self.states += [
                    np.zeros((self.lc[i], bs, kd), np.float32),
                    np.zeros((1, bs, self.lc[i], nah), np.float32),
                    np.zeros((self.lc[i], bs, vd), np.float32),
                    np.zeros((self.lc[i], bs, vd), np.float32),
                    np.zeros((bs, ed, clp), np.float32),
                    np.zeros((bs, ed, clp), np.float32)]
        self.states.append(np.zeros((bs, 128, 3, 19), np.float32))
        self.states.append(np.zeros((bs,), np.int64))
    def _io(self, x):
        ei = {"x": x.numpy()}; eo = ["encoder_out"]
        def bio(ts, i):
            for nm, t in zip(["cached_key","cached_nonlin_attn","cached_val1","cached_val2","cached_conv1","cached_conv2"], ts):
                ei[f"{nm}_{i}"] = t; eo.append(f"new_{nm}_{i}")
        for i in range(len(self.states[:-2])//6): bio(self.states[i*6:(i+1)*6], i)
        ei["embed_states"] = self.states[-2]; eo.append("new_embed_states")
        ei["processed_lens"] = self.states[-1]; eo.append("new_processed_lens")
        return ei, eo
    def run_chunk(self, x):  # x (1,T,80)
        ei, eo = self._io(x); out = self.sess.run(eo, ei); self.states = out[1:]
        return torch.from_numpy(out[0])  # (1,T',512)
    def encode_full(self, samples, tail=2.0):
        """Return full encoder_out (1, Ttot, 512) for a 1-D sample array."""
        wav = torch.cat([torch.from_numpy(samples.astype(np.float32)), torch.zeros(int(tail*16000))])
        feats = fbank(wav.numpy())  # (T,80)
        self.reset()
        outs = []; npf = 0
        while feats.shape[0] - npf >= self.T:
            seg = feats[npf:npf+self.T].unsqueeze(0); npf += self.offset
            outs.append(self.run_chunk(seg))
        if not outs: return torch.zeros(1,0,512)
        return torch.cat(outs, dim=1)
