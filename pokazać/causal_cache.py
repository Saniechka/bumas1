import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from causal_map import win_masks

QS = (30, 50)

#tworzy sie mapa cieplna czułych punktow

def compute_causal(bb, U8, path, dev='cuda', log=''):
    #przygotowanie
    N = len(U8)
    pst, pdone = path.replace('.npy', '_st.npy'), path.replace('.npy', '.done')
    if os.path.exists(pdone) and int(open(pdone).read()) >= N:
        return np.load(path, mmap_mode='r'), np.load(pst)
    out = np.load(path, mmap_mode='r+') if os.path.exists(path) else \
        np.lib.format.open_memmap(path, 'w+', np.float16, (N, len(QS), 64, 64))
    st = np.load(pst) if os.path.exists(pst) else np.zeros((N, len(QS), 3), np.float32)
    start = int(open(pdone).read()) if os.path.exists(pdone) else 0
    M = win_masks(dev); cnt = M.sum(0)
    t = time.time()

    with torch.no_grad():
        for n in range(start, N):
            x = to_t(U8[n:n + 1]).to(dev) # przesyła do gpu
            fs = to_t(np.stack([jpeg_u8(U8[n], q) for q in QS])).to(dev) #skompresowane wersje obrazka w pamięci
            with torch.autocast('cuda', dtype=torch.float16):
                z0 = bb(torch.cat([x, fs])).float() # odpowiedz dla oryginału i 2 wersji
            c0 = z0[:1].argmax(1) # czysta odpowiedz
            mg0 = margin(z0, c0.repeat(len(z0))) #pewność
            for j in range(len(QS)):
                f = fs[j:j + 1]
                md = []
                for s in range(0, len(M), 512): #  dzielimy okna na paczki po maks 512
                    m = M[s:s + 512, None]
                    with torch.autocast('cuda', dtype=torch.float16):
                        md.append(margin(bb(x + m * (f - x)).float(), c0.repeat(len(m)))) # zmywanie w oknie
                md = torch.cat(md)
                m0, mf = mg0[0].item(), mg0[1 + j].item() # pewność dla 2 wariantów
                drop = (m0 - md) / (abs(m0 - mf) + 0.5) #wskaznik spadku pewnosci 
                out[n, j] = (torch.einsum('w,wij->ij', drop, M) / cnt).cpu().numpy().astype(np.float16)#wskaźnik pewności dla maski siatek
                st[n, j] = [m0, mf, drop.max().item()]
            if (n + 1) % 100 == 0 or n + 1 == N:
                out.flush(); np.save(pst, st); open(pdone, 'w').write(str(n + 1))
                print(f'   mapy przyczynowe {log} {n + 1}/{N}  {time.time() - t:.0f}s', flush=True)
    return np.load(path, mmap_mode='r'), st


if __name__ == '__main__':
    split = sys.argv[1]
    os.makedirs('cache_v2', exist_ok=True)
    if split == 'val':
        rows = train_rows()
        va = np.sort(np.random.default_rng(0).permutation(len(rows))[:500])
        U8 = load_imgs([rows[i]['id'] for i in va], 'train')
    else:
        U8 = load_imgs(test_ids(), 'test')
    compute_causal(backbone('cuda'), U8, f'cache_v2/causal_{split}.npy', log=split)
