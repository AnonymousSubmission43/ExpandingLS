import torch


def edit(latents, pca, edit_directions, alpha=1):
    edit_latents = []
    for latent in latents:
        for pca_idx, start, end, strengths in edit_directions:
            # for strength in [-strengths*1.5, -strengths, -strengths*0.8, -strengths*0.5, 0, strengths*0.5, strengths*0.8, strengths, strengths*1.5]:
            for strength in [strengths]:
                if strength*alpha == 0:
                    edit_latents.append(latent)
                else:
                    delta = get_delta(pca, latent, pca_idx, strength*alpha)
                    delta_padded = torch.zeros(latent.shape).to('cuda')
                    delta_padded[start:end] += delta.repeat(end - start, 1)
                    edit_latents.append(latent + delta_padded)
    return torch.stack(edit_latents)


def get_delta(pca, latent, idx, strength):
    # pca: ganspace checkpoint. latent: (16, 512) w+
    w_centered = latent - pca['mean'].to('cuda')
    lat_comp = pca['comp'].to('cuda')
    lat_std = pca['std'].to('cuda')
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta
