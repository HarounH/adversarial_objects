import torch


def combine_objects(vs, fs, ts, cat_fn=lambda ls: torch.cat(ls, dim=1)):
    # Takes 3 lists of arrays without batch_sizes.
    # Each array represents the corresponding property of the underlying object.
    n = len(vs)
    v = vs[0]
    f = fs[0]
    t = ts[0]
    for i in range(1, n):
        face_offset = v.shape[1]  # Works for numpy and also torch
        v = cat_fn([v, vs[i]])  # torch.cat([v, vs[i]], dim=1)
        f = cat_fn([f, face_offset + fs[i]])  # torch.cat([f, face_offset + fs[i]], dim=1)
        t = cat_fn([t, ts[i]])  # torch.cat([t, ts[i]], dim=1)
    return [v, f, t]


def combine_images_in_order(image_list, output_shape, color_dim=1, eps=0.0, normalize_result=False):
    result = torch.zeros(output_shape, dtype=torch.float, device='cuda')
    # result = torch.zeros(image_list[0].shape, dtype=torch.float, device='cuda')
    for image in image_list:
        selector = (torch.abs(image).sum(dim=color_dim, keepdim=True) <= eps).float()
        result = result * selector + image
    if normalize_result:
        result = (result - result.min()) / (result.max() - result.min())
    return result
