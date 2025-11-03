def iou(preds, gt):
    dim = None
    if len(preds.shape) == 4:
        dim = (1, 2, 3)

    intersection = (preds * gt).sum(dim=dim)
    union = (preds + gt).clamp(max=1).sum(dim=dim)

    iou = intersection / (union + 1e-7)

    gt_has_art = (gt.sum(dim=dim) != 0)
    if dim is None and not gt_has_art.item():
        return 1
    elif dim is not None:
        for i in range(iou.shape[0]):
            if not gt_has_art[i]:
                iou[i] = 1

    return iou.mean()
