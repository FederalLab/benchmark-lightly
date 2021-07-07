def top_one_acc(target, predict):
    # Start a standard forward inference pass.
    predict = predict.max(1, keepdim=True)[1]
    return predict.eq(target.view_as(
        predict)).sum().item() / len(target)
