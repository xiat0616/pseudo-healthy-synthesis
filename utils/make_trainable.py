def make_trainable(model, val):
    model.trainable = val
    for l in model.layers:
        try:
            for k in l.layers:
                make_trainable(k, val)
        except:
            pass
        l.trainable = val