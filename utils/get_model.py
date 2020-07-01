def get_model(model, name):
    for l in model.layers:
        if l.name == name:
            return l
    print('Not found layer with name ' + name)
    return None