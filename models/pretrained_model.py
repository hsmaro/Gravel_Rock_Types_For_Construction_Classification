import timm

def load_model(model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=7)
    return model