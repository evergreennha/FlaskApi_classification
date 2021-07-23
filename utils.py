# Model predict 3 classes: cat. dog, panda

from libs import *

def load_model():
    model = models.resnet18(pretrained = True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)
    model.load_state_dict(torch.load("gender.pth"))     # Load your weight file
    model.to(torch.device("cpu"))
    model.eval()
    return model

def img_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    img_tranform = transforms.Compose([
            transforms.Resize((224, 105)),
            transforms.ToTensor(),
            normalize
        ])
    return img_tranform