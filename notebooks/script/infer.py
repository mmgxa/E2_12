import torch
import json

import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classnames = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/model.scripted.pt")
    model.to(device).eval()    
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    print(type(data))
    data = np.array(data)
    print(type(data))
    print(data.shape)
    img_t = transforms(data)
    input_tensor = img_t.type(torch.float32).unsqueeze(0)
    return input_tensor


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    # res = predictions.cpu().numpy().tolist()
    res = predictions.cpu()
    res1 = F.softmax(res, dim=-1)
    res2 = {classnames[i]: round(res1[0][i].detach().numpy().tolist() *100,2) for i in torch.topk(res1[0], 5).indices}
    return json.dumps(res2)
