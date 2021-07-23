from libs import *
from utils import load_model, img_transforms

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

@app.route("/predict",methods=["POST"])
def predict():
    class_names = ["cat","dog","panda"]
    model = load_model()
    img_tranform = img_transforms()
    with torch.no_grad():
        if request.files.get("image"):
            img = request.files["image"].read()
            img = Image.open(io.BytesIO(img))
            img_trans = img_tranform(img)
            img_trans = img_trans.unsqueeze(0)
            outputs=model(img_trans)
            result = list(torch.max(outputs,1))
            accuracy = float(result[0][0].numpy())
            class_id = result[1][0].numpy()
            class_name = class_names[class_id]
            dict_result = {"accuracy": accuracy,"class_name":class_name}
    return json.dumps(dict_result, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    app.run()
