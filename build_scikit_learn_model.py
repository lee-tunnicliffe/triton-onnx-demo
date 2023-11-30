import numpy as np
import onnxruntime as rt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from skl2onnx import to_onnx


def build_model():
    # Train a model.
    x, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)

    # Convert into ONNX format.
    options = {id(clf): {"zipmap": False}}
    onx = to_onnx(clf, x[:1], options=options)
    with open("./models/scikit_learn_model/1/model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # Compute the prediction with onnxruntime.
    sess = rt.InferenceSession("./models/scikit_learn_model/1/model.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: x.astype(np.float64)})[0]


if __name__ == '__main__':
    build_model()
