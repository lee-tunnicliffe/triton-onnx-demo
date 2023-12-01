import tensorflow as tf
import tensorflow_decision_forests as tfdf


def load_model():
    imported = tf.saved_model.load('failed-models/toy_model/1/model.savedmodel')


if __name__ == '__main__':
    load_model()
