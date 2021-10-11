'''
    基于TASTTEXT的评价二分类
    基于词
    带有Multi-Sample Dropout
    endpoint
'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, Embedding, GlobalAvgPool1D
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import replace

import numpy as np
import os
import jieba
import matplotlib.pyplot as plt
from utils import load_vocab, single_example_parser_eb, batched_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

params_label_size = 2
params_batch_size = 10
params_drop_rate = 0.2

params_drop_rate_project = 0.2
params_drop_num_project = 8

params_epochs = 100
params_lr = 1.0e-3
params_eps = 1.0e-6
params_patience = 7
params_mode = "train0"

params_check = "modelfiles/fasttext_msdrop_yy_" + str(params_drop_num_project)

tf.random.set_seed(100)
np.random.seed(100)


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

        newembedding = np.load("data/OriginalFile/embedding_matrix.npy")
        self.word_embeddings = Embedding(289689, 100, name='word_embeddings',
                                         embeddings_initializer=tf.constant_initializer(newembedding),
                                         dtype=tf.float32,
                                         )
        self.dropout = Dropout(params_drop_rate)

    def call(self, sen, **kwargs):
        return self.dropout(self.word_embeddings(sen)), tf.greater(sen, 0)


class FASTTEXT(Layer):
    def __init__(self, **kwargs):
        super(FASTTEXT, self).__init__(**kwargs)

        self.globalaveragepool1d = GlobalAvgPool1D()

    def call(self, inputs, **kwargs):
        now, mask = inputs

        return self.globalaveragepool1d(now, mask)


class Project(Layer):
    def __init__(self, **kwargs):
        super(Project, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dropoutproject = [Dropout(params_drop_rate_project) for _ in range(params_drop_num_project)]

        self.project = Dense(1, activation="sigmoid", kernel_initializer=TruncatedNormal(stddev=0.02))

        super(Project, self).build(input_shape)

    def call(self, inputs, **kwargs):
        now, label = inputs
        labelexpand = tf.expand_dims(tf.cast(label, tf.int32), axis=1)

        loss = 0.0
        logit = None

        for i in range(params_drop_num_project):
            output = self.dropoutproject[i](now)
            logits = self.project(output)
            loss += tf.reduce_mean(binary_crossentropy(labelexpand, logits))

            if i == 0:
                logit = logits
            else:
                logit += logits

        loss /= params_drop_num_project
        logit /= params_drop_num_project

        predict = tf.cast(tf.greater(logit, 0.5), tf.int32)
        accsum = tf.reduce_sum(tf.cast(tf.equal(predict, labelexpand), tf.float32))

        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labelexpand, 1), tf.equal(predict, 1)), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labelexpand, 1), tf.equal(predict, 0)), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labelexpand, 0), tf.equal(predict, 1)), tf.float32))

        prediction = tf.squeeze(predict, axis=-1)

        return loss, accsum, tf.shape(label)[0], tp, tn, fp, prediction


@tf.function(experimental_relax_shapes=True)
def train_step(data, model, optimizer):
    with tf.GradientTape() as tape:
        loss, accsum, num, tp, tn, fp, _ = model(data, training=True)

    trainable_variables = model.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, accsum, num, tp, tn, fp


@tf.function(experimental_relax_shapes=True)
def dev_step(data, model):
    loss, accsum, num, tp, tn, fp, _ = model(data, training=False)

    return loss, accsum, num, tp, tn, fp


class TextClassify():
    def __init__(self):
        self.word_dict = load_vocab("data/OriginalFile/word_dict.txt")

    def build_model(self, summary=True):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)  # 句子

        label = Input(shape=[], name='label', dtype=tf.int32)

        now, mask = Embeddings(name="embeddings")(sen)

        ft = FASTTEXT(name="fasttext")(inputs=(now, mask))

        loss, accsum, num, tp, tn, fp, prediction = Project()(inputs=(ft, label))

        model = Model(inputs=[sen, label], outputs=[loss, accsum, num, tp, tn, fp, prediction])

        if summary:
            model.summary()
            for tv in model.variables:
                print(tv.name, " : ", tv.shape)

        return model

    def train(self, train_file, val_file):

        model = self.build_model()

        if params_mode == "train1":
            model.load_weights(params_check + "/fasttext.h5")

        optimizer = Adam(learning_rate=params_lr)

        train_batch = batched_data(train_file,
                                   single_example_parser_eb,
                                   params_batch_size,
                                   {"sen": [-1], "label": []},
                                   shuffle=False)
        val_batch = batched_data(val_file,
                                 single_example_parser_eb,
                                 params_batch_size,
                                 {"sen": [-1], "label": []},
                                 shuffle=False)

        f1_max = 0.0
        greater = 0

        history = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }

        for epoch in range(params_epochs):
            loss = []
            acc = []
            num = []

            tp = []
            tn = []
            fp = []

            for batch, data in enumerate(train_batch):
                loss_, accsum_, num_, tp_, tn_, fp_ = train_step(data, model, optimizer)

                loss.append(loss_)
                acc.append(accsum_)
                num.append(num_)

                tp.append(tp_)
                tn.append(tn_)
                fp.append(fp_)

                loss_av = np.mean(loss)
                acc_av = np.sum(acc) / (np.sum(num) + params_eps)

                tpsum = np.sum(tp)
                tnsum = np.sum(tn)
                fpsum = np.sum(fp)
                precision = tpsum / (tpsum + fpsum + params_eps)
                recall = tpsum / (tpsum + tnsum + params_eps)
                f1 = 2.0 * precision * recall / (precision + recall + params_eps)

                print(
                    '\rEpoch %d/%d %d -loss: %.6f -acc:%6.1f -precision:%6.1f -recall:%6.1f -f1:%6.1f' % (
                        epoch + 1, params_epochs, batch + 1,
                        loss_av, 100.0 * acc_av,
                        100.0 * precision, 100.0 * recall, 100.0 * f1,
                    ), end=""
                )

            history["loss"].append(loss_av)
            history["acc"].append(acc_av)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["f1"].append(f1)

            loss_val, acc_val, precision_val, recall_val, f1_val = self.dev_train(val_batch, model)

            print(" -val_loss: %.6f -val_acc:%6.1f -val_precision:%6.1f -val_recall:%6.1f -val_f1:%6.1f\n" % (
                loss_val, 100.0 * acc_val,
                100.0 * precision_val, 100.0 * recall_val, 100.0 * f1_val))

            history["val_loss"].append(loss_val)
            history["val_acc"].append(acc_val)
            history["val_precision"].append(precision_val)
            history["val_recall"].append(recall_val)
            history["val_f1"].append(f1_val)

            if f1_val > f1_max:
                model.save_weights(params_check + '/tta.h5')
                f1_max = f1_val
                greater = 0
            else:
                greater += 1

                if greater == params_patience:
                    break

        with open(params_check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history))

    def dev_train(self, dev_data, model):
        loss = []
        acc = []
        num = []

        tp = []
        tn = []
        fp = []

        for batch, data in enumerate(dev_data):
            loss_, accsum_, num_, tp_, tn_, fp_ = dev_step(data, model)

            loss.append(loss_)
            acc.append(accsum_)
            num.append(num_)

            tp.append(tp_)
            tn.append(tn_)
            fp.append(fp_)

        loss_av = np.mean(loss)
        acc_av = np.sum(acc) / (np.sum(num) + params_eps)
        tp_sum = np.sum(tp)
        tn_sum = np.sum(tn)
        fp_sum = np.sum(fp)

        precision = tp_sum / (tp_sum + fp_sum + params_eps)
        recall = tp_sum / (tp_sum + tn_sum + params_eps)
        f1 = 2.0 * precision * recall / (precision + recall + params_eps)

        return loss_av, acc_av, precision, recall, f1

    def plot(self):
        with open(params_check + "/history.txt", "r", encoding="utf-8") as fr:
            history = fr.read()
            history = eval(history)

            plt.subplot(2, 2, 1)
            plt.plot(history["val_loss"])
            plt.title("val_loss")
            plt.subplot(2, 2, 2)
            plt.plot(history["val_acc"])
            plt.title("val_acc")
            plt.subplot(2, 1, 2)
            plt.plot(history["val_precision"])
            plt.plot(history["val_recall"])
            plt.plot(history["val_F1"])
            plt.title("val_precision,recall,F1")
            plt.legend(['P', 'R', 'F1'], loc='best', prop={'size': 4})
            plt.tight_layout()
            plt.savefig(params_check + "/record.png", dpi=500, bbox_inches="tight")
            plt.show()


def main():
    if not os.path.exists(params_check):
        os.makedirs(params_check)

    train_file = [
        'data/TFRecordFile/train_eb_word.tfrecord',
    ]
    val_file = [
        'data/TFRecordFile/val_eb_word.tfrecord',
    ]
    tc = TextClassify()

    if params_mode.startswith('train'):
        tc.train(train_file, val_file)

    elif params_mode == "plot":
        tc.plot()


if __name__ == "__main__":
    main()
