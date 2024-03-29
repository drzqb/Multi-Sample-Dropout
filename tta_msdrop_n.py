'''
    基于TTA的评价二分类
    不带有Multi-Sample Dropout
'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Embedding, LayerNormalization, Dropout
from tensorflow.keras.initializers import TruncatedNormal
from transformers import BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, binary_crossentropy
from utils import focal_loss, bce_loss_weight, focal_loss_new
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import os, time
import matplotlib.pyplot as plt
from utils import single_example_parser_eb, batched_data, checkpoint_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

params_maxword = 512
params_vocab_size = 21128
params_type_vocab_size = 2

params_head = 12
params_hidden_size = 768
params_intermediate_size = 4 * 768

params_batch_size = 4
params_drop_rate = 0.2

params_drop_rate_project = 0.2

params_epochs = 100
params_lr = 1.0e-5
params_eps = 1.0e-6
params_patience = 7
params_mode = "train0"
params_check = "modelfiles/tta_msdrop_n"

tf.random.set_seed(0)
np.random.seed(0)


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


def softmax(a, mask):
    """
    :param a: B*1*ML2
    :param mask: B*1*ML2
    """
    return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def load_model_weights_from_checkpoint_bert(model, checkpoint_file):
    """Load trained official modelfiles from checkpoint.

    :param model: Built keras modelfiles.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    """
    loader = checkpoint_loader(checkpoint_file)

    weights = [
        loader('bert/embeddings/position_embeddings'),
        loader('bert/embeddings/word_embeddings'),
        loader('bert/embeddings/token_type_embeddings'),
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ]
    model.get_layer('embeddings').set_weights(weights)

    weights_a = []
    weights_f = []
    for i in range(12):
        pre = 'bert/encoder/layer_' + str(i) + '/'
        weights_a.extend([
            loader(pre + 'attention/self/query/kernel'),
            loader(pre + 'attention/self/query/bias'),
            loader(pre + 'attention/self/key/kernel'),
            loader(pre + 'attention/self/key/bias'),
            loader(pre + 'attention/self/value/kernel'),
            loader(pre + 'attention/self/value/bias'),
            loader(pre + 'attention/output/dense/kernel'),
            loader(pre + 'attention/output/dense/bias'),
            loader(pre + 'attention/output/LayerNorm/gamma'),
            loader(pre + 'attention/output/LayerNorm/beta')])

        weights_f.extend([
            loader(pre + 'intermediate/dense/kernel'),
            loader(pre + 'intermediate/dense/bias'),
            loader(pre + 'output/dense/kernel'),
            loader(pre + 'output/dense/bias'),
            loader(pre + 'output/LayerNorm/gamma'),
            loader(pre + 'output/LayerNorm/beta')])

    weights = weights_a + weights_f
    model.get_layer('encoder').set_weights(weights)

    weights = [
        loader('bert/pooler/dense/kernel'),
        loader('bert/pooler/dense/bias'),
    ]
    model.get_layer('pooler').set_weights(weights)


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        sequencemask = tf.greater(sen, 0)
        mask = tf.tile(tf.expand_dims(sequencemask, axis=1), [params_head, 1, 1])
        seqlen = tf.shape(sen)[1]

        return mask, seqlen


class LayerNormalizeAndDrop(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalizeAndDrop, self).__init__(**kwargs)

        self.layernorm = LayerNormalization(name="layernorm")
        self.dropout = Dropout(params_drop_rate)

    def call(self, inputs, **kwargs):
        return self.dropout(self.layernorm(inputs))


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

        self.word_embeddings = Embedding(params_vocab_size,
                                         params_hidden_size,
                                         embeddings_initializer=create_initializer(),
                                         dtype=tf.float32,
                                         name="word_embeddings")

        self.token_embeddings = Embedding(params_type_vocab_size,
                                          params_hidden_size,
                                          embeddings_initializer=create_initializer(),
                                          dtype=tf.float32,
                                          name='token_type_embeddings')

        self.position_embeddings = self.add_weight(name='position_embeddings',
                                                   shape=[params_maxword, params_hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=create_initializer())

        self.layernormanddrop = LayerNormalizeAndDrop(name="layernormanddrop")

    def call(self, inputs, **kwargs):
        sen, seqlen = inputs
        sen_embed = self.word_embeddings(sen)

        token_embed = self.token_embeddings(tf.zeros_like(sen, dtype=tf.int32))
        pos_embed = self.position_embeddings[:seqlen]

        all_embed = self.layernormanddrop(sen_embed + token_embed + pos_embed)

        return all_embed[:, 0], all_embed


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.dense_q = Dense(params_hidden_size,
                             name='query',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_k = Dense(params_hidden_size,
                             name='key',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_v = Dense(params_hidden_size,
                             name='value',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_o = Dense(params_hidden_size,
                             name='output',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dropoutsoft = Dropout(params_drop_rate)
        self.dropoutres = Dropout(params_drop_rate)
        self.layernorm = LayerNormalization(name='layernormattn')

    def call(self, inputs, **kwargs):
        x, y, mask = inputs

        x_tmp = tf.expand_dims(x, axis=1)

        q = tf.concat(tf.split(self.dense_q(x_tmp), params_head, axis=-1), axis=0)
        k = tf.concat(tf.split(self.dense_k(y), params_head, axis=-1), axis=0)
        v = tf.concat(tf.split(self.dense_v(y), params_head, axis=-1), axis=0)
        qk = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / tf.sqrt(params_hidden_size / params_head)
        attention_output = tf.squeeze(self.dense_o(tf.concat(
            tf.split(tf.matmul(self.dropoutsoft(softmax(qk, mask)), v), params_head, axis=0),
            axis=-1)), axis=1)

        return self.layernorm(x + self.dropoutres(attention_output))


class FeedFord(Layer):
    def __init__(self, **kwargs):
        super(FeedFord, self).__init__(**kwargs)

        self.dense_ffgelu = Dense(params_intermediate_size,
                                  kernel_initializer=create_initializer(),
                                  dtype=tf.float32,
                                  name='intermediate',
                                  activation=gelu)
        self.dense_ff = Dense(params_hidden_size,
                              kernel_initializer=create_initializer(),
                              dtype=tf.float32,
                              name='output')
        self.dropoutres = Dropout(params_drop_rate)
        self.layernorm = LayerNormalization(name='layernormffd')

    def call(self, inputs, **kwargs):
        return self.layernorm(inputs + self.dropoutres(self.dense_ff(self.dense_ffgelu(inputs))))


class Encoder(Layer):
    def __init__(self, layers, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layers = layers

        self.attention = [Attention(name="attnlayer_%d" % k) for k in range(self.layers)]
        self.ffd = [FeedFord(name="ffdlayer_%d" % k) for k in range(self.layers)]

    def get_config(self):
        config = {"layers": self.layers}
        base_config = super(Encoder, self).get_config()
        return dict(base_config, **config)

    def call(self, inputs, **kwargs):
        x, y, mask = inputs
        for k in range(self.layers):
            x = self.ffd[k](self.attention[k](inputs=(x, y, mask)))

        return x


class Pool(Layer):
    def __init__(self, **kwargs):
        super(Pool, self).__init__(**kwargs)

        self.pooldense = Dense(params_hidden_size,
                               kernel_initializer=create_initializer(),
                               activation=tf.tanh)

    def call(self, inputs, **kwargs):
        return self.pooldense(inputs)


class Project(Layer):
    def __init__(self, **kwargs):
        super(Project, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dropoutproject = Dropout(params_drop_rate_project)

        self.project = Dense(1, activation="sigmoid", kernel_initializer=TruncatedNormal(stddev=0.02))

        super(Project, self).build(input_shape)

    def call(self, inputs, **kwargs):
        output = self.dropoutproject(inputs)
        logits = self.project(output)

        prediction = tf.cast(tf.squeeze(tf.greater(logits, 0.5), axis=-1), tf.int32)

        return logits, prediction


def metrics(logits, label):
    labelexpand = tf.expand_dims(tf.cast(label, tf.int32), axis=1)

    loss = tf.reduce_mean(binary_crossentropy(labelexpand, logits))

    predict = tf.cast(tf.greater(logits, 0.5), tf.int32)

    accsum = tf.reduce_sum(tf.cast(tf.equal(predict, labelexpand), tf.float32))

    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labelexpand, 1), tf.equal(predict, 1)), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labelexpand, 1), tf.equal(predict, 0)), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labelexpand, 0), tf.equal(predict, 1)), tf.float32))

    return loss, accsum, tf.shape(label)[0], tp, tn, fp


@tf.function(experimental_relax_shapes=True)
def train_step(data, model, optimizer):
    with tf.GradientTape() as tape:
        logits, _ = model(data["sen"], training=True)
        loss, accsum, num, tp, tn, fp = metrics(logits, data["label"])

    trainable_variables = model.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, accsum, num, tp, tn, fp


@tf.function(experimental_relax_shapes=True)
def dev_step(data, model):
    logits, _ = model(data["sen"], training=False)
    loss, accsum, num, tp, tn, fp = metrics(logits, data["label"])

    return loss, accsum, num, tp, tn, fp


class TextClassify():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model(self, summary=True):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)

        mask, seqlen = Mask(name="mask")(sen)

        now, keepstay = Embeddings(name="embeddings")(inputs=(sen, seqlen))

        now = Encoder(layers=12, name="encoder")(inputs=(now, keepstay, mask))

        now = Pool(name="pooler")(now)

        logits, prediciton = Project()(now)

        model = Model(inputs=[sen], outputs=[logits, prediciton])

        if summary:
            model.summary()
            for tv in model.variables:
                print(tv.name, " : ", tv.shape)

        return model

    def train(self, train_file, val_file):

        model = self.build_model()

        if params_mode == "train0":
            load_model_weights_from_checkpoint_bert(model,
                                                    "pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt")
        else:
            model.load_weights(params_check + "/tta.h5")

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

    def predict(self, sentences):
        m_samples = len(sentences)

        sents2id = self.tokenizer(sentences, padding=True, return_tensors="tf")["input_ids"]

        model = self.build_model(summary=False)
        model.load_weights(params_check + "/textclassify.h5")

        prediction = model.predict(sents2id)[:, 0]
        for i in range(m_samples):
            print(sentences[i] + " ----> ", prediction[i])

    def test(self, test_file):
        model = self.build_model(summary=False)
        model.load_weights(params_check + "/textclassify.h5")

        test_batch = batched_data(test_file,
                                  single_example_parser_eb,
                                  params_batch_size,
                                  ([-1], []),
                                  shuffle=False)

        # sigmoid crossentropy loss
        lossobj = BinaryCrossentropy()

        model.compile(loss=lossobj,
                      metrics=["acc",
                               Precision(name="precision"),
                               Recall(name="recall"),
                               F1Score(name="F1", num_classes=2, threshold=0.5, average="micro"),
                               ])

        print("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n\n")

        loss, acc, p, r, F1 = model.evaluate(test_batch, verbose=0)

        print("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n\n")

        print("loss: %f acc: %f precision: %f recall: %f F1: %f" % (loss, acc, p, r, F1))

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
        'data/TFRecordFile/train_eb_char.tfrecord',
    ]
    val_file = [
        'data/TFRecordFile/val_eb_char.tfrecord',
    ]
    tc = TextClassify()

    if params_mode.startswith('train'):
        tc.train(train_file, val_file)

    elif params_mode == 'predict':
        sentences = [
            "这台电视不错！",
            "给服务赞一个",
            "不怎么样",
            "太差劲了",
            "非常棒！！！"
        ]
        tc.predict(sentences)

    elif params_mode == "plot":
        tc.plot()

    elif params_mode == "test":
        tc.test(train_file)


if __name__ == "__main__":
    main()
