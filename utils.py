import tensorflow as tf
import collections
import re
import numpy as np


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def convert2Uni(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        print(type(text))
        print('####################wrong################')


def load_vocab(vocab_file):  # 获取BERT字表方法
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            tmp = reader.readline()
            if not tmp:
                break
            token = convert2Uni(tmp)
            token = token.rstrip("\n")
            vocab[token] = index
            index += 1
    return vocab


def single_example_parser_eb(serialized_example):
    context_features = {
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    sequence_features = {
        "sen": tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    label = context_parsed['label']

    return {"sen": sen, "label": label}


def single_example_parser_eb1(serialized_example):
    context_features = {
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    sequence_features = {
        "sen": tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    label = context_parsed['label']

    return sen, label


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, shuffle=True, buffer_size=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset


class replace():
    def __init__(self):
        rep = {
            '“': '"',
            '”': '"',
            ' ': ''
        }
        self.rep = dict((re.escape(k), v) for k, v in rep.items())
        self.pattern = re.compile("|".join(self.rep.keys()))

    def replace(self, words):
        return self.pattern.sub(lambda m: self.rep[re.escape(m.group(0))], words)


def softmax(a, mask):
    """
    :param a: B*ML1*ML2
    :param mask: B*ML1*ML2
    """
    return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)


def focal_loss(y_true, y_pred, gamma=15.0):
    """
    Focal Loss 针对样本不均衡
    :param y_true: 样本标签
    :param y_pred: 预测值（sigmoid）
    :return: focal loss
    """

    alpha = 0.5
    loss = tf.where(tf.equal(y_true, 1),
                    -alpha * (1.0 - y_pred) ** gamma * tf.math.log(y_pred),
                    -(1.0 - alpha) * y_pred ** gamma * tf.math.log(1.0 - y_pred))

    return tf.reduce_sum(loss)


def focal_loss_new(y_true, y_pred, gamma=2.0):
    """
    Focal Loss 针对样本不均衡以及带有Rdrop的情形
    :param y_true: 样本标签
    :param y_pred: 预测值（sigmoid）
    :return: focal loss
    """

    alpha = 0.5
    loss = tf.where(tf.equal(y_true, 1),
                    -alpha * (1.0 - y_pred) ** gamma * tf.math.log(y_pred),
                    -(1.0 - alpha) * y_pred ** gamma * tf.math.log(1.0 - y_pred))

    return tf.reduce_sum(loss)


def bce_loss_weight(y_true, y_pred):
    """
    bce Loss 针对样本不均衡，给出样本权重
    :param y_true: 样本标签
    :param y_pred: 预测值（sigmoid）
    :return: bce loss
    """
    class_weight = np.array([200., 2253.]) / (2253. + 200.)
    loss = tf.where(tf.equal(y_true, 1),
                    -class_weight[1] * tf.math.log(y_pred),
                    -class_weight[0] * tf.math.log(1.0 - y_pred))

    return tf.reduce_sum(loss)
