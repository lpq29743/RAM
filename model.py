import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
import time
from utils import get_batch_index

class RAM(object):

    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.n_hop = config.n_hop
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout
        
        self.word2id = config.word2id
        self.max_sentence_len = config.max_sentence_len
        self.max_aspect_len = config.max_aspect_len
        self.word2vec = config.word2vec
        self.sess = sess
        
        self.timestamp = str(int(time.time()))

    def build_model(self):
        with tf.name_scope('inputs'):
            self.sentences = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len])
            self.sentence_lens = tf.placeholder(tf.int32, None)
            self.sentence_locs = tf.placeholder(tf.float32, [None, self.max_sentence_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.dropout_keep_prob = tf.placeholder(tf.float32)
            
            inputs = tf.nn.embedding_lookup(self.word2vec, self.sentences)
            inputs = tf.cast(inputs, tf.float32)
            inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)
            aspect_inputs = tf.nn.embedding_lookup(self.word2vec, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.reduce_mean(aspect_inputs, 1)

        with tf.name_scope('weights'):
            weights = {
                'attention': tf.get_variable(
                    name='W_al',
                    shape=[self.n_hop, 1, self.n_hidden * 3 + self.embedding_dim + 1],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'gru_r': tf.get_variable(
                    name='W_r',
                    shape=[self.n_hidden, self.n_hidden * 2 + 1],
                    initializer=tf.orthogonal_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'gru_z': tf.get_variable(
                    name='W_z',
                    shape=[self.n_hidden, self.n_hidden * 2 + 1],
                    initializer=tf.orthogonal_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'gru_g': tf.get_variable(
                    name='W_g',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.orthogonal_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'gru_x': tf.get_variable(
                    name='W_x',
                    shape=[self.n_hidden, self.n_hidden * 2 + 1],
                    initializer=tf.orthogonal_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_hidden, self.n_class],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }
        
        with tf.name_scope('biases'):
            biases = {
                'attention': tf.get_variable(
                    name='B_al',
                    shape=[self.n_hop, 1, self.max_sentence_len],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }
            
        with tf.name_scope('updates'):
            updates = {
                'gru_r': tf.get_variable(
                    name='U_r',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.orthogonal_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'gru_z': tf.get_variable(
                    name='U_z',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.orthogonal_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('dynamic_rnn'):
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(
                self.n_hidden,
                initializer=tf.orthogonal_initializer(),
            )
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(
                self.n_hidden,
                initializer=tf.orthogonal_initializer(),
            )
            outputs, state, _ = tf.nn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                tf.unstack(tf.transpose(inputs, perm=[1, 0, 2])),
                sequence_length=self.sentence_lens,
                dtype=tf.float32,
                scope='BiLSTM'
            )
            outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.max_sentence_len, self.n_hidden * 2])
            batch_size = tf.shape(outputs)[0]

            outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            outputs_iter = outputs_iter.unstack(outputs)
            sentence_locs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            sentence_locs_iter = sentence_locs_iter.unstack(self.sentence_locs)
            sentence_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            sentence_lens_iter = sentence_lens_iter.unstack(self.sentence_lens)
            memory = tf.TensorArray(size=batch_size, dtype=tf.float32)
            def body(i, memory):
                a = outputs_iter.read(i)
                b = sentence_locs_iter.read(i)
                c = sentence_lens_iter.read(i)
                weight = 1 - b
                memory = memory.write(i, tf.concat([tf.multiply(a, tf.tile(tf.expand_dims(weight, -1), [1, self.n_hidden * 2])), tf.reshape(b, [-1, 1])], 1))
                return (i + 1, memory)
            def condition(i, memory):
                return i < batch_size
            _, memory_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, memory))
            self.memories = tf.reshape(memory_final.stack(), [-1, self.max_sentence_len, self.n_hidden * 2 + 1])

            e = tf.zeros([batch_size, self.n_hidden])
            scores_list = []
            aspect_inputs = tf.tile(tf.expand_dims(aspect_inputs, 1), [1, self.max_sentence_len, 1])
            for h in range(self.n_hop):
                memories_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
                memories_iter = memories_iter.unstack(self.memories)
                e_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
                e_iter = e_iter.unstack(e)
                aspect_inputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
                aspect_inputs_iter = aspect_inputs_iter.unstack(aspect_inputs)
                sentence_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
                sentence_lens_iter = sentence_lens_iter.unstack(self.sentence_lens)
                newe = tf.TensorArray(size=batch_size, dtype=tf.float32)
                score = tf.TensorArray(size=batch_size, dtype=tf.float32)
                def body(i, newe, score):
                    a = memories_iter.read(i)
                    olde = e_iter.read(i)
                    b = tf.tile(tf.expand_dims(olde, 0), [self.max_sentence_len, 1])
                    c = aspect_inputs_iter.read(i)
                    l = math_ops.to_int32(sentence_lens_iter.read(i))
                    g = tf.matmul(weights['attention'][h], tf.transpose(tf.concat([a, b, c], 1), perm=[1, 0])) + biases['attention'][h]
                    score_temp = tf.concat([tf.nn.softmax(tf.slice(g, [0, 0], [1, l])), tf.zeros([1, self.max_sentence_len - l])], 1)
                    score = score.write(i, score_temp)
                    i_AL = tf.reshape(tf.matmul(score_temp, a), [-1, 1])
                    olde = tf.reshape(olde, [-1, 1])
                    r = tf.nn.sigmoid(tf.matmul(weights['gru_r'], i_AL) + tf.matmul(updates['gru_r'], olde))
                    z = tf.nn.sigmoid(tf.matmul(weights['gru_z'], i_AL) + tf.matmul(updates['gru_z'], olde))
                    e0 = tf.nn.tanh(tf.matmul(weights['gru_x'], i_AL) + tf.matmul(weights['gru_g'], tf.multiply(r, olde)))
                    newe_temp = tf.multiply(1 - z, olde) + tf.multiply(z, e0)
                    newe = newe.write(i, newe_temp)
                    return (i + 1, newe, score)
                def condition(i, newe, score):
                    return i < batch_size
                _, newe_final, score_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, newe, score))
                e = tf.reshape(newe_final.stack(), [-1, self.n_hidden])
                batch_score = tf.reshape(score_final.stack(), [-1, self.max_sentence_len])
                scores_list.append(batch_score)
            self.scores = tf.transpose(tf.reshape(tf.stack(scores_list), [self.n_hop, -1, self.max_sentence_len]), [1, 0, 2])
            self.predict = tf.matmul(e, weights['softmax']) + biases['softmax']

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.predict, labels = self.labels))
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            self.predict_label = tf.argmax(self.predict, 1)
            self.correct_pred = tf.equal(self.predict_label, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))
            
        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        _dir = 'logs/' + str(self.timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)

    def train(self, data):
        sentences, aspects, sentence_lens, sentence_locs, labels = data
        cost, cnt = 0., 0

        for sample, num in self.get_batch_data(sentences, aspects, sentence_lens, sentence_locs, labels, self.batch_size, True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.cost, self.global_step, self.train_summary_op], feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.test(data)
        return cost / cnt, train_acc

    def test(self, data):
        sentences, aspects, sentence_lens, sentence_locs, labels = data
        cost, acc, cnt = 0., 0, 0

        for sample, num in self.get_batch_data(sentences, aspects, sentence_lens, sentence_locs, labels, int(len(sentences) / 2) + 1, False, 1.0):
            loss, accuracy, step, summary = self.sess.run([self.cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num

        self.test_summary_writer.add_summary(summary, step)
        return cost / cnt, acc / cnt

    def analysis(self, data, tag):
        sentences, aspects, sentence_lens, sentence_locs, labels = data
        with open('analysis/' + tag + '_' + str(self.timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(sentences, aspects, sentence_lens, sentence_locs, labels, int(len(sentences) / 2) + 1, False, 1.0):
                scores, correct_pred, predict_label = self.sess.run([self.scores, self.correct_pred, self.predict_label], feed_dict=sample)
                for a, b, c in zip(scores, correct_pred, predict_label):
                    for i in a:
                        i = str(i).replace('\n', '')
                        f.write('%s\n' % i)
                    b = str(b).replace('\n', '')
                    c = str(c).replace('\n', '')
                    f.write('%s\n%s\n' % (b, c))
        print('Finishing analyzing %s data' % tag)
    
    def run(self, train_data, test_data):
        saver = tf.train.Saver(tf.trainable_variables())
        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        max_acc, step = 0., -1
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, 'models/model_iter' + str(self.timestamp), global_step=i)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; test-loss=%.6f; test-acc=%.6f;' % (str(i), train_loss, train_acc, test_loss, test_acc))
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))
        print('Analyzing ...')
        saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
        self.analysis(train_data, 'train')
        self.analysis(test_data, 'test')

    def get_batch_data(self, sentences, aspects, sentence_lens, sentence_locs, labels, batch_size, is_shuffle, keep_prob):
        for index in get_batch_index(len(sentences), batch_size, is_shuffle):
            feed_dict = {
                self.sentences: sentences[index],
                self.aspects: aspects[index],
                self.sentence_lens: sentence_lens[index],
                self.sentence_locs: sentence_locs[index],
                self.labels: labels[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)
