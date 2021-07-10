import argparse
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import torch
import numpy as np
from datetime import datetime
import json
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import part_seg_model as model
import affordances_loader


TOWER_NAME = 'tower'

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='The number of GPUs to use [default: 2]')
parser.add_argument('--batch', type=int, default=16, help='Batch Size per GPU during training [default: 32]')
parser.add_argument('--epoch', type=int, default=201, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()


# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

all_obj_cats = [('Bowl', 0), ('Cup', 1), ('Hammer', 2), ('Knife', 3), ('Ladle', 4), ('Mallet', 5), ('Mug', 6), ('Pot', 7), ('Saw', 8), ('Scissors', 9), ('Scoop', 10), ('Shears', 11), ('Shovel', 12), ('Spoon', 13), ('Tenderizer', 14), ('Trowel', 15), ('Turner', 16)]

NUM_CATEGORIES = 17
NUM_PART_CATS = 8

print('#### Batch Size Per GPU: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Using GPUs: {0}'.format(FLAGS.num_gpu))

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.003
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
  os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
  os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
  os.mkdir(SUMMARIES_FOLDER)

def printout(flog, data):
  print(data)
  flog.write(data + '\n')

def convert_label_to_one_hot(labels):
  label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
  for idx in range(labels.shape[0]):
    label_one_hot[idx, labels[idx]] = 1
  return label_one_hot

def average_gradients(tower_grads):
  """Calculate average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been 
     averaged across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is None:
        continue
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    batch = tf.Variable(0, trainable=False)
    
    learning_rate = tf.compat.v1.train.exponential_decay(
            BASE_LEARNING_RATE,     # base learning rate
            batch * batch_size,     # global_var indicating the number of steps
            DECAY_STEP,             # step size
            DECAY_RATE,             # decay rate
            staircase=True          # Stair-case or continuous decreasing
            )
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)
  
    bn_momentum = tf.compat.v1.train.exponential_decay(
          BN_INIT_DECAY,
          batch*batch_size,
          BN_DECAY_DECAY_STEP,
          BN_DECAY_DECAY_RATE,
          staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

    lr_op = tf.summary.scalar('learning_rate', learning_rate)
    batch_op = tf.summary.scalar('batch_number', batch)
    bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)

    trainer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    # store tensors for different gpus
    tower_grads = []
    pointclouds_phs = []
    input_label_phs = []
    seg_phs =[]
    is_training_phs =[]

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
      for i in range(FLAGS.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            pointclouds_phs.append(tf.compat.v1.placeholder(tf.float32, shape=(batch_size, point_num, 3))) # for points
            input_label_phs.append(tf.compat.v1.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))) # for one-hot category label
            seg_phs.append(tf.compat.v1.placeholder(tf.int32, shape=(batch_size, point_num))) # for part labels
            is_training_phs.append(tf.compat.v1.placeholder(tf.bool, shape=()))

            seg_pred = model.get_model(pointclouds_phs[-1], input_label_phs[-1], \
                is_training=is_training_phs[-1], bn_decay=bn_decay, cat_num=NUM_CATEGORIES, \
                part_num=NUM_PART_CATS, batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)


            loss, per_instance_seg_loss, per_instance_seg_pred_res  \
              = model.get_loss(seg_pred, seg_phs[-1])

            total_training_loss_ph = tf.compat.v1.placeholder(tf.float32, shape=())
            total_testing_loss_ph = tf.compat.v1.placeholder(tf.float32, shape=())

            seg_training_acc_ph = tf.compat.v1.placeholder(tf.float32, shape=())
            seg_testing_acc_ph = tf.compat.v1.placeholder(tf.float32, shape=())
            seg_testing_acc_avg_cat_ph = tf.compat.v1.placeholder(tf.float32, shape=())

            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

        
            seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', seg_training_acc_ph)
            seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', seg_testing_acc_ph)
            seg_test_acc_avg_cat_op = tf.summary.scalar('seg_testing_acc_avg_cat', seg_testing_acc_avg_cat_ph)

            tf.compat.v1.get_variable_scope().reuse_variables()

            grads = trainer.compute_gradients(loss)

            tower_grads.append(grads)

    grads = average_gradients(tower_grads)

    train_op = trainer.apply_gradients(grads, global_step=batch)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), sharded=True, max_to_keep=20)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)
    
    init = tf.group(tf.compat.v1.global_variables_initializer(),
             tf.compat.v1.local_variables_initializer())
    sess.run(init)

    train_writer = tf.compat.v1.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(SUMMARIES_FOLDER + '/test')

    fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
    fcmd.write(str(FLAGS))
    fcmd.close()

    # write logs to the disk
    flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')


    train_dataset = affordances_loader.PartDataset(classification=False, npoints=point_num, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = affordances_loader.PartDataset(classification=False, npoints=point_num, split='val')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def train_one_epoch( epoch_num):
      is_training = True

      total_loss = 0.0
      total_seg_acc = 0.0

      for batch_id, data in enumerate(train_dataloader):
        points, part_label, cls_label_ = data
        if(points.size(0)<batch_size):
          break

        # points = tf.convert_to_tensor(points)
        # part_label = tf.convert_to_tensor(part_label)
        # cls_label_ = tf.convert_to_tensor(cls_label_)
        # print(cls_label_.shape)
        cls_label_one_hot = convert_label_to_one_hot(cls_label_)


        feed_dict = {
          # For the first gpu
          pointclouds_phs[0]: points, 
          input_label_phs[0]: cls_label_one_hot, 
          seg_phs[0]: part_label,
          is_training_phs[0]: is_training, 
        }

        # train_op is for both gpus, and the others are for gpu_1
        _, loss_val, per_instance_seg_loss_val, seg_pred_val, pred_seg_res \
            = sess.run([train_op, loss, per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
            feed_dict=feed_dict)

        per_instance_part_acc = np.mean(pred_seg_res == part_label.numpy(), axis=1)
        average_part_acc = np.mean(per_instance_part_acc)

        total_loss += loss_val
        total_seg_acc += average_part_acc
              
      total_loss = total_loss * 1.0 / len(train_dataloader)
      total_seg_acc = total_seg_acc * 1.0 / len(train_dataloader)

      lr_sum, bn_decay_sum, batch_sum, train_loss_sum, train_seg_acc_sum = sess.run(\
          [lr_op, bn_decay_op, batch_op, total_train_loss_sum_op, seg_train_acc_sum_op], \
          feed_dict={total_training_loss_ph: total_loss, seg_training_acc_ph: total_seg_acc})

      # train_writer.add_summary(train_loss_sum, i + epoch_num * num_train_file)
      # train_writer.add_summary(lr_sum, i + epoch_num * num_train_file)
      # train_writer.add_summary(bn_decay_sum, i + epoch_num * num_train_file)
      # train_writer.add_summary(train_seg_acc_sum, i + epoch_num * num_train_file)
      # train_writer.add_summary(batch_sum, i + epoch_num * num_train_file)

      printout(flog, '\tTraining Total Mean_loss: %f' % total_loss)
      printout(flog, '\t\tTraining Seg Accuracy: %f' % total_seg_acc)

    def eval_one_epoch(epoch_num):
      is_training = False

      total_loss = 0.0
      total_seg_acc = 0.0
      total_seen = 0

      total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
      total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)

      for batch_id, data in enumerate(test_dataloader):
        points, part_label, cls_label_ = data
        if(points.size(0)<batch_size):
          break

        # points = tf.convert_to_tensor(points)
        # part_label = tf.convert_to_tensor(part_label)
        # cls_label_ = tf.convert_to_tensor(cls_label_)
        # print(cls_label_.shape)
        cls_label_one_hot = convert_label_to_one_hot(cls_label_)


        feed_dict = {
          # For the first gpu
          pointclouds_phs[0]: points, 
          input_label_phs[0]: cls_label_one_hot, 
          seg_phs[0]: part_label,
          is_training_phs[0]: is_training, 
        }

        loss_val, per_instance_seg_loss_val, seg_pred_val, pred_seg_res \
              = sess.run([loss, per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
              feed_dict=feed_dict)

        per_instance_part_acc = np.mean(pred_seg_res == part_label.numpy(), axis=1)
        average_part_acc = np.mean(per_instance_part_acc)

        total_seen += 1
        total_loss += loss_val
        
        total_seg_acc += average_part_acc

        cls_label_ = cls_label_.numpy()

        for shape_idx in range(cls_label_.shape[0]):
          total_seen_per_cat[cls_label_[shape_idx]] += 1
          total_seg_acc_per_cat[cls_label_[shape_idx]] += per_instance_part_acc[shape_idx]

      total_loss = total_loss * 1.0 / total_seen
      total_seg_acc = total_seg_acc * 1.0 / total_seen

      test_loss_sum, test_seg_acc_sum = sess.run(\
          [total_test_loss_sum_op, seg_test_acc_sum_op], \
          feed_dict={total_testing_loss_ph: total_loss, \
          seg_testing_acc_ph: total_seg_acc})

      #test_writer.add_summary(test_loss_sum, (epoch_num+1) * num_train_file-1)
      #test_writer.add_summary(test_seg_acc_sum, (epoch_num+1) * num_train_file-1)

      printout(flog, '\tTesting Total Mean_loss: %f' % total_loss)
      printout(flog, '\t\tTesting Seg Accuracy: %f' % total_seg_acc)

      for cat_idx in range(NUM_CATEGORIES):
        if total_seen_per_cat[cat_idx] > 0:
          printout(flog, '\n\t\tCategory %s Object Number: %d' % (all_obj_cats[cat_idx][0], total_seen_per_cat[cat_idx]))
          printout(flog, '\t\tCategory %s Seg Accuracy: %f' % (all_obj_cats[cat_idx][0], total_seg_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))

    if not os.path.exists(MODEL_STORAGE_PATH):
      os.mkdir(MODEL_STORAGE_PATH)

    for epoch in range(TRAINING_EPOCHES):
      printout(flog, '\n<<< Testing on the test dataset ...')
      eval_one_epoch(epoch)

      printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

      train_one_epoch(epoch)

      if epoch % 5 == 0:
        cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch)+'.ckpt'))
        printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

      flog.flush()

    flog.close()

if __name__=='__main__':
  train()
