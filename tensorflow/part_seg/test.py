import open3d as o3d
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import json
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
import part_seg_model as model
import affordances_loader
import kitchen_dat_subset_loader
import affordance_net
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='train_results/trained_models_affordances/epoch_200.ckpt', help='Model checkpoint path')
parser.add_argument('--dataset', type=str, default="affordances", help='The number of GPUs to use [default: 2]')
parser.add_argument('--num_categories', type=int, default=17, help='Epoch to run [default: 50]')
parser.add_argument('--num_part_cats', type=int, default=8, help='Epoch to run [default: 50]')

FLAGS = parser.parse_args()

# DEFAULT SETTINGS
pretrained_model_path = FLAGS.model_path 
# hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
ply_data_dir = os.path.join(BASE_DIR, './PartAnnotation')
gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_results')
output_verbose = False  
should_save = False

# MAIN SCRIPT
point_num = 2048            
batch_size = 8

# test_file_list = os.path.join(BASE_DIR, 'testing_ply_file_list.txt')

# oid2cpid = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r')) 

# object2setofoid = {}
# for idx in range(len(oid2cpid)):
#   objid, pid = oid2cpid[idx]
#   if not objid in object2setofoid.keys():
#     object2setofoid[objid] = []
#   object2setofoid[objid].append(idx)

# all_obj_cat_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
# fin = open(all_obj_cat_file, 'r')
# lines = [line.rstrip() for line in fin.readlines()]
# objcats = [line.split()[1] for line in lines] 
# objnames = [line.split()[0] for line in lines] 
# on2oid = {objcats[i]:i for i in range(len(objcats))} 
# fin.close()
all_obj_cats = [('bowl', 0), ('cup', 1), ('hammer', 2), ('knife', 3), ('ladle', 4), ('mallet', 5), ('mug', 6), ('pot', 7), ('saw', 8), ('scissors', 9), ('scoop', 10), ('shears', 11), ('shovel', 12), ('spoon', 13), ('tenderizer', 14), ('trowel', 15), ('turner', 16)]


# color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
# color_map = json.load(open(color_map_file, 'r'))


entropy_neighbors=20

NUM_CATEGORIES = FLAGS.num_categories
NUM_PART_CATS = FLAGS.num_part_cats

# cpid2oid = json.load(open(os.path.join(hdf5_data_dir, 'catid_partid_to_overallid.json'), 'r'))

def get_entropy(pred_choice,ground_truth,points,cls_label,total_entropy_per_cat,num_points,entropy_neighbors):

    ground_truth = ground_truth.numpy()
    points = points.numpy()


    def get_neigborhood_with_n_points(pcd, pcd_tree, poi, points=20):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[poi], 20)
        #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        return k, idx

    def get_neigborhood_in_radius(pcd, pcd_tree, poi, radius=0.005):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[poi], radius)
        #np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
        return k, idx

    pcd = o3d.geometry.PointCloud()
    pcd.points= o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    entropy_predicted = []
    entropy_gt = []
    for poi in range(len(pcd.points)):
        _, neigborhood_idx = get_neigborhood_with_n_points(pcd, pcd_tree, poi, entropy_neighbors)
        entropy_predicted.append(np.sum(pred_choice.take(neigborhood_idx) != pred_choice[poi]))
        entropy_gt.append(np.sum(ground_truth.take(neigborhood_idx) != ground_truth[poi]))

    entropy = np.sum(entropy_predicted)/(num_points*entropy_neighbors)*100
    total_entropy_per_cat[cls_label] += entropy
    return entropy, entropy_predicted, entropy_gt

def printout(flog, data):
  print(data)
  flog.write(data + '\n')

def output_color_point_cloud(data, seg, out_file):
  with open(out_file, 'w') as f:
    l = len(seg)
    for i in range(l):
      color = color_map[seg[i]]
      f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud_red_blue(data, seg, out_file):
  with open(out_file, 'w') as f:
    l = len(seg)
    for i in range(l):
      if seg[i] == 1:
        color = [0, 0, 1]
      elif seg[i] == 0:
        color = [1, 0, 0]
      else:
        color = [0, 0, 0]

      f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def pc_normalize(pc):
  l = pc.shape[0]
  centroid = np.mean(pc, axis=0)
  pc = pc - centroid
  m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
  pc = pc / m
  return pc

def placeholder_inputs():
  pointclouds_ph = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, point_num, 3))
  input_label_ph = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))
  return pointclouds_ph, input_label_ph

def output_color_point_cloud(data, seg, out_file):
  with open(out_file, 'w') as f:
    l = len(seg)
    for i in range(l):
      color = color_map[seg[i]]
      f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def load_pts_seg_files(pts_file, seg_file, catid):
  with open(pts_file, 'r') as f:
    pts_str = [item.rstrip() for item in f.readlines()]
    pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
  with open(seg_file, 'r') as f:
    part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
    seg = np.array([cpid2oid[catid+'_'+str(x)] for x in part_ids])
  return pts, seg

def pc_augment_to_point_num(pts, pn):
  assert(pts.shape[0] <= pn)
  cur_len = pts.shape[0]
  res = np.array(pts)
  while cur_len < pn:
    res = np.concatenate((res, pts))
    cur_len += pts.shape[0]
  return res[:pn, :]

def convert_label_to_one_hot(labels):
  label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
  for idx in range(labels.shape[0]):
    label_one_hot[idx, labels[idx]] = 1
  return label_one_hot

def predict():
  is_training = False
  
  with tf.device('/gpu:'+str(gpu_to_use)):
    pointclouds_ph, input_label_ph = placeholder_inputs()
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=())

    seg_pred = model.get_model(pointclouds_ph, input_label_ph, \
        cat_num=NUM_CATEGORIES, part_num=NUM_PART_CATS, is_training=is_training_ph, \
        batch_size=batch_size, num_point=point_num, weight_decay=0.0, bn_decay=None)
    
  saver = tf.compat.v1.train.Saver()

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  with tf.compat.v1.Session(config=config) as sess:
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    flog = open(os.path.join(output_dir, 'log.txt'), 'a')

    printout(flog, 'Loading model %s' % pretrained_model_path)
    saver.restore(sess, pretrained_model_path)
    printout(flog, 'Model restored.')
    
    # batch_data = np.zeros([batch_size, point_num, 3]).astype(np.float32)

    # total_acc = 0.0
    # total_seen = 0
    # total_acc_iou = 0.0

    # total_per_cat_acc = np.zeros((NUM_OBJ_CATS)).astype(np.float32)
    # total_per_cat_iou = np.zeros((NUM_OBJ_CATS)).astype(np.float32)
    # total_per_cat_seen = np.zeros((NUM_OBJ_CATS)).astype(np.int32)

    # ffiles = open(test_file_list, 'r')
    # lines = [line.rstrip() for line in ffiles.readlines()]
    # pts_files = [line.split()[0] for line in lines]
    # seg_files = [line.split()[1] for line in lines]
    # labels = [line.split()[2] for line in lines]
    # ffiles.close()

    # len_pts_files = len(pts_files)


    if FLAGS.dataset == "affordances":

      test_dataset = affordances_loader.PartDataset(classification=False, npoints=point_num, split='test')
      test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    elif FLAGS.dataset == "kitchen":
      test_dataset = kitchen_dat_subset_loader.PartDataset(classification=False, npoints=point_num, split='test')
      test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif FLAGS.dataset == "affordance_net":
      test_dataset = affordance_net.AffordNetDataset(split='test')
      test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

      all_classes = affordance_net.all_classes
      all_obj_cats= list(all_classes.items())

    is_training = False

    total_loss = 0.0
    total_seg_acc = 0.0
    total_seen = 0
    total_seg_acc_seen = 0
    total_acc_iou = 0.0
    entropy = []


    total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
    total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)
    total_per_cat_iou = np.zeros(NUM_CATEGORIES).astype(np.float32)
    total_entropy_per_cat = np.zeros(NUM_CATEGORIES).astype(np.float32)

    all_objects = []

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
        pointclouds_ph: points, 
        input_label_ph: cls_label_one_hot, 
        #seg_phs[0]: part_label,
        is_training_ph: is_training, 
      }

      pred_seg_res = sess.run(seg_pred, \
            feed_dict=feed_dict)
      
      
      #pred_seg_res = pred_seg_res[0, ...]

      pred_seg_res = np.argmax(pred_seg_res, axis=2)


      if should_save:
        for batch_id in range(points.shape[0]):
          pred_points = {}

          pred_points['cls'] =  int(cls_label_[batch_id])
          pred_points['points'] = []

          for i in range(points.shape[1]):
            point = {}
            point['x'] = float(points[batch_id][i][0])
            point['y'] = float(points[batch_id][i][1])
            point['z'] = float(points[batch_id][i][2])


            point['label'] = int(part_label[batch_id][i])
            point['pred'] = int(pred_seg_res[batch_id][i])

            pred_points['points'].append(point)

          all_objects.append(pred_points)
          

      # print(pred_seg_res.shape, part_label.shape)

      part_label_per_class = part_label.numpy()
      per_instance_part_acc = np.mean(pred_seg_res == part_label_per_class, axis=1)
      average_part_acc = np.mean(per_instance_part_acc)

      total_seg_acc_seen += 1
      
      total_seg_acc += average_part_acc

      cls_label_ = cls_label_.numpy()

      for shape_idx in range(cls_label_.shape[0]):
        total_seen_per_cat[cls_label_[shape_idx]] += 1
        total_seg_acc_per_cat[cls_label_[shape_idx]] += per_instance_part_acc[shape_idx]

      for shape_idx in range (part_label_per_class.shape[0]):
        mask = np.int32(pred_seg_res[shape_idx] == part_label_per_class[shape_idx])
        total_iou = 0.0
        for part_id in range(NUM_PART_CATS): #normally all part classes within this specific subclass
            n_pred = np.sum(pred_seg_res[shape_idx] == part_id)
            n_gt = np.sum(part_label_per_class[shape_idx] == part_id)
            n_intersect = np.sum(np.int32(part_label_per_class[shape_idx] == part_id) * mask)
            n_union = n_pred + n_gt - n_intersect
            if n_union == 0:
                total_iou += 1
            else:
                total_iou += n_intersect * 1.0 / n_union
        avg_iou = total_iou / NUM_PART_CATS
        total_acc_iou += avg_iou
        total_seen += 1
        total_per_cat_iou[cls_label_[shape_idx]] += avg_iou    

      for point_set_no in range (cls_label_.shape[0]):
        e, _, _ = get_entropy(pred_seg_res[point_set_no],part_label[point_set_no],points[point_set_no,],cls_label_[point_set_no],total_entropy_per_cat,point_num,entropy_neighbors)
        entropy.append(e)

    total_seg_acc = total_seg_acc * 1.0 / total_seg_acc_seen


    #test_writer.add_summary(test_loss_sum, (epoch_num+1) * num_train_file-1)
    #test_writer.add_summary(test_seg_acc_sum, (epoch_num+1) * num_train_file-1)

    mean_IoU = total_acc_iou/total_seen

    print('\nAccuracy: %f IoU: %f Entropy: %f' %
          (total_seg_acc, mean_IoU, np.mean(entropy)))

    for cat_idx in range(NUM_CATEGORIES):
      if total_seen_per_cat[cat_idx] > 0:
        print('\n\tCategory %s:\n\t\tObject Number: %d' % (all_obj_cats[cat_idx][0], total_seen_per_cat[cat_idx]))
        print('\t\tSeg Accuracy: %f' % (total_seg_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))
        print('\t\tSeg IoU: %f' % (total_per_cat_iou[cat_idx]/total_seen_per_cat[cat_idx]))
        print('\t\tEntropy: %f' % (total_entropy_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))


    if should_save:
      with open('output.json', 'w') as f:
        json.dump(all_objects, f)

    # for shape_idx in range(len_pts_files):
    #   if shape_idx % 100 == 0:
    #     printout(flog, '%d/%d ...' % (shape_idx, len_pts_files))

    #   cur_gt_label = on2oid[labels[shape_idx]] # 0/1/.../15

    #   cur_label_one_hot = np.zeros((1, NUM_OBJ_CATS), dtype=np.float32)
    #   cur_label_one_hot[0, cur_gt_label] = 1

    #   pts_file_to_load = os.path.join(ply_data_dir, pts_files[shape_idx])
    #   seg_file_to_load = os.path.join(ply_data_dir, seg_files[shape_idx])

    #   pts, seg = load_pts_seg_files(pts_file_to_load, seg_file_to_load, objcats[cur_gt_label])
    #   ori_point_num = len(seg)

    #   batch_data[0, ...] = pc_augment_to_point_num(pc_normalize(pts), point_num)

    #   seg_pred_res = sess.run(seg_pred, feed_dict={
    #         pointclouds_ph: batch_data,
    #         input_label_ph: cur_label_one_hot, 
    #         is_training_ph: is_training})

    #   seg_pred_res = seg_pred_res[0, ...]

    #   iou_oids = object2setofoid[objcats[cur_gt_label]]
    #   non_cat_labels = list(set(np.arange(NUM_PART_CATS)).difference(set(iou_oids)))

    #   mini = np.min(seg_pred_res)
    #   seg_pred_res[:, non_cat_labels] = mini - 1000

    #   seg_pred_val = np.argmax(seg_pred_res, axis=1)[:ori_point_num]

    #   seg_acc = np.mean(seg_pred_val == seg)

    #   total_acc += seg_acc
    #   total_seen += 1

    #   total_per_cat_seen[cur_gt_label] += 1
    #   total_per_cat_acc[cur_gt_label] += seg_acc

    #   mask = np.int32(seg_pred_val == seg)

    #   total_iou = 0.0
    #   iou_log = ''
    #   for oid in iou_oids:
    #     n_pred = np.sum(seg_pred_val == oid)
    #     n_gt = np.sum(seg == oid)
    #     n_intersect = np.sum(np.int32(seg == oid) * mask)
    #     n_union = n_pred + n_gt - n_intersect
    #     iou_log += '_' + str(n_pred)+'_'+str(n_gt)+'_'+str(n_intersect)+'_'+str(n_union)+'_'
    #     if n_union == 0:
    #       total_iou += 1
    #       iou_log += '_1\n'
    #     else:
    #       total_iou += n_intersect * 1.0 / n_union
    #       iou_log += '_'+str(n_intersect * 1.0 / n_union)+'\n'

    #   avg_iou = total_iou / len(iou_oids)
    #   total_acc_iou += avg_iou
    #   total_per_cat_iou[cur_gt_label] += avg_iou
      
    #   if output_verbose:
    #     output_color_point_cloud(pts, seg, os.path.join(output_dir, str(shape_idx)+'_gt.obj'))
    #     output_color_point_cloud(pts, seg_pred_val, os.path.join(output_dir, str(shape_idx)+'_pred.obj'))
    #     output_color_point_cloud_red_blue(pts, np.int32(seg == seg_pred_val), 
    #         os.path.join(output_dir, str(shape_idx)+'_diff.obj'))

    #     with open(os.path.join(output_dir, str(shape_idx)+'.log'), 'w') as fout:
    #       fout.write('Total Point: %d\n\n' % ori_point_num)
    #       fout.write('Ground Truth: %s\n' % objnames[cur_gt_label])
    #       fout.write('Accuracy: %f\n' % seg_acc)
    #       fout.write('IoU: %f\n\n' % avg_iou)
    #       fout.write('IoU details: %s\n' % iou_log)

    # printout(flog, 'Accuracy: %f' % (total_acc / total_seen))
    # printout(flog, 'IoU: %f' % (total_acc_iou / total_seen))

    # for cat_idx in range(NUM_OBJ_CATS):
    #   printout(flog, '\t ' + objcats[cat_idx] + ' Total Number: ' + str(total_per_cat_seen[cat_idx]))
    #   if total_per_cat_seen[cat_idx] > 0:
    #     printout(flog, '\t ' + objcats[cat_idx] + ' Accuracy: ' + \
    #         str(total_per_cat_acc[cat_idx] / total_per_cat_seen[cat_idx]))
    #     printout(flog, '\t ' + objcats[cat_idx] + ' IoU: '+ \
    #         str(total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]))
    
with tf.Graph().as_default():
  predict()
