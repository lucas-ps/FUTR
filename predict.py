import lmdb
import torch
import torch.nn as nn
import numpy
import pdb
import os
import copy
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from utils import normalize_duration, eval_file_more

def load_lmdb_features(args, vid_file):
    """ Loads TSN features from LMDB dataset for a specific video """
    features = []
    i = 1
    if args.dataset == "50salads":
        dataset_path = "/media/lucas/Linux SSD/rulstm/RULSTM/data/50-salads/rgb/"
    elif args.dataset == "breakfast":
        dataset_path = "/media/lucas/Linux SSD/rulstm/RULSTM/data/Breakfast1/rgb/"
    else:
        dataset_path = "/media/lucas/Linux SSD/rulstm/RULSTM/data/ek55/rgb/"

    lmdb_env = lmdb.open(dataset_path, readonly=True, lock=False)

    # Ground truths provided are different to file names for stereo vodeos, not sure why, this code fixes the filenames
    if '_stereo01_' in vid_file:
        old_name = vid_file
        vid_file = vid_file.replace('_stereo01_', '_').replace('.txt', '_ch1.txt')
        parts = vid_file.split("_")
        parts.pop(1)
        vid_file = "_".join(parts)


    # For each frame key, load the corresponding feature representation from the LMDB dataset
    with lmdb_env.begin() as txn:
        while True:
            key = f"{vid_file.replace('.txt', '')}_frame_{i:010d}.jpg"
            # if self.args.dataset == "breakfast":
            #     keys = key.split("_")
            #     key = "_".join(keys[2:])
            value = txn.get(key.encode('utf-8'))
            #print(key)
            #print(value)
            if value is not None:
                frame_features = np.frombuffer(value, 'float32')
                features.append(frame_features[-1024:])
                #print (frame_features.shape)
            else:
                break
            i += 1
    features = np.array(features)
    features = features.transpose()
    #print(vid_file)
    if features.shape == (0,):
        vid_file = vid_file.replace('ch1', 'ch0')
        features = load_lmdb_features(args, vid_file)
    #print(features.shape)
    return features


def predict(model, vid_list, args, obs_p, n_class, actions_dict, device):

    # Counters for top-1 and top-5 accuracy
    correct_top1 = 0
    correct_top5 = 0
    total_frames = 0

    # Counters for precision and recall
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)


    model.eval()
    with torch.no_grad():
        if args.dataset == 'breakfast':
            data_path = './datasets/breakfast'
        elif args.dataset == '50salads' :
            data_path = './datasets/50salads'
        elif args.dataset == 'ek55' :
            data_path = './datasets/ek55'
        gt_path = os.path.join(data_path, 'groundTruth')
        features_path = os.path.join(data_path, 'features')

        eval_p = [0.1, 0.2, 0.3, 0.5]
        pred_p = 0.5
        sample_rate = args.sample_rate
        NONE = n_class-1

        TP_actions = np.zeros((len(eval_p), len(actions_dict)))
        FP_actions = np.zeros((len(eval_p), len(actions_dict)))
        FN_actions = np.zeros((len(eval_p), len(actions_dict)))
        top1_actions = np.zeros((len(eval_p)))

        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE

        for vid in tqdm(vid_list, desc="Processing videos"):
            file_name = vid.split('/')[-1].split('.')[0]

            # load ground truth actions
            gt_file = os.path.join(gt_path, file_name+'.txt')
            gt_read = open(gt_file, 'r')
            gt_seq = gt_read.read().split('\n')[:-1]
            gt_read.close()

            if args.input_type == "TSN":
                features_file = load_lmdb_features(args, file_name)
                features = features_file.transpose()
            else:
                features_file = os.path.join(features_path, file_name+'.npy')
                features = np.load(features_file).transpose()

            vid_len = len(gt_seq)
            past_len = int(obs_p*vid_len)
            future_len = int(pred_p*vid_len)

            past_seq = gt_seq[:past_len]
            features = features[:past_len]
            inputs = features[::sample_rate, :]
            inputs = torch.Tensor(inputs).to(device)

            outputs = model(inputs=inputs.unsqueeze(0), mode='test')
            output_action = outputs['action']

            # top5_preds = torch.topk(output_action, 5, dim=-1)[1]

            output_dur = outputs['duration']
            output_label = output_action.max(-1)[1]

            # fine the forst none class
            none_mask = None
            for i in range(output_label.size(1)) :
                if output_label[0,i] == NONE :
                    none_idx = i
                    break
                else :
                    none = None
            if none_idx is not None :
                none_mask = torch.ones(output_label.shape).type(torch.bool)
                none_mask[0, none_idx:] = False

            output_dur = normalize_duration(output_dur, none_mask.to(device))

            pred_len = (0.5+future_len*output_dur).squeeze(-1).long()

            pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
            predicted = torch.ones(future_len)
            action = output_label.squeeze()

            for i in range(len(action)) :
                predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
                pred_len[i+1] = pred_len[i] + pred_len[i+1]
                if i == len(action) - 1 :
                    predicted[int(pred_len[i]):] = action[i]


            prediction = past_seq
            for i in range(len(predicted)):
                prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

            #evaluation
            for i in range(len(eval_p)):
                p = eval_p[i]
                eval_len = int((obs_p+p)*vid_len)
                eval_prediction = prediction[:eval_len]
                TP_action, FN_action, FP_action, top1_acc = eval_file_more(gt_seq, eval_prediction, obs_p, actions_dict)
                # print(T_action)
                # print(F_action)
                TP_actions[i] += TP_action
                FN_actions[i] += FN_action
                FP_actions[i] += FP_action
                top1_actions[i] += top1_acc

        results = []
        for i in range(len(eval_p)):
            acc = 0
            n = 0
            total_precision = 0
            total_recall = 0
            total_top1 = 0
            for j in range(len(actions_dict)):
                total_actions = TP_actions + FN_actions
                if total_actions[i,j] != 0:
                    acc += float(TP_actions[i,j]/total_actions[i,j])
                    n+=1
                
                # Calculate precision and recall for each class
                if TP_actions[i,j] + FP_actions[i,j] != 0:
                    precision = TP_actions[i,j] / (TP_actions[i,j] + FP_actions[i,j])
                    total_precision += precision

                if TP_actions[i,j] + FN_actions[i,j] != 0:
                    recall = TP_actions[i,j] / (TP_actions[i,j] + FN_actions[i,j])
                    total_recall += recall                
                
            avg_top1_accuracy = top1_actions[i] / len(vid_list)
            # print(top1_actions[i,j])
            # print(total_actions[i,j])

            avg_class_precision = total_precision / len(actions_dict)
            avg_class_recall = total_recall / len(actions_dict)

            result = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_p[i])
            result = result +'\nAverage Top-1 Accuracy: %.4f'%(avg_top1_accuracy) +'\n'
            result = result +'\n\nMean over Classes: %.4f'%(acc)
            result = result +'\nAverage Class Precision: %.4f'%(avg_class_precision)
            result = result +'\nAverage Class Recall: %.4f'%(avg_class_recall) 
            results.append(result)
            print(result)
        print('--------------------------------')

        return






