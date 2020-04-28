import os
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import cv2

def sanity_check_result_summary(hyperparameter_config, VI_type, dir_path: str='D:/UMelb/PhD_Projects/RLCT/sanity_check/', task_numbers: list=list(range(0,216))):

    results_total = pd.DataFrame()

    for i in task_numbers:
        file_path = dir_path + 'taskid%s/configuration_plus_results.csv'%(i)
        if os.path.isfile(file_path):
            result_onetask = pd.read_csv(file_path)
            if result_onetask.shape[0] != 0:
                one_taskid = result_onetask.iloc[0, :].to_frame().T
                one_taskid = pd.concat([pd.DataFrame.from_dict({'taskid':[i]}), one_taskid], axis=1)
                results_total = pd.concat([results_total, one_taskid],axis=0)

    for dataset in ['lr_synthetic', 'tanh_synthetic', 'reducedrank_synthetic']:

        results_dataset = results_total[results_total['dataset']==dataset]

        keys, values = zip(*hyperparameter_config.items())
        hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for config_index in range(len(hyperparameter_experiments)):
            key, value = zip(*hyperparameter_experiments[config_index].items())
            beta_auto = value[0]
            dpower = value[1]
            if VI_type == 'implicit':
                lr_primal = value[2]
                lr_dual = value[3]
            else:
                lr = value[2]

            if VI_type == 'implicit':
                if beta_auto == 'beta_auto_liberal':
                    results_dataset_config = results_dataset[results_dataset.beta_auto_liberal.eq(True) & results_dataset.dpower.eq(dpower) & results_dataset.lr_primal.eq(lr_primal) & results_dataset.lr_dual.eq(lr_dual)]
                elif beta_auto == 'beta_auto_conservative':
                    results_dataset_config = results_dataset[results_dataset.beta_auto_conservative.eq(True) & results_dataset.dpower.eq(dpower) & results_dataset.lr_primal.eq(lr_primal) & results_dataset.lr_dual.eq(lr_dual)]
            else:
                if beta_auto == 'beta_auto_liberal':
                    results_dataset_config = results_dataset[results_dataset.beta_auto_liberal.eq(True) & results_dataset.dpower.eq(dpower) & results_dataset.lr.eq(lr)]
                elif beta_auto == 'beta_auto_conservative':
                    results_dataset_config = results_dataset[results_dataset.beta_auto_conservative.eq(True) & results_dataset.dpower.eq(dpower) & results_dataset.lr.eq(lr)]

            if results_dataset_config.shape[0] != 0:
                training_sample_size = results_dataset_config.loc[:, ['syntheticsamplesize']].values[:,0]
                batch_size = results_dataset_config.loc[:, ['batchsize']].values[:,0]
                true_RLCT = results_dataset_config.loc[:, ['true_RLCT']].values[:,0]
                d_on_2 = results_dataset_config.loc[:, ['d_on_2']].values[:,0]
                estimated_RLCT_OLS = results_dataset_config.loc[:, ['mean RLCT estimates (OLS)']].values[:, 0]
                std_estimated_RLCT_OLS = results_dataset_config.loc[:, ['std RLCT estimates (OLS)']].values[:, 0]
                estimated_RLCT_robust = results_dataset_config.loc[:, ['mean RLCT estimates (robust)']].values[:,0]
                std_estimated_RLCT_robust = results_dataset_config.loc[:, ['std RLCT estimates (robust)']].values[:,0]

                fig = plt.figure(figsize=(10, 7))
                ax = plt.subplot(1, 1, 1)
                plt.plot(training_sample_size[::2], d_on_2[::2],
                         training_sample_size[::2], true_RLCT[::2])
                plt.errorbar(training_sample_size[::2], estimated_RLCT_OLS[::2], yerr=std_estimated_RLCT_OLS[::2])
                plt.errorbar(training_sample_size[::2], estimated_RLCT_robust[::2], yerr=std_estimated_RLCT_robust[::2])

                plt.legend(('d_on_2', 'true RLCT', 'estimated RLCT (OLS)', 'estimated RLCT (robust)'), loc='upper right', fontsize=16)
                trans_offset1 = mtransforms.offset_copy(ax.transData, fig=fig,x=-0.2, y=0.10, units='inches')
                for t, (x, y) in enumerate(zip(training_sample_size[::2], estimated_RLCT_OLS[::2])):
                    plt.text(x, y, 'n=%d' % (training_sample_size[::2][t]), transform=trans_offset1)
                trans_offset2 = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.2, y=-0.10, units='inches')
                for t, (x, y) in enumerate(zip(training_sample_size[::2], estimated_RLCT_robust[::2])):
                    plt.text(x, y, 'n=%d' % (training_sample_size[::2][t]), transform=trans_offset2)

                plt.xlabel('training sample size', fontsize=16)
                plt.ylabel('RLCT', fontsize=16)
                plt.ylim((0, max(d_on_2)+3.5))
                if VI_type == 'implicit':
                    plt.title('taskid:{}, dataset:{}, {}, lowerbatchsize, dpower:{}, lr_primal:{}, lr_dual:{}'.format(results_dataset_config.loc[:,['taskid']].values[0:], dataset, beta_auto, dpower, lr_primal, lr_dual), fontsize=10)
                    plt.savefig(dir_path + 'dataset{}_{}_{}_lowerbatchsize_dpower{}_lrprimal{}_lrdual{}.png'.format(dataset, VI_type, beta_auto, dpower, lr_primal, lr_dual))
                else:
                    plt.title('taskid:{}, dataset:{}, {}, lowerbatchsize, dpower:{} lr:{}'.format(results_dataset_config.loc[:, ['taskid']].values[0:], dataset, beta_auto, dpower, lr),fontsize=10)
                    plt.savefig(dir_path + 'dataset{}_{}_{}_lowerbatchsize_dpower{}_lr{}.png'.format(dataset, VI_type, beta_auto, dpower, lr))
                plt.close()

                fig = plt.figure(figsize=(10, 7))
                ax = plt.subplot(1, 1, 1)
                plt.plot(training_sample_size[1::2], d_on_2[1::2],
                         training_sample_size[1::2], true_RLCT[1::2])
                plt.errorbar(training_sample_size[1::2], estimated_RLCT_OLS[1::2], yerr=std_estimated_RLCT_OLS[1::2])
                plt.errorbar(training_sample_size[1::2], estimated_RLCT_robust[1::2], yerr=std_estimated_RLCT_robust[1::2])

                plt.legend(('d_on_2', 'true RLCT', 'estimated RLCT (OLS)', 'estimated RLCT (robust)'),loc='upper right', fontsize=16)
                trans_offset1 = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.2, y=0.10, units='inches')
                for t, (x, y) in enumerate(zip(training_sample_size[1::2], estimated_RLCT_OLS[1::2])):
                    plt.text(x, y, 'n=%d' % (training_sample_size[1::2][t]),transform=trans_offset1)
                trans_offset2 = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.2, y=-0.10, units='inches')
                for t, (x, y) in enumerate(zip(training_sample_size[1::2], estimated_RLCT_robust[1::2])):
                    plt.text(x, y, 'n=%d' % (training_sample_size[1::2][t]),transform=trans_offset2)

                plt.xlabel('training sample size', fontsize=16)
                plt.ylabel('RLCT', fontsize=16)
                plt.ylim((0, max(d_on_2) + 3.5))
                if VI_type == 'implicit':
                    plt.title('taskid:{}, dataset:{}, {}, higherbatchsize, dpower:{}, lr_primal:{}, lr_dual:{}'.format(results_dataset_config.loc[:, ['taskid']].values[0:], dataset, beta_auto, dpower, lr_primal,lr_dual), fontsize=10)
                    plt.savefig(dir_path + 'dataset{}_{}_{}_higherbatchsize_dpower{}_lrprimal{}_lrdual{}.png'.format(dataset,VI_type,beta_auto,dpower, lr_primal,lr_dual))
                else:
                    plt.title('taskid:{}, dataset:{}, {}, higherbatchsize, dpower:{} lr:{}'.format(results_dataset_config.loc[:, ['taskid']].values[0:], dataset, beta_auto, dpower, lr),fontsize=10)
                    plt.savefig(dir_path + 'dataset{}_{}_{}_higherbatchsize_dpower{}_lr{}.png'.format(dataset, VI_type, beta_auto, dpower,lr))
                plt.close()


def main(VI_type):

    save_dir = 'D:/UMelb/PhD_Projects/RLCT/{}_sanity_check/'.format(VI_type)

    if VI_type == 'implicit':
        hyperparameter_config = {
            'betasend': [0.2, 0.5, 1.5],
            'dpower': [2/5, 4/5],
            'lr_primal': [0.05, 0.01],
            'lr_dual': [0.005, 0.001]
        }
    elif VI_type == 'explicit':
        hyperparameter_config = {
            'beta_auto': ['beta_auto_liberal', 'beta_auto_conservative'],
            'dpower': [2 / 5, 4 / 5],  # 1/5, 2/5, 3/5, 4/5
            'lr': [0.05, 0.01, 0.001]
        }

    if VI_type == 'implicit':
       sanity_check_result_summary(hyperparameter_config= hyperparameter_config, VI_type=VI_type, dir_path= save_dir, task_numbers= list(range(0, 288)))
    else:
        sanity_check_result_summary(hyperparameter_config=hyperparameter_config, VI_type=VI_type, dir_path=save_dir, task_numbers=list(range(0, 216)))

    for dataset in ['lr_synthetic', 'tanh_synthetic', 'reducedrank_synthetic']:
        video_name = save_dir + '{}_{}.mp4'.format(VI_type, dataset)

        file_paths = []

        keys, values = zip(*hyperparameter_config.items())
        hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for config_index in range(len(hyperparameter_experiments)):
            key, value = zip(*hyperparameter_experiments[config_index].items())
            beta_auto = value[0]
            dpower = value[1]
            if VI_type == 'implicit':
               lr_primal = value[2]
               lr_dual = value[3]
            elif VI_type == 'explicit':
               lr = value[2]

            if VI_type == 'implicit':
               one_image_path1 = save_dir + 'dataset{}_{}_{}_lowerbatchsize_dpower{}_lrprimal{}_lrdual{}.png'.format(dataset, VI_type, beta_auto, dpower, lr_primal, lr_dual)
               one_image_path2 = save_dir + 'dataset{}_{}_{}_higherbatchsize_dpower{}_lrprimal{}_lrdual{}.png'.format(dataset, VI_type, beta_auto, dpower, lr_primal, lr_dual)
            else:
                one_image_path1 = save_dir + 'dataset{}_{}_{}_lowerbatchsize_dpower{}_lr{}.png'.format(dataset, VI_type, beta_auto, dpower, lr)
                one_image_path2 = save_dir + 'dataset{}_{}_{}_higherbatchsize_dpower{}_lr{}.png'.format(dataset, VI_type,beta_auto, dpower,lr)

            if os.path.isfile(one_image_path1):
               file_paths += [one_image_path1]
            if os.path.isfile(one_image_path2):
               file_paths += [one_image_path2]

        frame = cv2.imread(file_paths[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 1, (width, height))
        for file_path in file_paths:
            video.write(cv2.imread(file_path))
        cv2.destroyAllWindows()
        video.release()
