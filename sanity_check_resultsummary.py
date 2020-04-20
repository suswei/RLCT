import os
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2

def sanity_check_result_summary(hyperparameter_config, dir_path: str='D:/UMelb/PhD_Projects/RLCT/sanity_check/', task_numbers: int=216):

    results_total = pd.DataFrame()

    for i in range(task_numbers):
        file_path = dir_path + 'taskid%s/configuration_plus_results.csv'%(i)
        if os.path.isfile(file_path):
            one_taskid = pd.read_csv(file_path).iloc[0, :].to_frame().T
            one_taskid = pd.concat([pd.DataFrame.from_dict({'taskid':[i]}), one_taskid], axis=1)
            results_total = pd.concat([results_total, one_taskid],axis=0)

    for dataset in ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic']:

        results_dataset = results_total[results_total['dataset']==dataset]

        keys, values = zip(*hyperparameter_config.items())
        hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for config_index in range(len(hyperparameter_experiments)):
            key, value = zip(*hyperparameter_experiments[config_index].items())
            epochs = value[0]
            lr_primal = value[1]
            lr_dual = value[2]

            results_dataset_config = results_dataset[results_dataset.epochs.eq(epochs) & results_dataset.lr_primal.eq(lr_primal) & results_dataset.lr_dual.eq(lr_dual)]
            if results_dataset_config.shape[0] != 0:
                sample_size = results_dataset_config.loc[:, ['syntheticsamplesize']].values[:,0]
                true_RLCT = results_dataset_config.loc[:, ['true_RLCT']].values[:,0]
                d_on_2 = results_dataset_config.loc[:, ['d_on_2']].values[:,0]
                estimated_RLCT_OLS = results_dataset_config.loc[:, ['mean RLCT estimates (OLS)']].values[:,0]
                std_estimated_RLCT_OLS = results_dataset_config.loc[:, ['std RLCT estimates (OLS)']].values[:,0]
                estimated_RLCT_GLS = results_dataset_config.loc[:, ['mean RLCT estimates (GLS)']].values[:,0]
                std_estimated_RLCT_GLS = results_dataset_config.loc[:, ['std RLCT estimates (GLS)']].values[:,0]

                plt.figure(figsize=(10, 7))
                plt.plot(sample_size, d_on_2,
                         sample_size, true_RLCT)
                plt.errorbar(sample_size, estimated_RLCT_OLS, yerr=std_estimated_RLCT_OLS)
                plt.legend(('d_on_2', 'true RLCT', 'estimated RLCT (OLS)'), loc='upper right', fontsize=16)
                plt.xlabel('sample size', fontsize=16)
                plt.ylabel('RLCT', fontsize=16)
                plt.ylim((0, max(d_on_2)+2))
                plt.title('taskid:{}, dataset:{}, epochs:{}, lr_primal:{}, lr_dual:{}'.format(results_dataset_config.loc[:,['taskid']].values[0:], dataset, epochs, lr_primal, lr_dual), fontsize=10)
                plt.savefig(dir_path + 'dataset{}_epochs{}_lrprimal{}_lrdual{}.png'.format(dataset, epochs, lr_primal, lr_dual))
                plt.close()

def Display_Image(save_path, file_paths: list = ['Not_Specified'], subfile_titles: list = ['Not_Specified'], figtitle: str = 'Not_Specified', n_column: int = 2):

    if file_paths == ['Not_Specified']:
        print('Please specify the paths to read in the images first.')
    else:
        images = []
        for file_path in file_paths:
            images.append(mpimg.imread(file_path))
        if len(images) == 1:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax = plt.subplot(1,1,1)
            ax.axis('off')
            plt.imshow(images[0])
        else:
            columns = n_column
            if len(images)%columns == 0:
                fig, ax = plt.subplots(len(images) // columns, columns) #figsize=(10 * columns, 7 * (len(images) // columns))
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            else:
                fig, ax = plt.subplots(len(images) // columns + 1, columns) #figsize=(10 * columns, 7 * (len(images) // columns + 1))
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            #fig.tight_layout()
            if len(images) <= columns:
                for i,image in enumerate(images):
                    ax[i] = plt.subplot(1, columns, i + 1)
                    ax[i].axis('off')
                    if subfile_titles != ['Not_Specified']:
                        ax[i].set_title(subfile_titles[i])
                    plt.imshow(image)
            else:
                for i, image in enumerate(images):
                    if len(images)%columns == 0:
                        ax[i // columns, i - (i // columns) * columns] = plt.subplot(len(images) / columns, columns, i + 1)
                    else:
                        ax[i // columns, i - (i // columns) * columns] = plt.subplot(len(images) / columns + 1, columns, i + 1)
                    ax[i//columns,i-(i//columns)*columns].axis('off')
                    if subfile_titles != ['Not_Specified']:
                       ax[i//columns,i-(i//columns)*columns].set_title(subfile_titles[i])
                    plt.imshow(image)
            if len(images)%columns != 0:
                n_blank = columns - (len(images)%columns)
                for j in range(n_blank):
                   ax[-1, -(j+1)].axis('off')
            if figtitle != 'Not_Specified':
               fig.suptitle(figtitle, y=1.02, fontsize=18, verticalalignment='top')
            plt.savefig(save_path)

def main():

    save_dir = 'D:/UMelb/PhD_Projects/RLCT/sanity_check/'
    hyperparameter_config = {
        'epochs': [20, 50, 100,200],
        'lr_primal': [0.05, 0.01, 0.005],
        'lr_dual': [0.005, 0.001]
    }
    sanity_check_result_summary(hyperparameter_config= hyperparameter_config, dir_path= 'D:/UMelb/PhD_Projects/RLCT/sanity_check/', task_numbers= 216)

    for dataset in ['lr_synthetic', '3layertanh_synthetic', 'reducedrank_synthetic']:
        video_name = save_dir + '{}.mp4'.format(dataset)

        file_paths = []

        keys, values = zip(*hyperparameter_config.items())
        hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for config_index in range(len(hyperparameter_experiments)):
            key, value = zip(*hyperparameter_experiments[config_index].items())
            epochs = value[0]
            lr_primal = value[1]
            lr_dual = value[2]

            one_image_path = save_dir + 'dataset{}_epochs{}_lrprimal{}_lrdual{}.png'.format(dataset, epochs, lr_primal, lr_dual)
            if os.path.isfile(one_image_path):
               file_paths += [one_image_path]

        frame = cv2.imread(file_paths[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 1, (width, height))
        for file_path in file_paths:
            video.write(cv2.imread(file_path))
        cv2.destroyAllWindows()
        video.release()
