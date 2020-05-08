import os
import pandas as pd
import itertools
import pickle
import plotly.graph_objects as go
import cv2

def draw_plot(method, dataframe, fig_title, save_path):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                             y=dataframe.loc[:, ['d on 2']].values[:, 0],
                             mode='lines+markers', text=['n=500', 'n=1000', 'n=5000'], name='d on 2'))
    fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                             y=dataframe.loc[:, ['true_RLCT']].values[:, 0],
                             mode='lines+markers', name='true RLCT'))

    if method=='thm4':
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct robust thm4 mean']].values[:, 0],
                                 error_y=dict(type='data',
                                              array=dataframe.loc[:, ['rlct robust thm4 std']].values[:, 0],
                                              visible=True),
                                 mode='lines+markers', name='robust thm4'))

        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct ols thm4 mean']].values[:, 0],
                                 error_y=dict(type='data',
                                              array=dataframe.loc[:, ['rlct ols thm4 std']].values[:, 0],
                                              visible=True),
                                 mode='lines+markers', name='ols thm4'))

    elif method=='thm4average':
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct robust thm4 average']].values[:, 0],
                                 mode='lines+markers', name='robust thm4 average'))
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct ols thm4 average']].values[:, 0],
                                 mode='lines+markers', name='ols thm4 average'))
    elif method=='varTI':
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct var TI mean']].values[:, 0],
                                 error_y=dict(type='data',
                                              array=dataframe.loc[:, ['rlct var TI std']].values[:, 0],
                                              visible=True),
                                 mode='lines+markers', name='var TI'))

    fig.update_xaxes(title_text='sample size')
    fig.update_layout(
        title={'text': fig_title,
               'y': 0.92,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size =10, color='black', family='Arial, sans-serif'),
        xaxis=dict(tickmode='linear',tick0=0, dtick=500)
    )
    fig.write_image(save_path)


def sanity_check_result_summary(VItype, hyperparameter_config):

    dir_path = './{}_sanity_check/'.format(VItype)

    hyperparameter_config_subset = {key: value for key, value in hyperparameter_config.items() if key not in ['dataset', 'syntheticsamplesize']}
    keys, values = zip(*hyperparameter_config_subset.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_config_total = pd.DataFrame()

    for i in range(len(hyperparameter_config['dataset'])*len(hyperparameter_config['syntheticsamplesize'])*len(hyperparameter_experiments)):
        config_file_path = dir_path + 'taskid%s/config.pkl'%(i)
        results_file_path = dir_path + 'taskid%s/results.pkl'%(i)
        if os.path.isfile(config_file_path) and os.path.isfile(results_file_path):
            config = pickle.load(open(config_file_path, "rb"))
            results = pickle.load(open(results_file_path,"rb"))

            results_config = {key: value for key, value in results.items() if key not in ['rlct robust thm4 array', 'rlct ols thm4 array', 'rlct var TI array']}
            results_config.update({'taskid': i})
            results_config.update({key: value for key, value in config.items() if key in tuple(['dataset','syntheticsamplesize']) + keys})
            results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config, orient='index').transpose()],axis=0)

    for dataset in hyperparameter_config['dataset']:

        results_config_dataset = results_config_total[results_config_total['dataset']==dataset]

        for config_index in range(len(hyperparameter_experiments)):
            temp = hyperparameter_experiments[config_index]
            results_oneconfig_dataset = results_config_dataset
            for key in keys:
                results_oneconfig_dataset = results_oneconfig_dataset[results_oneconfig_dataset[key]==temp[key]]

            for method in ['thm4', 'thm4average', 'varTI']:
                save_path = dir_path + '{}_{}_{}'.format(dataset, VItype, method)
                for key in keys:
                    save_path += '_{}{}'.format(key, temp[key])
                save_path += '.png'

                fig_title = '{}, {}, {}, <br>'.format(dataset, VItype, method)
                for index, key in enumerate(keys[0:-1]):
                    if index % 2 == 0:
                        fig_title += '{}: {}, '.format(key, temp[key])
                    else:
                        if index == 1:
                           fig_title += '{}: {}, '.format(key, temp[key])
                           fig_title += '{}: {}, <br>'.format('dpower', temp['dpower'])
                        elif index < len(keys[0:-1])-1:
                           fig_title += '{}: {}, <br>'.format(key, temp[key])
                        else:
                            fig_title += '{}: {}'.format(key, temp[key])

                if method == 'thm4':
                    draw_plot(method, results_oneconfig_dataset, fig_title, save_path)
                elif method == 'thm4average':
                    draw_plot(method, results_oneconfig_dataset, fig_title, save_path)
                else:
                    draw_plot(method, results_oneconfig_dataset, fig_title, save_path)

def main( ):
    VItype = 'implicit'
    hyperparameter_config = {
        'dataset': ['reducedrank_synthetic', 'tanh_synthetic'],
        'syntheticsamplesize': [500, 1000, 5000],
        'batchsize': [100],  # 50, 100
        'betasend': [0.5, 1.5],
        'n_hidden_D': [128],  # 128, 256
        'num_hidden_layers_D': [2],  # 1,2
        'n_hidden_G': [128],  # 128, 256
        'num_hidden_layers_G': [2],  # 1,2
        'dpower': [2 / 5],  # 2/5, 4/5
    }

    dir_path = './{}_sanity_check/'.format(VItype)

    sanity_check_result_summary(hyperparameter_config= hyperparameter_config, VItype=VItype)

    hyperparameter_config_subset =  {key: value for key, value in hyperparameter_config.items() if key not in ['dataset','syntheticsamplesize']}
    keys, values = zip(*hyperparameter_config_subset.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for dataset in hyperparameter_config['dataset']:
        for method in ['thm4', 'thm4average', 'varTI']:
            video_name = dir_path + '{}_{}_{}.mp4'.format(dataset, VItype, method)

            file_paths = []
            for config_index in range(len(hyperparameter_experiments)):
                temp = hyperparameter_experiments[config_index]
                one_image_path = dir_path + '{}_{}_{}'.format(dataset, VItype,method)
                for key in keys:
                    one_image_path += '_{}{}'.format(key, temp[key])
                one_image_path += '.png'

                if os.path.isfile(one_image_path):
                   file_paths += [one_image_path]

            frame = cv2.imread(file_paths[0])
            height, width, layers = frame.shape
            video = cv2.VideoWriter(video_name, 0, 1, (width, height))
            for file_path in file_paths:
                video.write(cv2.imread(file_path))
            cv2.destroyAllWindows()
            video.release()