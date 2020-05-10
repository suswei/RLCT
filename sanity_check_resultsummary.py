import os
import pandas as pd
import itertools
import pickle
import plotly.graph_objects as go
import cv2

def draw_plot(method, dataframe, fig_title, save_path):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                             y=dataframe.loc[:, ['d_on_2']].values[:, 0],
                             mode='lines+markers', name='d on 2'))
    fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                             y=dataframe.loc[:, ['trueRLCT']].values[:, 0],
                             mode='lines+markers', name='true RLCT'))

    if method=='thm4':
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct_robust_thm4_mean']].values[:, 0],
                                 error_y=dict(type='data',
                                              array=dataframe.loc[:, ['rlct_robust_thm4_std']].values[:, 0],
                                              visible=True),
                                 mode='lines+markers', name='robust thm4'))

        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct_ols_thm4_mean']].values[:, 0],
                                 error_y=dict(type='data',
                                              array=dataframe.loc[:, ['rlct_ols_thm4_std']].values[:, 0],
                                              visible=True),
                                 mode='lines+markers', name='ols thm4'))

    elif method=='thm4average':
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct_robust_thm4_average']].values[:, 0],
                                 mode='lines+markers', name='robust thm4 average'))
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct_ols_thm4_average']].values[:, 0],
                                 mode='lines+markers', name='ols thm4 average'))
    elif method=='varTI':
        fig.add_trace(go.Scatter(x=dataframe.loc[:, ['syntheticsamplesize']].values[:, 0],
                                 y=dataframe.loc[:, ['rlct_var_TI_mean']].values[:, 0],
                                 error_y=dict(type='data',
                                              array=dataframe.loc[:, ['rlct_var_TI_std']].values[:, 0],
                                              visible=True),
                                 mode='lines+markers', name='var TI'))

    fig.update_xaxes(title_text='sample size')
    fig.update_layout(
        title={'text': fig_title,
               'y': 0.9,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size =10, color='black', family='Arial, sans-serif'),
        xaxis=dict(tickmode='linear',tick0=0, dtick=500)
    )
    fig.write_image(save_path)


def sanity_check_result_summary(VItype, hyperparameter_config):

    dir_path = './{}_sanity_check/'.format(VItype)

    hyperparameter_config_subset = {key: value for key, value in hyperparameter_config.items() if key not in ['dataset', 'syntheticsamplesize', 'MCs']}
    keys, values = zip(*hyperparameter_config_subset.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_config_total = pd.DataFrame()
    for i in range(len(hyperparameter_config['dataset'])*len(hyperparameter_config['syntheticsamplesize'])*len(hyperparameter_experiments)*len(hyperparameter_config['MCs'])):
        config_file_path = dir_path + 'taskid{}/config.pkl'.format(i)
        results_file_path = dir_path + 'taskid{}/results.pkl'.format(i)
        if os.path.isfile(config_file_path) and os.path.isfile(results_file_path):
            config = pickle.load(open(config_file_path, "rb"))
            results = pickle.load(open(results_file_path,"rb"))

            results_config = {key: value for key, value in config.items() if key in tuple(['dataset','syntheticsamplesize']) + keys + ('trueRLCT', 'w_dim')}
            results_config.update({key: value[0] for key, value in results.items() if key in ['rlct robust thm4 array', 'rlct ols thm4 array']})
            results_config = {key: [value] for key, value in results_config.items()}

            results_config_total = pd.concat([results_config_total,pd.DataFrame.from_dict(results_config)],axis=0)

    results_config_total['d_on_2'] = results_config_total.apply(lambda row: row.w_dim / 2, axis=1)

    method = 'thm4'
    results_config_mean_std = results_config_total.groupby([ele for ele in list(results_config_total.columns) if ele not in ['rlct robust thm4 array','rlct ols thm4 array']]).agg(
        rlct_robust_thm4_mean=('rlct robust thm4 array', 'mean'),
        rlct_robust_thm4_std=('rlct robust thm4 array', 'std'),
        rlct_ols_thm4_mean=('rlct ols thm4 array', 'mean'),
        rlct_ols_thm4_std=('rlct ols thm4 array', 'std')
    ).reset_index()

    for dataset in hyperparameter_config['dataset']:

        results_config_dataset = results_config_mean_std[results_config_mean_std['dataset']==dataset]

        for config_index in range(len(hyperparameter_experiments)):
            temp = hyperparameter_experiments[config_index]
            results_oneconfig_dataset = results_config_dataset
            for key in keys:
                results_oneconfig_dataset = results_oneconfig_dataset[results_oneconfig_dataset[key]==temp[key]]

            save_path = dir_path + '{}_{}_{}'.format(dataset, VItype, method)
            for key in keys:
                save_path += '_{}{}'.format(key, temp[key])
            save_path += '.png'

            fig_title = '{}, {}, {}, <br>'.format(dataset, VItype, method)
            for index, key in enumerate(keys):
                if index < len(keys)-1:
                    fig_title += '{}: {}, '.format(key, temp[key])
                else:
                    fig_title += '{}: {}'.format(key, temp[key])
            '''
            for index, key in enumerate(keys): # if there are too many hyperparameters, exhibit the fig title in several lines.
                if index % 2 == 0:
                    fig_title += '{}: {}, '.format(key, temp[key])
                else:
                    if index < len(keys)-1:
                        fig_title += '{}: {}, <br>'.format(key, temp[key])
                    else:
                        fig_title += '{}: {}'.format(key, temp[key])
            '''
            draw_plot(method, results_oneconfig_dataset, fig_title, save_path)

def main( ):
    VItype = 'implicit'
    hyperparameter_config = {
        'dataset': ['reducedrank_synthetic', 'tanh_synthetic'],
        'syntheticsamplesize': [1000, 5000],
        'dpower': [0.4],
        'betasbegin': [0.05],
        'betasend': [0.2],
        'MCs': 50*[1]
    }

    dir_path = './{}_sanity_check/'.format(VItype)

    sanity_check_result_summary(hyperparameter_config= hyperparameter_config, VItype=VItype)

    hyperparameter_config_subset =  {key: value for key, value in hyperparameter_config.items() if key not in ['dataset','syntheticsamplesize', 'MCs']}
    keys, values = zip(*hyperparameter_config_subset.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    method = 'thm4'
    for dataset in hyperparameter_config['dataset']:
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