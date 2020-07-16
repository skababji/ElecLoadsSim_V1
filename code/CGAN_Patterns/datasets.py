import pandas as pd
import numpy as np
from scipy import signal

from CGAN_Patterns.hyperparam import Hyperparam

import xml.etree.ElementTree as ET


def eliminate_overlap(peak_idx, window):
    peak_idx = np.pad(peak_idx, (0, window), 'constant', constant_values=-1)
    i = 0
    while i < len(peak_idx):
        j = i + 1
        while j < len(peak_idx):
            if j >= len(peak_idx) - window:
                peak_idx = peak_idx[peak_idx != -1]
                return peak_idx
            if (peak_idx[j] - peak_idx[i]) < window:
                peak_idx[j] = -1
                j = j + 1
            else:
                i = j
                break


def stack_cycles(master_window,load_data_df, load_tmplt):
    name = load_tmplt.name
    window_size = int(load_tmplt['tmplt_end'] - load_tmplt['tmplt_start'])
    power_seq = load_data_df[name].values
    time_seq = load_data_df['UNIX_TS']
    this_mean = np.mean(power_seq)
    power_seq = power_seq - this_mean
    tmplt = power_seq[load_tmplt['tmplt_start'].astype(int):load_tmplt['tmplt_end'].astype(int)]
    fir_coeff = tmplt[::-1]
    detector = signal.lfilter(fir_coeff, 1, power_seq)
    max_vl = max(detector)
    max_idx = np.argmax(detector)
    detector[max_idx] = -max_vl
    max2_vl = max(detector)
    detector[max_idx] = max_vl
    peak_threshold = load_tmplt['tmplt_threshold'] * max2_vl
    peak_idx = np.where(detector > peak_threshold)
    peak_idx = np.array(peak_idx).ravel()

    peak_idx = eliminate_overlap(peak_idx, window_size)

    samples = np.zeros(window_size + 1)
    for i in range(0, peak_idx.shape[0]):
        if peak_idx[i] < window_size:
            continue
        new_sample = np.array(load_data_df[name][peak_idx[i] - window_size:peak_idx[i]])
        new_sample = np.append(time_seq[peak_idx[i] - window_size], new_sample)
        samples = np.vstack((samples, new_sample))
    samples = samples[~np.all(samples == 0, axis=1)]
    rpt_samples = samples

    """Code below is for repeating examples while shifting"""
    if Hyperparam.SHFT_STEPS >= 1:
        shft_samples = np.zeros_like(samples)
        shft_samples[:, 1:] = np.roll(samples[:, 1:], Hyperparam.SHFT_STEPS, axis=1)
        shft_samples[:, 1:Hyperparam.SHFT_STEPS + 1] = 0
        shft_samples[:, 0] = samples[:, 0]
        rpt_samples = shft_samples

    """Commented Code below is for repeating examples. Can be used to correct  previous OK versions """
    if Hyperparam.RPTD_EXAMPLES >= 2:
        for i in range(Hyperparam.RPTD_EXAMPLES - 1):
            rpt_samples = np.vstack([rpt_samples, samples])

    tmplt = tmplt + this_mean
    tmplt = np.pad(tmplt, (0, master_window - tmplt.shape[0]), 'constant',
                   constant_values=(0, 0))  # pad 1d array with zeros
    padder = np.zeros([rpt_samples.shape[0], master_window - rpt_samples.shape[1]])
    rpt_samples = np.hstack([rpt_samples, padder])

    load_data_df = pd.DataFrame(rpt_samples)
    load_data_df = load_data_df.rename(columns={0: 'starttime'})
    load_data_df.insert(loc=0, column='name', value=name)
    return load_data_df, tmplt


    tmplt = tmplt + this_mean
    tmplt = np.pad(tmplt, (0, master_window - tmplt.shape[0]), 'constant',
                   constant_values=(0, 0))  # pad 1d array with zeros
    padder = np.zeros([rpt_samples.shape[0], master_window - rpt_samples.shape[1]])
    rpt_samples = np.hstack([rpt_samples, padder])

    load_data_df = pd.DataFrame(rpt_samples)
    load_data_df = load_data_df.rename(columns={0: 'starttime'})
    load_data_df.insert(loc=0, column='name', value=name)
    return load_data_df, tmplt


class Datasets:
    def __init__(self, xml_file):
        self.parse_xml(xml_file)

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        x = ET.tostring(root, encoding='utf8').decode('utf8')

        datasets = []
        n_datasets = len(list(root))
        for i in range(n_datasets):
            this_dataset_descp = root[i].attrib
            datasets.append(this_dataset_descp)

        lists_short_names = []
        lists_long_names = []
        lists_tmplt_start = []
        lists_tmplt_end = []
        lists_tmplt_threshold = []
        n_datasets = len(list(root))
        for i in range(n_datasets):
            sublist_short_names = []
            sublist_long_names = []
            sublist_tmplt_start = []
            sublist_tmplt_end = []
            sublist_tmplt_threshold = []
            n_loads_sublist = len(list(root[i]))
            for j in range(n_loads_sublist):
                tmp = root[i][j].attrib.get('short_name')
                sublist_short_names.append(tmp)
                tmp = root[i][j].attrib.get('long_name')
                sublist_long_names.append(tmp)
                tmp = int(root[i][j].attrib.get('tmplt_start'))
                sublist_tmplt_start.append(tmp)
                tmp = int(root[i][j].attrib.get('tmplt_end'))
                sublist_tmplt_end.append(tmp)
                tmp = float(root[i][j].attrib.get('tmplt_threshold'))
                sublist_tmplt_threshold.append(tmp)
            lists_short_names.append(sublist_short_names)
            lists_long_names.append(sublist_long_names)
            lists_tmplt_start.append(sublist_tmplt_start)
            lists_tmplt_end.append(sublist_tmplt_end)
            lists_tmplt_threshold.append(sublist_tmplt_threshold)

        self.datasets=datasets
        self.n_datasets=n_datasets
        self.short_names=lists_short_names
        self.long_names=lists_long_names
        self.tmplt_start=np.array(lists_tmplt_start)
        self.tmplt_end=np.array(lists_tmplt_end)
        self.tmplt_threshold=np.array(lists_tmplt_threshold)
        self.datasets_steps=self.calc_datasets_step()
        self.datasets_loads_windows=self.calc_loads_windows()
        self.master_window=int(np.max(np.concatenate(self.datasets_loads_windows))+1)
        self.tmplt_lookup_df=self.make_tmplt_lookup()
        self.all_short_names, self.all_long_names=self.merge_all_names()

    def make_tmplt_lookup(self):
        tmplt_lookup = []
        for i in range(self.n_datasets):
            tmplt_start = (np.array(self.tmplt_start[i])/self.datasets_steps[i]).reshape(1, -1).astype(int)
            tmplt_end = (np.array(self.tmplt_end[i])/self.datasets_steps[i]).reshape(1, -1).astype(int)
            tmplt_threshold = np.array(self.tmplt_threshold[i]).reshape(1, -1)
            tmplt_lookup_arr = np.concatenate([tmplt_start, tmplt_end, tmplt_threshold], axis=0)
            tmplt_lookup_df = pd.DataFrame(data=tmplt_lookup_arr, index=['tmplt_start', 'tmplt_end', 'tmplt_threshold'],
                                           columns=self.short_names[i])
            tmplt_lookup.append(tmplt_lookup_df)
        return  tmplt_lookup

    def calc_datasets_step(self):
        datasets_steps=[]
        for i in range(self.n_datasets):
            this_step=int(Hyperparam.TS_UNIFIED/int(self.datasets[i].get('sampling_period')))
            datasets_steps.append(this_step)
        return np.array(datasets_steps)

    def calc_loads_windows(self):
        datasets_loads_windows=[]
        for i in range(self.n_datasets):
            tmp=((np.array(self.tmplt_end[i])-np.array(self.tmplt_start[i]))/self.datasets_steps[i]).astype(int)
            datasets_loads_windows.append(tmp)
        return datasets_loads_windows


    def read_raw(self, raw_input_path,dataset):
        data_file=dataset.get('data_file')
        rP = pd.read_csv(raw_input_path + data_file)
        return rP

    
    def resample(self, rP, sampling_period):
        sampling_period=int(sampling_period)
        step=int(Hyperparam.TS_UNIFIED/sampling_period)
        rP = rP.iloc[step-1:, :]
        rP_resampled = rP.iloc[::step, :]
        rP_resampled.reset_index(drop=True, inplace=True)
        return rP_resampled

    def make_examples(self,dataset_name,loads_df,tmplt_lookup_df,documenter):
        examples = []
        for i in range(0, tmplt_lookup_df.shape[1]):
            load_name = tmplt_lookup_df.columns[i]
            load_filtered, load_tmplt = stack_cycles(self.master_window, loads_df[['UNIX_TS', load_name]], tmplt_lookup_df[load_name])
            documenter.record_input(dataset_name+" Data Examples for: " + load_name + " after Matched Filter = " + str(load_filtered.shape[0]) + " \n")
            examples.append(load_filtered)
        return pd.concat(examples)


    def combine_examples(self,examples_list):
        return pd.concat(examples_list,axis=0)

    def merge_all_names(self):
        all_short_names=np.concatenate(self.short_names)
        all_long_names=np.concatenate(self.long_names)
        return np.unique(all_short_names),np.unique(all_long_names)
