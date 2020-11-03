import numpy as np
import cupy as cp

from pykilosort.preprocess import get_good_channels, get_Nbatch, gpufilter, get_whitening_matrix
from pykilosort.main import run_preprocess, run_spikesort, run_export
from PyQt5 import QtCore


def find_good_channels(context):
    params = context.params
    probe = context.probe
    raw_data = context.raw_data
    intermediate = context.intermediate

    if 'igood' not in intermediate:
        if params.minfr_goodchannels > 0:  # discard channels that have very few spikes
            # determine bad channels
            with context.time('good_channels'):
                intermediate.igood = get_good_channels(raw_data=raw_data, probe=probe, params=params)
                intermediate.igood = intermediate.igood.ravel()
            # Cache the result.
            context.write(igood=intermediate.igood)

        else:
            intermediate.igood = np.ones_like(probe.chanMap, dtype=bool)

    # probe.chanMap = probe.chanMap[intermediate.igood]
    # probe.xc = probe.xc[intermediate.igood]
    # probe.yc = probe.yc[intermediate.igood]
    # probe.kcoords = probe.kcoords[intermediate.igood]
    probe.Nchan = len(probe.chanMap)
    #
    context.probe = probe
    return context


def filter_and_whiten(raw_traces, params, probe, whitening_matrix):
    num_of_batches = get_Nbatch(raw_traces, params)
    Wrot = cp.asarray(whitening_matrix, dtype=np.float32)

    sample_rate = params.fs
    high_pass_freq = params.fshigh
    low_pass_freq = params.fslow
    NT = params.NT
    NTbuff = params.NTbuff
    ntbuff = params.ntbuff

    whitened_arrays = []

    for ibatch in range(num_of_batches):

        # number of samples to start reading at.
        i = max(0, (NT - ntbuff) * ibatch - 2 * ntbuff)
        if ibatch == 0:
            # The very first batch has no pre-buffer, and has to be treated separately
            ioffset = 0
        else:
            ioffset = ntbuff

        buff = raw_traces[i:i + NTbuff]
        if buff.size == 0:
            print("Loaded buffer has an empty size!")
            break  # this shouldn't really happen, unless we counted data batches wrong

        nsampcurr = buff.shape[0]  # how many time samples the current batch has
        if nsampcurr < NTbuff:
            buff = np.concatenate(
                (buff, np.tile(buff[nsampcurr - 1], (NTbuff, 1))), axis=0)

        # apply filters and median subtraction
        buff = cp.asarray(buff, dtype=np.float32)

        datr = gpufilter(buff, chanMap=probe.chanMap, fs=sample_rate, fshigh=high_pass_freq, fslow=low_pass_freq)
        assert datr.flags.c_contiguous

        datr = datr[ioffset:ioffset + NT, :]  # remove timepoints used as buffers
        # TODO: unclear - comment says we are scaling by 200. Is wrot already scaled?
        #               - we should definitely scale as we could be hit badly by precision here.
        datr = cp.dot(datr, Wrot)  # whiten the data and scale by 200 for int16 range
        assert datr.flags.c_contiguous

        whitened_arrays.append(datr)

    concatenated_array = cp.concatenate(tuple(whitened_arrays), axis=0)
    array_means = cp.mean(concatenated_array, axis=0)
    array_stds = cp.std(concatenated_array, axis=0)
    whitened_array = (concatenated_array - array_means) / array_stds
    return whitened_array.get()


def get_whitened_traces(raw_data, probe, params, whitening_matrix):
    if whitening_matrix is None:
        whitening_matrix = get_whitening_matrix(raw_data=raw_data, probe=probe, params=params)
    whitened_traces = filter_and_whiten(raw_traces=raw_data, params=params,
                                        probe=probe, whitening_matrix=whitening_matrix)
    return whitened_traces, whitening_matrix


def get_predicted_traces(matrix_U, matrix_W, sorting_result, time_limits):
    W = cp.asarray(matrix_W, dtype=np.float32)
    U = cp.asarray(matrix_U, dtype=np.float32)

    buffer = W.shape[0]

    predicted_traces = cp.zeros((U.shape[0], 4 * buffer + (time_limits[1] - time_limits[0])), dtype=np.int16)

    sorting_result = cp.asarray(sorting_result)

    all_spike_times = sorting_result[:, 0]
    included_spike_pos = cp.asarray((time_limits[0] - buffer // 2 < all_spike_times) &
                                    (all_spike_times < time_limits[1] + buffer // 2)).nonzero()[0]

    spike_times = all_spike_times[included_spike_pos].astype(np.int32)
    spike_templates = sorting_result[included_spike_pos, 1].astype(np.int32)
    spike_amplitudes = sorting_result[included_spike_pos, 2]

    for s, spike in enumerate(spike_times):
        amplitude = spike_amplitudes[s]
        U_i = U[:, spike_templates[s], :]
        W_i = W[:, spike_templates[s], :]

        addendum = cp.ascontiguousarray(cp.matmul(U_i, W_i.T) * amplitude, dtype=np.int16)

        pred_pos = cp.arange(buffer) + spike - time_limits[0] + buffer + buffer // 2
        predicted_traces[:, pred_pos] += addendum

    output = predicted_traces[:, buffer * 2:-buffer * 2]

    return cp.asnumpy(output).T * 4


class KiloSortWorker(QtCore.QThread):
    foundGoodChannels = QtCore.pyqtSignal(object)
    finishedPreprocess = QtCore.pyqtSignal(object)
    finishedSpikesort = QtCore.pyqtSignal(object)
    finishedAll = QtCore.pyqtSignal(object)

    def __init__(self, context, data_path, output_directory, steps, *args, **kwargs):
        super(KiloSortWorker, self).__init__(*args, **kwargs)
        self.context = context
        self.data_path = data_path
        self.output_directory = output_directory

        assert isinstance(steps, list) or isinstance(steps, str)
        self.steps = steps if isinstance(steps, list) else [steps]

    def run(self):
        if "preprocess" in self.steps:
            self.context = run_preprocess(self.context)
            self.finishedPreprocess.emit(self.context)

        if "spikesort" in self.steps:
            self.context = run_spikesort(self.context)
            self.finishedSpikesort.emit(self.context)

        if "export" in self.steps:
            run_export(self.context, self.data_path, self.output_directory)
            self.finishedAll.emit(self.context)

        if "goodchannels" in self.steps:
            self.context = find_good_channels(self.context)
            self.foundGoodChannels.emit(self.context)
