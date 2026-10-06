# Adrian Foy September 2023
# Updated: vectorized SOS notch (same IIR coefficients as the original loop).

"""Module to apply a notch filter to an input signal"""

import math

import numpy as np
from scipy.signal import sosfilt

from intanutil.report import print_progress


def apply_notch_filter(header, data):
    """Checks header to determine if notch filter should be applied, and if so,
    apply notch filter to all signals in data['amplifier_data'].
    """
    # If data was not recorded with notch filter turned on, return without
    # applying notch filter. Similarly, if data was recorded from Intan RHX
    # software version 3.0 or later, any active notch filter was already
    # applied to the saved data, so it should not be re-applied.
    if (header['notch_filter_frequency'] == 0
            or header['version']['major'] >= 3):
        return

    # Apply notch filter individually to each channel in order
    print('Applying notch filter...')
    print_step = 10
    percent_done = print_step
    fs = float(header['sample_rate'])
    f_notch = float(header['notch_filter_frequency'])
    for i in range(header['num_amplifier_channels']):
        data['amplifier_data'][i, :] = notch_filter(
            data['amplifier_data'][i, :],
            fs,
            f_notch,
            10)

        percent_done = print_progress(i, header['num_amplifier_channels'],
                                      print_step, percent_done)


def notch_filter(signal_in, f_sample, f_notch, bandwidth):
    """Implements a notch filter (e.g., for 50 or 60 Hz) on vector 'signal_in'.

    f_sample = sample rate of data (input Hz or Samples/sec)
    f_notch = filter notch frequency (input Hz)
    bandwidth = notch 3-dB bandwidth (input Hz).  A bandwidth of 10 Hz is
    recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Example:  If neural data was sampled at 30 kSamples/sec
    and you wish to implement a 60 Hz notch filter:

    out = notch_filter(signal_in, 30000, 60, 10);
    """
    x = np.asarray(signal_in, dtype=np.float64).ravel()
    if x.size < 3:
        return np.asarray(signal_in)

    # Same coefficients as the historical sample-wise IIR (calculate_iir_parameters).
    t_step = 1.0 / float(f_sample)
    f_c = float(f_notch) * t_step
    d = math.exp(-2.0 * math.pi * (float(bandwidth) / 2.0) * t_step)
    b = (1.0 + d * d) * math.cos(2.0 * math.pi * f_c)
    a = (1.0 + d * d) / 2.0
    # Transfer function (a0=1):
    # y[n] = a*b2*x[n-2] + a*b1*x[n-1] + a*b0*x[n] - a2*y[n-2] - a1*y[n-1]
    # with b0=1, b1=-2*cos(2πfc), b2=1, a1=-b, a2=d²
    b0 = a * 1.0
    b1 = a * (-2.0 * math.cos(2.0 * math.pi * f_c))
    b2 = a * 1.0
    a1 = -b
    a2 = d * d
    sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)
    return sosfilt(sos, x)


def calculate_iir_parameters(bandwidth, t_step, f_c):
    """Calculates parameters d, b, a0, a1, a2, a, b0, b1, and b2 used for
    IIR filter and return them in a dict.
    """
    parameters = {}
    d = math.exp(-2.0*math.pi*(bandwidth/2.0)*t_step)
    b = (1.0 + d*d) * math.cos(2.0*math.pi*f_c)
    a0 = 1.0
    a1 = -b
    a2 = d*d
    a = (1.0 + d*d)/2.0
    b0 = 1.0
    b1 = -2.0 * math.cos(2.0*math.pi*f_c)
    b2 = 1.0

    parameters['d'] = d
    parameters['b'] = b
    parameters['a0'] = a0
    parameters['a1'] = a1
    parameters['a2'] = a2
    parameters['a'] = a
    parameters['b0'] = b0
    parameters['b1'] = b1
    parameters['b2'] = b2
    return parameters


def calculate_iir(i, signal_in, signal_out, iir_parameters):
    """Calculates a single sample of IIR filter passing signal_in through
    iir_parameters, resulting in signal_out.
    """
    sample = ((
        iir_parameters['a'] * iir_parameters['b2'] * signal_in[i - 2]
        + iir_parameters['a'] * iir_parameters['b1'] * signal_in[i - 1]
        + iir_parameters['a'] * iir_parameters['b0'] * signal_in[i]
        - iir_parameters['a2'] * signal_out[i - 2]
        - iir_parameters['a1'] * signal_out[i - 1])
        / iir_parameters['a0'])

    return sample
