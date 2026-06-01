# Adrian Foy September 2023

"""Interacts with RHD data, both directly at the binary level with RHD data
blocks and at the Python level with dictionaries of NumPy arrays.
"""


import os
import struct

import numpy as np

from intanutil.report import print_record_time_summary, print_progress

LOAD_STIM_DATA = False


def calculate_data_size(header, filename, fid):
    """Calculates how much data is present in this file. Returns:
    data_present: Bool, whether any data is present in file
    filesize: Int, size (in bytes) of file
    num_blocks: Int, number of 60 or 128-sample data blocks present
    num_samples: Int, number of samples present in file
    """
    bytes_per_block = get_bytes_per_data_block(header)

    # Determine filesize and if any data is present.
    filesize = os.path.getsize(filename)
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    # If the file size is somehow different than expected, raise an error.
    if bytes_remaining % bytes_per_block != 0:
        raise FileSizeError(
            'Something is wrong with file size : '
            'should have a whole number of data blocks')

    # Calculate how many data blocks are present.
    num_blocks = int(bytes_remaining / bytes_per_block)

    num_samples = calculate_num_samples(header, num_blocks)

    print_record_time_summary(num_samples,
                              header['sample_rate'],
                              data_present)

    return data_present, filesize, num_blocks, num_samples


def read_all_data_blocks(header, num_samples, num_blocks, fid):
    """Reads all data blocks present in file, allocating memory for and
    returning 'data' dict containing all data.
    """
    data, index = initialize_memory(header, num_samples)
    print("Reading data from file...")
    print_step = 10
    percent_done = print_step
    for i in range(num_blocks):
        read_one_data_block(data, header, index, fid)
        index = advance_index(index, header['num_samples_per_data_block'])
        percent_done = print_progress(i, num_blocks, print_step, percent_done)
    return data


def check_end_of_file(filesize, fid):
    """Checks that the end of the file was reached at the expected position.
    If not, raise FileSizeError.
    """
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining != 0:
        raise FileSizeError('Error: End of file not reached.')


def parse_data(header, data):
    """Parses raw data into user readable and interactable forms (for example,
    extracting raw digital data to separate channels and scaling data to units
    like microVolts, degrees Celsius, or seconds.)
    """
    print('Parsing data...')
    extract_digital_data(header, data)
    if LOAD_STIM_DATA and 'stim_data_raw' in data:
        extract_stim_data(data)
    scale_analog_data(header, data)
    scale_timestamps(header, data)


def data_to_result(header, data, result):
    """Merges data from all present signals into a common 'result' dict. If
    any signal types have been allocated but aren't relevant (for example,
    no channels of this type exist), does not copy those entries into 'result'.
    """
    result['t'] = data['t']
    if 'stim_data' in data:
        result['stim_data'] = data['stim_data']

    if header['dc_amplifier_data_saved']:
        result['dc_amplifier_data'] = data['dc_amplifier_data']

    if header['num_amplifier_channels'] > 0:
        if 'compliance_limit_data' in data:
            result['compliance_limit_data'] = data['compliance_limit_data']
        if 'charge_recovery_data' in data:
            result['charge_recovery_data'] = data['charge_recovery_data']
        if 'amp_settle_data' in data:
            result['amp_settle_data'] = data['amp_settle_data']
        result['amplifier_data'] = data['amplifier_data']

    if header['num_board_adc_channels'] > 0:
        result['board_adc_data'] = data['board_adc_data']

    if header['num_board_dac_channels'] > 0:
        result['board_dac_data'] = data['board_dac_data']

    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_data'] = data['board_dig_in_data']

    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_data'] = data['board_dig_out_data']

    return result


def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 128 sample datablock."""
    # RHS files always have 128 samples per data block.
    # Use this number along with numbers of channels to accrue a sum of how
    # many bytes each data block should contain.
    num_samples_per_data_block = 128

    # Timestamps (one channel always present): Start with 4 bytes per sample.
    bytes_per_block = bytes_per_signal_type(
        num_samples_per_data_block,
        1,
        4)

    # Amplifier data: Add 2 bytes per sample per enabled amplifier channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_amplifier_channels'],
        2)

    # DC Amplifier data (absent if flag was off).
    if header['dc_amplifier_data_saved']:
        bytes_per_block += bytes_per_signal_type(
            num_samples_per_data_block,
            header['num_amplifier_channels'],
            2)

    # Stimulation data: Add 2 bytes per sample per enabled amplifier channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_amplifier_channels'],
        2)

    # Analog inputs: Add 2 bytes per sample per enabled analog input channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_board_adc_channels'],
        2)

    # Analog outputs: Add 2 bytes per sample per enabled analog output channel.
    bytes_per_block += bytes_per_signal_type(
        num_samples_per_data_block,
        header['num_board_dac_channels'],
        2)

    # Digital inputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            num_samples_per_data_block,
            1,
            2)

    # Digital outputs: Add 2 bytes per sample.
    # Note that if at least 1 channel is enabled, a single 16-bit sample
    # is saved, with each bit corresponding to an individual channel.
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block += bytes_per_signal_type(
            num_samples_per_data_block,
            1,
            2)

    return bytes_per_block


def bytes_per_signal_type(num_samples, num_channels, bytes_per_sample):
    """Calculates the number of bytes, per data block, for a signal type
    provided the number of samples (per data block), the number of enabled
    channels, and the size of each sample in bytes.
    """
    return num_samples * num_channels * bytes_per_sample


def read_one_data_block(data, header, index, fid):
    """Reads one 60 or 128 sample data block from fid into data,
    at the location indicated by index."""
    samples_per_block = header['num_samples_per_data_block']

    read_timestamps(fid,
                    data,
                    index,
                    samples_per_block)

    read_analog_signals(fid,
                        data,
                        index,
                        samples_per_block,
                        header)

    read_digital_signals(fid,
                         data,
                         index,
                         samples_per_block,
                         header)


def read_timestamps(fid, data, index, num_samples):
    """Reads timestamps from binary file as a NumPy array, indexing them
    into 'data'.
    """
    start = index
    end = start + num_samples
    format_sign = 'i'
    format_expression = '<' + format_sign * num_samples
    read_length = 4 * num_samples
    data['t'][start:end] = np.array(struct.unpack(
        format_expression, fid.read(read_length)))


def read_analog_signals(fid, data, index, samples_per_block, header):
    """Reads all analog signal types present in RHD files: amplifier_data,
    aux_input_data, supply_voltage_data, temp_sensor_data, and board_adc_data,
    into 'data' dict.
    """

    read_analog_signal_type(fid,
                            data['amplifier_data'],
                            index,
                            samples_per_block,
                            header['num_amplifier_channels'])

    if header['dc_amplifier_data_saved']:
        read_analog_signal_type(fid,
                                data['dc_amplifier_data'],
                                index,
                                samples_per_block,
                                header['num_amplifier_channels'])

    if LOAD_STIM_DATA and 'stim_data_raw' in data:
        read_analog_signal_type(fid,
                                data['stim_data_raw'],
                                index,
                                samples_per_block,
                                header['num_amplifier_channels'])
    else:
        # Skip stimulation payload when not used by downstream pipeline.
        fid.seek(2 * samples_per_block * header['num_amplifier_channels'], 1)

    read_analog_signal_type(fid,
                            data['board_adc_data'],
                            index,
                            samples_per_block,
                            header['num_board_adc_channels'])

    read_analog_signal_type(fid,
                            data['board_dac_data'],
                            index,
                            samples_per_block,
                            header['num_board_dac_channels'])


def read_digital_signals(fid, data, index, samples_per_block, header):
    """Reads all digital signal types present in RHD files: board_dig_in_raw
    and board_dig_out_raw, into 'data' dict.
    """

    read_digital_signal_type(fid,
                             data['board_dig_in_raw'],
                             index,
                             samples_per_block,
                             header['num_board_dig_in_channels'])

    read_digital_signal_type(fid,
                             data['board_dig_out_raw'],
                             index,
                             samples_per_block,
                             header['num_board_dig_out_channels'])


def read_analog_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be an analog signal type within 'data', for example
    data['amplifier_data'] or data['aux_input_data']. Each sample is assumed
    to be of dtype 'uint16'.
    """

    if num_channels < 1:
        return
    end = start + num_samples
    tmp = np.fromfile(fid, dtype='uint16', count=num_samples*num_channels)
    dest[range(num_channels), start:end] = (
        tmp.reshape(num_channels, num_samples))


def read_digital_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be a digital signal type within 'data', either
    data['board_dig_in_raw'] or data['board_dig_out_raw'].
    """

    if num_channels < 1:
        return
    end = start + num_samples
    dest[start:end] = np.array(struct.unpack(
        '<' + 'H' * num_samples, fid.read(2 * num_samples)))


def calculate_num_samples(header, num_data_blocks):
    """Calculates number of samples in file (per channel).
    """
    return int(header['num_samples_per_data_block'] * num_data_blocks)


def initialize_memory(header, num_samples):
    """Pre-allocates NumPy arrays for each signal type that will be filled
    during this read, and initializes index for data access.
    """
    print('\nAllocating memory for data...')
    data = {}

    # Create zero array for timestamps.
    data['t'] = np.zeros(num_samples, dtype=np.int32)

    # Create zero array for amplifier data.
    data['amplifier_data'] = np.zeros(
        [header['num_amplifier_channels'], num_samples], dtype=np.uint16)

    # Create zero array for DC amplifier data.
    if header['dc_amplifier_data_saved']:
        data['dc_amplifier_data'] = np.zeros(
            [header['num_amplifier_channels'], num_samples], dtype=np.uint16)

    # Create zero array for stim data (optional, can be disabled to reduce RAM).
    if LOAD_STIM_DATA:
        data['stim_data_raw'] = np.zeros(
            [header['num_amplifier_channels'], num_samples], dtype=np.uint16)
        data['stim_data'] = np.zeros(
            [header['num_amplifier_channels'], num_samples], dtype=np.int16)

    # Create zero array for board ADC data.
    data['board_adc_data'] = np.zeros(
        [header['num_board_adc_channels'], num_samples], dtype=np.uint16)

    # Create zero array for board DAC data.
    data['board_dac_data'] = np.zeros(
        [header['num_board_dac_channels'], num_samples], dtype=np.uint16)

    # By default, this script interprets digital events (digital inputs
    # and outputs) as booleans. if unsigned int values are preferred
    # (0 for False, 1 for True), replace the 'dtype=np.bool_' argument
    # with 'dtype=np.uint' as shown.
    # The commented lines below illustrate this for digital input data;
    # the same can be done for digital out.

    # data['board_dig_in_data'] = np.zeros(
    #     [header['num_board_dig_in_channels'], num_samples['board_dig_in']],
    #     dtype=np.uint)
    # Create 16-row zero array for digital in data, and 1-row zero array for
    # raw digital in data (each bit of 16-bit entry represents a different
    # digital input.)
    data['board_dig_in_data'] = np.zeros(
        [header['num_board_dig_in_channels'], num_samples],
        dtype=np.bool_)
    data['board_dig_in_raw'] = np.zeros(
        num_samples,
        dtype=np.uint16)

    # Create 16-row zero array for digital out data, and 1-row zero array for
    # raw digital out data (each bit of 16-bit entry represents a different
    # digital output.)
    data['board_dig_out_data'] = np.zeros(
        [header['num_board_dig_out_channels'], num_samples],
        dtype=np.bool_)
    data['board_dig_out_raw'] = np.zeros(
        num_samples,
        dtype=np.uint16)

    # Set index representing position of data (shared across all signal types
    # for RHS file) to 0
    index = 0

    return data, index


def scale_timestamps(header, data):
    """Verifies no timestamps are missing, and scales timestamps to seconds.
    """
    # Check for gaps in timestamps.
    num_gaps = np.sum(np.not_equal(
        data['t'][1:]-data['t'][:-1], 1))
    if num_gaps == 0:
        print('No missing timestamps in data.')
    else:
        print('Warning: {0} gaps in timestamp data found.  '
              'Time scale will not be uniform!'
              .format(num_gaps))

    # Scale time steps (units = seconds).
    data['t'] = data['t'] / header['sample_rate']


def scale_analog_data(header, data):
    """Scales all analog data signal types (amplifier data, stimulation data,
    DC amplifier data, board ADC data, and board DAC data) to suitable
    units (microVolts, Volts, microAmps).
    """
    # Scale amplifier data (units = microVolts).
    # Memory-safe path: avoids a huge intermediate int32 allocation.
    amp = data['amplifier_data'].astype(np.float32, copy=True)
    amp -= np.float32(32768.0)
    amp *= np.float32(0.195)
    data['amplifier_data'] = amp
    if 'stim_data' in data:
        data['stim_data'] = np.multiply(
            header['stim_step_size'],
            data['stim_data'] / 1.0e-6)

    # Scale DC amplifier data (units = Volts).
    if header['dc_amplifier_data_saved']:
        dc = data['dc_amplifier_data'].astype(np.float32, copy=True)
        dc -= np.float32(512.0)
        dc *= np.float32(-0.01923)
        data['dc_amplifier_data'] = dc

    # Scale board ADC data (units = Volts).
    adc = data['board_adc_data'].astype(np.float32, copy=True)
    adc -= np.float32(32768.0)
    adc *= np.float32(312.5e-6)
    data['board_adc_data'] = adc

    # Scale board DAC data (units = Volts).
    dac = data['board_dac_data'].astype(np.float32, copy=True)
    dac -= np.float32(32768.0)
    dac *= np.float32(312.5e-6)
    data['board_dac_data'] = dac


def extract_digital_data(header, data):
    """Extracts digital data from raw (a single 16-bit vector where each bit
    represents a separate digital input channel) to a more user-friendly 16-row
    list where each row represents a separate digital input channel. Applies to
    digital input and digital output data.
    """
    for i in range(header['num_board_dig_in_channels']):
        data['board_dig_in_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_in_raw'],
                (1 << header['board_dig_in_channels'][i]['native_order'])
                ),
            0)

    for i in range(header['num_board_dig_out_channels']):
        data['board_dig_out_data'][i, :] = np.not_equal(
            np.bitwise_and(
                data['board_dig_out_raw'],
                (1 << header['board_dig_out_channels'][i]['native_order'])
                ),
            0)


def extract_stim_data(data):
    """Extracts stimulation data from stim_data_raw and stim_polarity vectors
    to individual lists representing compliance_limit_data,
    charge_recovery_data, amp_settle_data, stim_polarity, and stim_data
    """
    # Interpret 2^15 bit (compliance limit) as True or False.
    data['compliance_limit_data'] = np.bitwise_and(
        data['stim_data_raw'], 32768) >= 1

    # Interpret 2^14 bit (charge recovery) as True or False.
    data['charge_recovery_data'] = np.bitwise_and(
        data['stim_data_raw'], 16384) >= 1

    # Interpret 2^13 bit (amp settle) as True or False.
    data['amp_settle_data'] = np.bitwise_and(
        data['stim_data_raw'], 8192) >= 1

    # Interpret 2^8 bit (stim polarity) as +1 for 0_bit or -1 for 1_bit.
    data['stim_polarity'] = 1 - (2 * (np.bitwise_and(
        data['stim_data_raw'], 256) >> 8))

    # Get least-significant 8 bits corresponding to the current amplitude.
    curr_amp = np.bitwise_and(data['stim_data_raw'], 255)

    # Multiply current amplitude by the correct sign.
    data['stim_data'] = curr_amp * data['stim_polarity']


def advance_index(index, samples_per_block):
    """Advances index used for data access by suitable values per data block.
    """
    # For RHS, all signals sampled at the same sample rate:
    # Index should be incremented by samples_per_block every data block.
    index += samples_per_block
    return index


class FileSizeError(Exception):
    """Exception returned when file reading fails due to the file size
    being invalid or the calculated file size differing from the actual
    file size.
    """
