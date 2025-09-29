# 5G NR PSS Detection Implementation Python
# Based on OpenAirInterface5G pss_nr.c

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate

def load_sigmf_file(filename):
    """Load SigMF file and return samples"""
    # Try interleaved int16 -> complex64
    raw_data = np.fromfile(filename, dtype=np.int16)
    samples = (raw_data[::2] + 1j * raw_data[1::2]).astype(np.complex64) / 32768.0
    print(f"Loaded {len(samples)} samples from {filename}")
    return samples

def generate_pss_nr(N_ID_2):
    """Generate PSS sequence - 3GPP TS 38.211 Section 7.4.2.2"""
    LENGTH_PSS_NR = 127
    if N_ID_2 < 0 or N_ID_2 > 2:
        raise ValueError(f"Invalid N_ID_2: {N_ID_2}")
    
    # Initial m-sequence: {0,1,1,0,1,1,1}
    x = np.zeros(LENGTH_PSS_NR, dtype=int)
    x[:7] = [0,1,1,0,1,1,1]
    for i in range(LENGTH_PSS_NR - 7):
        x[i+7] = (x[i+4] + x[i]) % 2
    
    d_pss = np.zeros(LENGTH_PSS_NR, dtype=int)
    for n in range(LENGTH_PSS_NR):
        m = (n + 43*N_ID_2) % LENGTH_PSS_NR
        d_pss[n] = 1 - 2*x[m]  # Map 0/1 -> 1/-1
    
    return d_pss

def generate_pss_time_domain(N_ID_2, ofdm_symbol_size=512):
    """Generate time-domain PSS"""
    pss_freq = generate_pss_nr(N_ID_2)
    synchro_f = np.zeros(ofdm_symbol_size, dtype=complex)
    k = ofdm_symbol_size//2 - 63  # Center 127 subcarriers
    for i in range(127):
        idx = (k + i) % ofdm_symbol_size
        synchro_f[idx] = pss_freq[i]
    return np.fft.ifft(synchro_f)

def pss_correlation(rxdata, pss_time):
    """Sliding correlation with 4-sample step"""
    search_len = len(rxdata) - len(pss_time) + 1
    corr_vals = []
    positions = []
    for n in range(0, search_len, 4):
        corr_val = np.sum(np.conj(pss_time) * rxdata[n:n+len(pss_time)])
        corr_vals.append(corr_val)
        positions.append(n)
    return np.array(corr_vals), np.array(positions)

def pss_search_time_nr(rxdata, ofdm_symbol_size=512):
    """Detect PSS and find N_ID_2 and peak position"""
    peak_value = 0
    peak_position = 0
    pss_source = 0
    avg = np.zeros(3)
    correlation_data = []

    for pss_index in range(3):
        pss_time = generate_pss_time_domain(pss_index, ofdm_symbol_size)
        corr, positions = pss_correlation(rxdata, pss_time)
        corr_power = np.abs(corr)**2
        correlation_data.append(corr_power)
        avg[pss_index] = np.mean(corr_power)
        max_idx = np.argmax(corr_power)
        max_val = corr_power[max_idx]
        if max_val > peak_value:
            peak_value = max_val
            peak_position = positions[max_idx]
            pss_source = pss_index

    threshold = 5 * avg[pss_source]
    if peak_value < threshold:
        print("PSS detection FAILED")
        return -1, -1, correlation_data

    print(f"PSS DETECTED: N_ID_2 = {pss_source}, position = {peak_position}")
    return peak_position, pss_source, correlation_data

def plot_correlation(correlation_data):
    """Plot time-domain correlation for all N_ID_2"""
    plt.figure(figsize=(12,5))
    for pss_index, corr in enumerate(correlation_data):
        plt.plot(corr, label=f"N_ID_2={pss_index}")
    plt.title("Time-domain PSS Correlation")
    plt.xlabel("Sample index (steps of 4)")
    plt.ylabel("Correlation magnitude")
    plt.grid(True)
    plt.legend()
    plt.show()

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    filename = "/home/ubuntu/Desktop/Python Converter/1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data"
    samples = load_sigmf_file(filename)

    ofdm_symbol_size = 512  # Adjust according to SDR sample rate
    peak_pos, nid2, correlation_data = pss_search_time_nr(samples, ofdm_symbol_size)

    if peak_pos != -1:
        print(f"\n✓ SUCCESS")
        print(f"N_ID_2: {nid2}")
        print(f"Peak position: {peak_pos} samples")
        print(f"Time: {peak_pos/7680000*1000:.3f} ms")
        print(f"PCI = 3 × N_ID(1) + {nid2} (N_ID(1) from SSS decoding)")
    else:
        print("✗ FAILED - No PSS found")

    if correlation_data:
        plot_correlation(correlation_data)
