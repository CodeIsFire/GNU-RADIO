import numpy as np
import sigmf
from sigmf import SigMFFile
import struct

def convert_sigmf_to_oai_replay_node(data_path, output_path, block_size=1920, scale_shift=4):
    meta_path = data_path.replace('.sigmf-data', '.sigmf-meta')
    dataset = sigmf.sigmffile.fromfile(meta_path)

    # Add START_INDEX if missing
    dataset.get_captures()[0][SigMFFile.START_INDEX_KEY] = 0

    dtype_str = dataset.get_global_field(SigMFFile.DATATYPE_KEY)
    print(f"SigMF data type: {dtype_str}")

    samples = dataset.read_samples()
    if samples.dtype != np.complex64:
        samples = samples.astype(np.complex64)

    I32 = np.round(samples.real).astype(np.int32)
    Q32 = np.round(samples.imag).astype(np.int32)

    I16 = (I32 >> scale_shift).astype(np.int16)
    Q16 = (Q32 >> scale_shift).astype(np.int16)

    interleaved = np.empty(I16.size * 2, dtype=np.int16)
    interleaved[0::2] = I16
    interleaved[1::2] = Q16

    num_samples = I16.size
    num_blocks = num_samples // block_size
    trimmed = num_blocks * block_size

    print(f"🧱 Total blocks: {num_blocks} with {block_size} samples each")

    with open(output_path, 'wb') as f:
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size

            block_IQ = interleaved[start*2:end*2]  # 2 int16 per sample

            # --- Build header ---
            size = block_size
            nbAnt = 1
            timestamp = i * block_size  # simple linear increment
            option_value = 0
            option_flag = 0

            header = struct.pack('<IIQII', size, nbAnt, timestamp, option_value, option_flag)
            f.write(header)
            f.write(block_IQ.tobytes())

    print(f"✔️ Converted and saved {trimmed} samples ({num_blocks} blocks) to '{output_path}'")

# Example usage
if __name__ == "__main__":
    data_path = '/home/ubuntu/Desktop/Python Converter/1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-data'
    output_path = '/home/ubuntu/Desktop/Python Converter/converted_for_oai.dat'

    convert_sigmf_to_oai_replay_node(data_path, output_path)
