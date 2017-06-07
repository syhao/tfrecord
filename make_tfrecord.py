import tensorflow as tf
import argparse
import soundfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
import librosa



def read_and_decode(record_queue,batch_size):
    reader=tf.TFRecordReader()
    records=[]
    maxlen=0
    for i in range(batch_size):
        _,serialized_example=reader.read(record_queue)
        features=tf.parse_single_example(serialized_example,features={'wav_raw': tf.FixedLenFeature([], tf.string),'shape': tf.FixedLenFeature([2], tf.int64),} )
        shape = tf.cast(features['shape'], tf.int32)
        get_wave = tf.decode_raw(features['wav_raw'], tf.float32)
    #get_wave.set_shape(features['shape'])
        get_wave = tf.reshape(get_wave,tf.pack([shape[0],shape[1]]))
        if shape[1]>=maxlen:
            maxlen=shape[1]
        records.append(get_wave)
    x = np.zeros(batch_size,shape[0],maxlen)
    for i in range(batch_size):
        x[i,:,:]=records[i]

    return x

def decode_tfrecord(file_path,batch_size=100):
    record_queue=tf.train.string_input_producer([file_path])
    get_wave=read_and_decode(record_queue,batch_size)
    #get_wave = tf.train.shuffle_batch([get_wave],
    #                                    batch_size=1,
    #                                    num_threads=2,
    #                                    capacity=1000 + 3 * 100,
    #                                    min_after_dequeue=100)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    wavbatch = sess.run([get_wave])
    print wavbatch
    # for i in range(batch_size):
    #     print wavbatch[0][i].shape



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(wav_paths,output_name):
    file_list=librosa.util.find_files(wav_paths,ext='wav')
    output=tf.python_io.TFRecordWriter(output_name)
    for file in file_list:
        single_tfrecode(file,output)
    output.close()


def single_tfrecode(file_name, output):
    wav_data = get_spectrogram(file_name)
    print wav_data.shape
    wav_data = wav_data.astype(np.float32)
    wav_raw = wav_data.tostring()
    example=tf.train.Example(features=tf.train.Features(feature={
        'wav_raw': _bytes_feature(wav_raw),
        'shape':tf.train.Feature(int64_list = tf.train.Int64List(value =list(wav_data.shape))),
     }))
    output.write(example.SerializeToString())


def get_spectrogram(file_name, step=10, window=20, max_freq=None, eps=1e-14):
    with soundfile.SoundFile(file_name) as audio_file:
        audio = audio_file.read(dtype='float32')
        samplerate = audio_file.samplerate
        if audio.ndim >= 2:
            np.mean(audio, 1)
        if max_freq is None:
            max_freq = samplerate / 2
        if max_freq > samplerate / 2:
            raise ValueError("max_freq must not greate than half of sample rate")
        if step > window:
            raise ValueError("step must lower than window size")
        hop_length = int(0.001 * step * samplerate)
        fft_length = int(0.001 * window * samplerate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=samplerate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))


def spectrogram(data, fft_length, sample_rate, hop_length):
    assert not np.iscomplexobj(data)
    window = np.hanning(fft_length)[:,None]
    window_nornal = np.sum(window ** 2)
    scale = window_nornal * sample_rate
    trunc_length = (len(data) - fft_length) % hop_length
    x = data[:len(data) - trunc_length]
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)
    assert np.all(x[:, 1] == data[hop_length:(hop_length + fft_length)])
    print(x.shape)
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale
    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])
    return x, freqs


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('wavfiles_path',type=str,help="wav files path to generate tfrecords")
    parser.add_argument('output_tfrecorder_name',type=str,help='file name to save tf record')
    args = parser.parse_args()
    create_tfrecord(args.wavfiles_path,args.output_tfrecorder_name)
    #decode_tfrecord("./records")


