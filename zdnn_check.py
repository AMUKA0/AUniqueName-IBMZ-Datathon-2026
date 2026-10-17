import tensorflow as tf, numpy as np
m = tf.keras.models.load_model('models/lstm_tmean7.keras')
m(np.zeros((8, 30, 5), 'float32'), training=False)
libs = sorted({l.split()[-1] for l in open('/proc/self/maps') if 'zdnn' in l.lower() or '_z_ops' in l})
print("zDNN-related libraries loaded in this process:")
for l in libs: print("  ", l)
