import json,time,threading
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'model/environment_config.json').read_text())
MEAN=np.asarray(CFG['mean'],dtype=np.float32)
SCALE=np.asarray(CFG['scale'],dtype=np.float32)
THRESHOLD=float(CFG['int8_threshold'])

class Engine:
    def __init__(self):
        self.lock=threading.Lock()
        self.interpreter=tf.lite.Interpreter(model_path=str(ROOT/'model/environment_model_int8.tflite'),num_threads=1)
        self.interpreter.allocate_tensors()
        self.i=self.interpreter.get_input_details()[0]
        self.o=self.interpreter.get_output_details()[0]
        assert self.i['dtype']==np.int8 and self.o['dtype']==np.int8
        self.keras=None
    def predict(self,features,mode='int8'):
        vals=np.asarray(features,dtype=np.float32)
        if vals.shape!=(3,) or not np.isfinite(vals).all(): raise ValueError('Expect exactly three finite values')
        x=(vals-MEAN)/SCALE
        if mode=='int8':
            iq,iz=self.i['quantization']; oq,oz=self.o['quantization']
            unclip=np.rint(x/iq+iz)
            clipped=bool(np.any((unclip<-128)|(unclip>127)))
            q=np.clip(unclip,-128,127).astype(np.int8).reshape(1,3)
            with self.lock:
                t=time.perf_counter_ns();self.interpreter.set_tensor(self.i['index'],q)
                self.interpreter.invoke()
                output=self.interpreter.get_tensor(self.o['index']).flatten()
                elapsed=(time.perf_counter_ns()-t)/1e6
            y=(output.astype(np.float32)-oz)*oq
            mse=float(np.mean((x-y)**2))
            return {'mode':'int8','status':'INPUT_OUT_OF_RANGE' if clipped else ('ABNORMAL' if mse>THRESHOLD else 'NORMAL'),
                    'score':mse,'threshold':THRESHOLD,'input_clipped':clipped,'inference_ms':elapsed}
        if mode=='float32':
            if self.keras is None:
                with self.lock:
                    if self.keras is None:
                        self.keras=tf.keras.models.load_model(str(ROOT/'model/environment_model_float32.keras'),compile=False)
            with self.lock:
                t=time.perf_counter_ns()
                y=self.keras(np.asarray(x,dtype=np.float32).reshape(1,3),training=False).numpy().flatten()
                elapsed=(time.perf_counter_ns()-t)/1e6
            mse=float(np.mean((x-y)**2))
            # Quantized model's threshold is NOT calibrated for float model!
            return {'mode':'float32','status':'UNTHRESHOLDED','score':mse,'threshold':None,
                    'input_clipped':False,'inference_ms':elapsed}
        raise ValueError('mode must be int8 or float32')
