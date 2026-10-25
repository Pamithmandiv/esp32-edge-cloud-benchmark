import os,time
from typing import Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Header,Depends
from pydantic import BaseModel,Field
from .inference import Engine,CFG
engine=None
@asynccontextmanager
async def lifespan(app:FastAPI):
    global engine
    engine=Engine()
    engine.predict([27.5,41.0,83.0]) # warm up model runtime
    yield
app=FastAPI(title='Food Factory Safety Research - Cloud Inference',lifespan=lifespan)

def auth(x_api_key:str|None=Header(default=None)):
    k=os.getenv('API_KEY','')
    if k and x_api_key!=k: raise HTTPException(401,'Invalid API key')
class Request(BaseModel):
    temperature_c:float=Field(ge=-50,le=120)
    relative_humidity_pct:float=Field(ge=0,le=100)
    mq135_raw:float=Field(ge=0,le=4095)
    mode:Literal['int8','float32']='int8'
    source:Literal['DATASET','REAL','SIMULATED_DEMO']='DATASET'
@app.get('/health')
def health():return {'ok':True,'model':'environment_model_int8.tflite','note':'experimental, not a validated safety alarm'}
@app.post('/predict',dependencies=[Depends(auth)])
def predict(req:Request):
    t=time.perf_counter_ns()
    try: r=engine.predict([req.temperature_c,req.relative_humidity_pct,req.mq135_raw],req.mode)
    except ValueError as e: raise HTTPException(400,str(e))
    r.update({'temperature_c':req.temperature_c,'relative_humidity_pct':req.relative_humidity_pct,'mq135_raw':req.mq135_raw,'source':req.source,
              'features':CFG['features'],'server_total_ms':(time.perf_counter_ns()-t)/1e6})
    return r
