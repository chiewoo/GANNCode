import numpy as np

def ToGAnnConn(fannConn):
    return np.array(fannConn, dtype=[('from','i'),('to','i'),('weight','f')])
