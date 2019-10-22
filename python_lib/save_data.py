import numpy as np
import pickle
import bz2

def save_pickle_zip(filename, myobj):
    """
    save object to file using pickle
    
    @param filename: name of destination file
    @type filename: str
    @param myobj: object to save (has to be pickleable)
    @type myobj: obj
    """

    f = bz2.BZ2File(filename, 'wb')

    pickle.dump(myobj, f, protocol=2)
    f.close()



def load_pickle_zip(filename):
    """
    Load from filename using pickle
    
    @param filename: name of file to load from
    @type filename: str
    """

    f = bz2.BZ2File(filename, 'rb')

    myobj = pickle.load(f)
    f.close()
    return myobj
