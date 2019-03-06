from sacred import Experiment
# https://sacred.readthedocs.io/en/latest/configuration.html#prefi
#from sacred.observers import MongoObserver
import os

ex = Experiment('ISBI2012 U-Net')
ex.add_source_file("main.py")
ex.add_source_file("ISBI2012Data.py")
ex.add_source_file("model.py")
ex.add_source_file("dataaug.py")

ex.add_config('config.json')


# if the directory already exists, add a number to it.
# therefore dont overwrite old stuff
@ex.config
def my_config(params):
    if not os.path.exists(params["savedir"]):
        os.makedirs(str(params["savedir"]))
    elif not params["resume"]:
        dirindex = 1
        while os.path.exists(params["savedir"][:-1] + str(dirindex) + "/"):
            dirindex += 1
        params["savedir"] = params["savedir"][:-1] + str(dirindex) + "/"
        os.makedirs(str(params["savedir"]))
    else:
        params["savedir"] = params["resume"][:params["resume"].rfind("/")+1]

    # if not params["evaluate"]:
    #     mongourl = (("mongodb://mongodbconnection")
    #     ex.observers.append(MongoObserver.create(url=mongourl,
    #                                              db_name='dbname'))
