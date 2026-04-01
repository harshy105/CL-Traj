import os

# Set all the paths here

SAVE_PATH = "" # path of training checkpoint, quant eval and qual eval
LOG_DIR = "" # training logs path
data_folder = "" # datasets folder
DATA_SET_NAME = "nuScenes" # selected dataset

if DATA_SET_NAME == "nuScenes":
    EXTRACTED_MAP = os.path.join(data_folder, DATA_SET_NAME, "cities_graph.obj") # complete cities graph
    EXTRACTED_SCENES_DB = os.path.join(data_folder, DATA_SET_NAME, "ExtractedScenes_db") # dynamic contexts
    NUSCENES_PATH = os.path.join(data_folder, DATA_SET_NAME, "raw") # raw nuScenes data
    TARGET_PATH = os.path.join(data_folder, DATA_SET_NAME, "processed") # process nuScenes training data 
    DEEP_SCENARIO_PATH = "" # supress import errors
    TARGET_PATH_DS = "" # supress import errors
elif DATA_SET_NAME == "deep_scenario":
    EXTRACTED_MAP = "" # supress import errors
    EXTRACTED_SCENES_DB = "" # supress import errors
    NUSCENES_PATH = "" # supress import errors
    TARGET_PATH = "" # supress import errors
    DEEP_SCENARIO_PATH = os.path.join(data_folder, DATA_SET_NAME, "raw") # raw DeepScenario data
    TARGET_PATH_DS = os.path.join(data_folder, DATA_SET_NAME, "processed") # process DeepScenario training data
else:
    raise ValueError