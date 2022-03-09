import datetime
from pathlib import Path
import os

from CGAN_Patterns.hyperparam import Hyperparam

class Paths():

    def __init__(self, key):

        self.raw_input_path= "raw_input/"


        self.training_runs_path="training_runs/"


        self.synth_patterns_path="synth_output/patterns/"
        self.synth_habits_path = "synth_output/habits/"
        self.synth_aggreg_path = "synth_output/aggregate/"

        self.currentDT = datetime.datetime.now()
        currentDT=self.currentDT
        self.trial = Hyperparam.trial_folder_prefix + str(currentDT.year) + str(currentDT.month) \
                     + str(currentDT.day) + str(currentDT.hour) + str(currentDT.minute)
        trial=self.trial

        if key=='train':
            if not os.path.exists(self.training_runs_path + trial):
                os.makedirs(self.training_runs_path + trial)
            current_dir = self.training_runs_path + trial
            self.current_path = current_dir + "/"
        elif key=='generate':
            if not os.path.exists(self.synth_patterns_path + trial):
                os.makedirs(self.synth_patterns_path + trial)
            current_dir = self.synth_patterns_path + trial
            self.current_path = current_dir + "/"
        else:
            print('Please enter valid key i.e. "train" or "generate"')



    """# Enter Input Folder that contains pattern model"""
    def enter_input_folder(self):
        print('Enter the input folder that contains filtered_patterns.csv :')
        input_folder = input()
        print('Reference Input Files for feature engineering exist in: ' + input_folder)
        return self.training_runs_path + input_folder + "/"