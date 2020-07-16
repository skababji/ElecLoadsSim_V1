import os
import numpy as np
import tensorflow as tf


from CGAN_Patterns.hyperparam import Hyperparam

class Documenter():

    
    def __init__(self, paths):
        self.paths=paths
        self.readme= open(paths.current_path + "readme.txt", "a")
        self.code_folder = os.path.dirname(os.getcwd())

    def record_gen_input(self, input_trial_folder):
        readme = self.readme
        readme.write("Project Name: Generating Patterns for appliances \n")
        readme.write("-------------------------------------------- \n")
        readme.write("Patterns Generated based on the model trained in trial: "+input_trial_folder+"\n")

    def record_gen_time(self, time):
        readme=self.readme
        readme.write("-------------------------------------------- \n")
        readme.write("Time consumed for generating requested patterns="+str(time/60)+" seconds \n")

    def record_header(self):
        readme=self.readme
        readme.write("Project Name: GAN for Appliances \n")
        readme.write("Trial ID: " + self.paths.trial + "\n")
        readme.write("Main Code Folder: " + self.code_folder + "\n")
        readme.write("Date: " +self.paths.currentDT.strftime("%d-%B-%Y") + "\n")
        readme.write("--------------------------------------- \n")
    
    def record_datasets(self, datasets):
        readme=self.readme
        readme.write("---------------------------------------- \n")
        for i in range(datasets.n_datasets):
            readme.write("Training Data by: "+ str(datasets.datasets[i].get('data_name')) +"- Raw: Power Sampled Every " + str(datasets.datasets[i].get('sampling_period')) + " Seconds \n")
            readme.write(str(datasets.short_names[i]) + "\n")
            readme.write(str(datasets.long_names[i]) + "\n")
            readme.write(str(datasets.tmplt_threshold[i]) + "\n")
            readme.write("Downsampling: " + str(datasets.datasets_steps[i]) + "\n")
            readme.write("Window (from AMPd): " + str(datasets.datasets_loads_windows[i]) + " power samples i.e. " + str(
                (datasets.datasets_loads_windows[i]* Hyperparam.TS_UNIFIED / 60).astype(int)) + " minutes\n")
            readme.write("---------------------------------------- \n")


    def record_params(self,preprocessor,datasets):
        readme=self.readme
        readme.write("---------------------------------------- \n")
        readme.write("GAN Type:" + str(Hyperparam.MODE) + "\n")
        readme.write("TF Version: " + str(tf.__version__) + "\n")
        readme.write("---------------------------------------- \n")
        readme.write("---------------------------------------- \n")
        readme.write("PARAMETERS: \n")
        readme.write("Unified Granularity:" + str(Hyperparam.TS_UNIFIED) + " seconds \n")
        readme.write("Number of added features to data points:" + str(Hyperparam.ADNTL_FEATURES) + "\n")
        if Hyperparam.ADNTL_FEATURES > 0:
            readme.write("Number of histogram bins:" + str(Hyperparam.HISTOGRAM_BINS) + "\n")
        readme.write("Examples duplication factor:" + str(Hyperparam.RPTD_EXAMPLES) + "\n")
        readme.write("Shift step in repeated examples:" + str(Hyperparam.SHFT_STEPS) + "\n")
        readme.write(
            "Master Window: " + str(datasets.master_window) + " power samples i.e. " + str(
                datasets.master_window * Hyperparam.TS_UNIFIED / 60) + " minutes\n")
        readme.write("Preprocess: " + Hyperparam.APPLIED_CLEANING + "\n")
        readme.write("Weights Initializer for Disc and Gen: " + str(Hyperparam.WEIGHTS_INIT) + "\n")
        
        readme.write("---------------------------------------- \n")
        readme.write("Feature Scaler: "+ preprocessor.X_scaler_type + "\n")
        readme.write("Label/Condition Encoding: "+ preprocessor.y_encoder_type +"\n")
        readme.write("Epochs: " + str(Hyperparam.EPOCHS) + "\n")
        readme.write("Minibatch Size: " + str(Hyperparam.MB_SIZE) + "\n")
        readme.write("Noise: " + str(Hyperparam.NOISE) + "\n")
        readme.write("Noise Dimension: " + str(Hyperparam.Z_DIM) + "\n")
        readme.write("Pattern Dimension: " + str(preprocessor.X_dim) + "\n")
        readme.write("No. of Conditions (Loads): " + str(preprocessor.y_dim) + "\n")
        readme.write("Lambda (Not applied for Vanilla and Logit): " + str(Hyperparam.LAMBDA) + "\n")
        readme.write("Lambda_G (ony for regularized Generator): " + str(Hyperparam.LAMBDA_G) + "\n")
    
    def record_plotter(self, datasets):
        readme=self.readme
        readme.write("Picked Loads for Plotting and Evaluation: \n")
        readme.write(str(datasets.all_short_names) + "\n")
        readme.write("---------------------------------------- \n")
    
    def record_input(self,input):
        readme=self.readme
        readme.write(input)
        readme.write("---------------------------------------- \n")

    def record_eval(self,evaluator):
        readme=self.readme
        readme.write("--------------------------------------\n")
        readme.write("EVALUATOR: \n")
        stringlist = []
        evaluator.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        readme.write(short_model_summary)

        readme.write(" \n")

    def record_cgan_tf_arch(self,cgan_tf):
        readme=self.readme
        readme.write("--------------------------------------\n")
        readme.write("DISCRIMINATOR: \n")
        readme.write("Dropout in Discriminator:" + str(np.round(1 - Hyperparam.D_KEEP_PERC, 2)) + "\n")
        readme.write("No. of steps applied to dicsriminator:" + str(Hyperparam.D_STEPS) + "\n")

        for i in range(int(len(cgan_tf.theta_D) / 2)):
            readme.write("Layer_" + str(i) + ": " + str(cgan_tf.theta_D[i].shape) + "\n")

        readme.write("--------------------------------------\n")
        readme.write("GENERATOR: \n")
        readme.write("Dropout in Generator:" + str(np.round(1 - Hyperparam.G_KEEP_PERC, 2)) + "\n")
        readme.write("No. of steps applied to generator:" + str(Hyperparam.G_STEPS) + "\n")

        for i in range(int(len(cgan_tf.theta_G) / 2)):
            readme.write("Layer_" + str(i) + ": " + str(cgan_tf.theta_G[i].shape) + "\n")

        readme.write("--------------------------------------\n")
    
    def close_file(self):
        readme=self.readme
        readme.write("---------------THE END ----------------- \n")
        readme.close()
