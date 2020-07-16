import os
import numpy as np
import tensorflow as tf

from CGAN_Habits.hyperparam import Hyperparam
from CGAN_Habits.plotter import Plotter


class Documenter():

    
    def __init__(self, paths):
        self.paths=paths
        self.readme = open(paths.current_path + "readme.txt", "a")
        self.code_folder = os.path.dirname(paths.code_path)

    def record_header(self, input_folder, datasets):
        readme=self.readme
        readme.write(tf.__version__)
        readme.write("Project Name: GAN for Appliances \n")
        readme.write("Trial ID: " + self.paths.trial + "\n")
        readme.write("Picked Feature: " + Hyperparam.PICKED_FEATURE + "\n")
        readme.write("No. Of Features Considered in GAN-habits:" + str(Hyperparam.features) + "\n")
        readme.write("Main Code Folder: " + self.code_folder + "\n")
        readme.write("Date: " +self.paths.currentDT.strftime("%d-%B-%Y") + "\n")
        readme.write("--------------------------------------- \n")
        readme.write("INPUT REF: " + input_folder + "\n")
        readme.write("---------------------------------------- \n")
        readme.write("Training Data by: Various - Given: Power Every Minute \n")
        readme.write("---------------------------------------- \n")
        readme.write("GAN Type:" + str(Hyperparam.MODE) + "\n")
        readme.write("TF Version: " + str(tf.__version__) + "\n")
        readme.write("---------------------------------------- \n")
        readme.write("---------------------------------------- \n")
        readme.write("Picked Loads for Training: \n")
        readme.write(str(datasets.all_short_names) + "\n")
        readme.write("Samples drawn from distributions:" + str(Hyperparam.SAMPLES) + '\n')


    def record_params(self,preprocessor):
        readme=self.readme
        readme.write("Feature Scaler: MinMax (-1,1) \n")
        X_dim = preprocessor.scaled_X.shape[1]
        y_dim = preprocessor.encoded_y.shape[1]
        num_classes = y_dim
        readme.write("Epochs: " + str(Hyperparam.EPOCHS) + "\n")
        readme.write("Minibatch Size: " + str(Hyperparam.MB_SIZE) + "\n")
        readme.write("Noise: " + str(Hyperparam.NOISE) + "\n")
        readme.write("Noise Dimension: " + str(Hyperparam.Z_DIM) + "\n")
        readme.write("Features: " + str(X_dim) + "\n")
        readme.write("No. of Conditions (Loads): " + str(num_classes) + "\n")
        readme.write("Lambda (Not applied for Vanilla and Logit): " + str(Hyperparam.LAMBDA) + "\n")
        readme.write("Lambda_G (only for regularized Generator): " + str(Hyperparam.LAMBDA_G) + "\n")
        readme.write("Weights Initializer for Disc and Gen: " + str(Hyperparam.WEIGHTS_INIT) + "\n")
        readme.write("--------------------------------------\n")

    def record_plotter(self):
        readme=self.readme
        readme.write("Picked Loads for Plotting and Evaluation: \n")
        readme.write(str(Plotter.PLOT_LOADS) + "\n")
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
