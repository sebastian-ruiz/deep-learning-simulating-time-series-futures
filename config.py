from pathlib import Path
import os
from dotenv import load_dotenv

class Config():

    def __init__(self):
        load_dotenv()
        self.path_root = os.getenv("PATH_ROOT")
        self.path_img = "/assets/images"
        self.path_pgf = "/assets/images/pgf"
        self.path_data = "/assets/data"
        self.path_encoded_data = "/assets/encoded_data"
        self.path_caches = "/data_importer"
        self.path_ae_model = "/autoencoders/saved_model"
        self.path_gan_model = "/gans/saved_model"

    def get_path(self, dir):
        return os.path.abspath(self.path_root + "/" + dir)

    def get_path_img(self):
        return os.path.abspath(self.path_root + self.path_img)

    def get_path_data(self):
        return os.path.abspath(self.path_root + self.path_data)

    def get_path_caches(self, dir="caches"):
        return os.path.abspath(self.path_root + self.path_caches + "/" + dir)

    def get_filepath(self, file_dir, file_name):
        return os.path.abspath(self.path_root + "/" + file_dir + "/" + file_name.replace(" ", "_").lower())

    def get_filepath_img(self, file_name):
        return self.get_filepath(self.path_img, file_name + ".png")

    def get_filepath_pgf(self, file_name):
        return self.get_filepath(self.path_pgf, file_name + ".pgf")

    def get_filepath_data(self, file_name):
        return self.get_filepath(self.path_data, file_name + ".pkl")

    def get_filepath_encoded_data(self, file_name):
        return self.get_filepath(self.path_encoded_data, file_name + ".pkl")

    def get_filepath_ae_model(self, file_name):
        return self.get_filepath(self.path_ae_model, file_name + ".h5")

    def get_filepath_gan_model(self, file_name):
        return self.get_filepath(self.path_gan_model, file_name + ".h5")

    def get_filepath_gan_logs(self, file_name):
        return self.get_filepath('/gans/logs', file_name)

    def get_filepath_autoencoder_logs(self, file_name):
        return self.get_filepath('/autoencoders/logs', file_name)

    def file_exists(self, file_name):
        return os.path.isfile(file_name)