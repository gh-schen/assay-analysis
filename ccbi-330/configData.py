import json

class configData():
    def __init__(self, inpath):
        infile = open(inpath, 'r')
        injson = json.load(infile)
        infile.close()

        # file path
        self.feature_path = injson["feature_path"]
        self.count_path = injson["count_path"]
        self.output_prefix = injson["output_prefix"]

        # parameters
        self.total_iterations = injson["total_iterations"]
        self.iteration_start_seed = injson["iteration_start_seed"]
        