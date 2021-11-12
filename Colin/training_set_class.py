from training_set_creation_funcs import *


class TrainingData:

    def __init__(self, lab_xlsx, sheetname, abf_path):
        self.lab_xslx = lab_xlsx
        self.abf_path = abf_path
        self.sheetname = sheetname

        self.peaks = align_peaks(self.lab_xslx.parse(self.sheetname), self.abf_path)
        self.traces = grab_traces(self.peaks, self.abf_path)




