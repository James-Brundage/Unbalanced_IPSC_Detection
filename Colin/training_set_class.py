from training_set_funcs import *
import pickle as pkl

# labels = pd.read_excel(r'/Users/colinmason/Box/James data.xlsx')
class TrainingData:

    def __init__(self, abf_folder):
        self.abf_folder = abf_folder
    def make_set(self, labels_path):

        print('Reading massive labels file.')
        labels = pd.read_excel(labels_path)
        print('Done.')

        for i in tqdm.tqdm(range(len(os.listdir(self.abf_folder)))):
            abf_name = list(set(labels['ABF File']))[i]

            if os.path.exists(f'/Users/colinmason/Desktop/yorglab/10K_pikl/{abf_name}.pkl'):
                print(f'SKIPPED: {abf_name}')
                continue
            else:
                print(1)
                # pth = os.path.join(self.abf_folder, abf_name)
                # peaks = align_peaks(df=labels, recording_name=abf_name, abf_path=pth)
                # traces = grab_traces(peaks, abf_path=pth, limit=0)
                # pkl_obj = open(f'/Users/colinmason/Desktop/yorglab/10K_pikl/{abf_name}.pkl', 'wb')
                # pkl.dump(traces, pkl_obj)
                # print(f'TO PKL: {abf_name}')

    def erase_contents(self, dir_path):

        paths = []
        for dirpath, _, filenames in os.walk(dir_path):
            for f in filenames:
                paths.append(os.path.abspath(os.path.join(dirpath, f)))

        for file in paths:

            file = open(file, "r+")
            file.truncate(0)
            file.close()




test1 = TrainingData(r'/Users/colinmason/Box/Current 10K')
test1.make_set(r'/Users/colinmason/Box/James data.xlsx')
#
# with open('/Users/colinmason/Desktop/yorglab/10K_pikl/7_31_2019 290 Continuous Export.abf.pkl', 'rb') as f:
#     data = pkl.load(f)
#
# print(type(data))
# print(data)
# print(data.shape)




