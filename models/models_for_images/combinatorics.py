from coursera_convNet import run_conv_net_2d
# from visualizer import 
import csv
import pandas as pd


filter_list = [8, 16, 32, 64, 128]
unit_list = [32, 64, 128, 256, 512, 1024]
convolution_list = [0, 1, 2, 3]

test_acc_list = []
test_loss_list = []
fit_time_list = []
epoch_list = []
test_filter_list = []
test_unit_list = []
test_convolution_list = []

for f in range(len(filter_list)):   # len(filter_list)
    for u in range(len(unit_list)): # len(unit_list)
        for c in range(len(convolution_list)):  # len(convolution_list)

            test_acc, test_loss, fit_time, n_epochs = run_conv_net_2d(unit_list[u], filter_list[f], convolution_list[c])
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            fit_time_list.append(fit_time)
            epoch_list.append(n_epochs)
            test_filter_list.append(filter_list[f])
            test_unit_list.append(unit_list[u])
            test_convolution_list.append(convolution_list[c])



# with open('computed_values', 'w') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(test_acc_list)
#     wr.writerow(test_loss_list)
#     wr.writerow(fit_time_list)



df = pd.DataFrame.from_dict({'filters':test_filter_list,'units':test_unit_list,'convolutions':test_convolution_list,'accuracy':test_acc_list,'loss':test_loss_list,'time':fit_time_list,'epochs':epoch_list})
df.to_excel('convNet2D_testing.xlsx', header=True, index=False)