import matplotlib.pyplot as plt
import numpy as np

test_accuracy_dict = dict()
test_accuracy_dict['t-product'] = [0.74213836, 0.69811321, 0.72955975, 0.74213836, 0.75471698, 0.72955975,0.75471698, 0.77358491, 0.77358491, 0.76100629, 0.7672956 ,0.77987421, 0.77987421, 0.77987421, 0.77358491, 0.77358491, 0.77358491, 0.77358491]
test_accuracy_dict['c-product'] = [0.71698113, 0.74213836, 0.73584906, 0.69811321, 0.69811321, 0.69811321,0.71698113, 0.71069182, 0.69811321, 0.6918239 , 0.71069182,0.69811321]
test_accuracy_dict['rand. ortho.'] = [0.72327044, 0.69811321, 0.74213836, 0.72327044, 0.72327044, 0.72955975, 0.71698113, 0.6918239 , 0.69811321, 0.73584906, 0.69811321,0.69811321]
test_accuracy_dict['data-dependent'] = [0.71069182, 0.70440252, 0.6918239 , 0.68553459, 0.68553459, 0.67295597, 0.66666667, 0.63522013, 0.65408805, 0.65408805, 0.66666667, 0.67924528]
test_accuracy_dict['haar'] = [0.64779874, 0.68553459, 0.68553459, 0.67295597, 0.67924528, 0.70440252,0.67295597, 0.66666667, 0.66666667, 0.62264151, 0.63522013,0.66666667]
# test_accuracy_dict['haar_banded'] = [0.68553459, 0.6918239 , 0.6918239 , 0.70440252, 0.6918239 ,0.71698113, 0.69811321, 0.64779874, 0.67295597, 0.64779874,0.66666667, 0.64150943]
test_accuracy_dict['facewise'] = [0.66037736, 0.59748428, 0.6163522 , 0.61006289, 0.62264151, 0.6163522, 0.61006289, 0.62264151, 0.59119497, 0.56603774, 0.59748428, 0.59119497]
# test_accuracy_dict['banded'] = [0.57232704, 0.57232704, 0.55345912, 0.57232704, 0.55974843, 0.57861635,0.56603774, 0.55974843, 0.55345912, 0.54716981, 0.55345912, 0.56603774]
test_accuracy_dict['matrix'] = [0.59119497, 0.64779874, 0.66666667, 0.6918239 , 0.67295597,0.66037736, 0.67924528, 0.6918239 , 0.69811321, 0.69811321,0.69811321, 0.74842767, 0.74842767, 0.74842767, 0.72327044, 0.72955975, 0.72327044, 0.71069182]

#%%

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'image.interpolation': None})
plt.rcParams['figure.figsize'] = [10, 7]
plt.rcParams['figure.dpi'] = 200

key_list = list(test_accuracy_dict.keys())
marker_list = ['-o', '-v', '-s', '-p', '-X', '-h', '--']

plt.figure()
for i, key in enumerate(key_list):
    num_basis = len(test_accuracy_dict[key])
    if key == 'matrix':
        plt.plot(np.arange(1, num_basis + 1), test_accuracy_dict[key], '--', color='k', label=key, linewidth=3, markersize=10)
    else:
        plt.plot(np.arange(1, num_basis + 1), test_accuracy_dict[key], '-o', label=key, linewidth=3, markersize=10)

plt.xlabel('number of basis elements')
plt.xticks(np.arange(1, 19))
plt.ylabel('test accuracy')
plt.legend()
plt.savefig('/Users/elizabethnewman/Desktop/' + 'test_accuracy_per_basis.png')
plt.show()
