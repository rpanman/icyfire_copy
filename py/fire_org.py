import numpy as np




def correct_dispersion(pred, integer = True):
    grid = {}
    acc = {}
    index = [1,2,3]
    
    for i in index:
        grid[str(i).format(str(i))] = np.where(np.round(pred) == i)
        if integer == True:
            acc[str(i).format(str(i))] = np.round(pred[grid[str(i)]])
        else:
            acc[str(i).format(str(i))] = pred[grid[str(i)]]
    return grid, acc

def error_dispersion(pred, act, pred1):
    mask = {}
    conf = {}
    comb = ['123','132','213','231','312','321']
    for i in comb:
        predict_mask = np.where(np.logical_and(pred == int(i[0]), act == int(i[1])))
        #actual_mask = np.where(act[predict_mask] == int(i[1]))
        #predict_mask = np.reshape(predict_mask,(np.shape(predict_mask)[1],))
        mask[i[:2].format(i)] = predict_mask[0]
        predict = pred[mask[i[:2]]]
        conf[i[:2].format(i)] = np.reshape(predict, (len(predict),1))
        actual = act[mask[i[:2]]]
        actual = np.reshape(actual, (len(actual),1))
        conf[i[:2].format(i)]= np.hstack((conf[i[:2].format(i)],actual))
    return mask, conf

def multistep(actual):
    mask = np.where(actual == 3)
    acc = np.copy(actual)
    acc[mask] = 2
    return acc
def deletion(data, actual):
    mask = np.where(actual == 1)
    data = np.delete(data, mask, axis = 0)
    acc = np.copy(actual)
    acc = np.delete(acc, mask)
    return data, acc



def scattering(acc, conf):
    data = {}
    error_labels = sorted(conf)
    actual_labels = sorted(acc)
    data['x'] = np.array([])
    data['y'] = np.array([])
    for i  in error_labels:
        data['x'] = np.append(data['x'],(conf[i][:,0]-1))
        data['y'] = np.append(data['y'], (conf[i][:,1] -1))
    for i in actual_labels:
        data['x'] = np.append(data['x'], (acc[i]-1))
        xy = [int(i)] * len(acc[i])
        xy = np.array(xy)
        data['y'] = np.append(data['y'], xy -1)
    return data
