import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
import random
import array
import time
import json
import io
import os
import configparser

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

conf = configparser.ConfigParser()
dirname = os.path.dirname(__file__)
conf.read(dirname+'\\conf.ini','UTF-8')
print("ML"+dirname)
path_to_test = dirname + '\\testing\\test\\'
path_to_model = conf['SETTING']['path_model']
name_new_model = dirname + '\\models\\' + conf['SETTING']['name_model']
print(conf['SETTING']['choosing'])
print(conf['SETTING']['training'])
print(conf['SETTING']['generating'])
print(conf['SETTING']['name_model'])
input_size = int(conf['SETTING']['segments']) * 5  
my_device = conf['SETTING']['device']
if my_device == 'cuda' and torch.cuda.is_available() == False:
    print('ERROR: CUDA not availabel')
    my_device = 'cpu'
print(conf['SETTING']['path_model'])


hidden_size = 64    
output_size = 1     
learning_rate = 0.000001
threshold = 0.25 

class QualityClassifier(nn.Module):
    def __init__(self):
        super(QualityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x.float())
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Obj:
  def __init__(this_obj, data, quality):
    this_obj.data = data
    this_obj.quality = quality
    
def create_obj(namefile):
  json_file = namefile
  with open(json_file) as json_data:
    neurondata = json.load(json_data) 

  boneArr = neurondata['bones']
  angleArr = neurondata['angles']
  frameBeginArr = neurondata['beginframe']
  frameEndArr = neurondata['endframe']
  
  axisArr = []
  axisArr = neurondata['axises']
  i = 0
  numberAxis = []
 
  quality = neurondata['quality']
  for i in range(len(axisArr)):
    #tmpAxis = 50.0
    if axisArr[i] == 'X' or axisArr[i] == 'x':
      numberAxis.append(50)
    if axisArr[i] == 'Z' or axisArr[i] == 'z':
      numberAxis.append(10)
    if axisArr[i] == 'Y' or axisArr[i] == 'y':
      numberAxis.append(-30)
  i = 0
  #inputs = []
  data = []
  data.append(angleArr)
  data.append(numberAxis)
  data.append(frameBeginArr)
  data.append(boneArr)
  data.append(frameEndArr)
  n1 = Obj(data, quality)
  return n1


print(len([name for name in os.listdir(path_to_test) if os.path.isfile(os.path.join(path_to_test, name))]))

nrFile = len([name for name in os.listdir(path_to_test) if os.path.isfile(os.path.join(path_to_test, name))])-1

objects = []

def create_move(phase, ampA, ampB):
    move = [] 
    boneArr = [1,2,1,2,1]
    angleArr = [int(round(ampA*math.cos(1/3*math.pi+phase))),int(round(ampB*math.cos(1*math.pi))),int(round(ampA*math.cos(1*math.pi+phase))),int(round(ampB*math.cos(2*math.pi))),int(round(ampA*math.cos(5/3*math.pi+phase)))]
    frameBeginArr = [0,0,10,20,20]
    frameEndArr = []
    axisArr = [50,50,50,50,50]
    i =0
    
    for a in angleArr:
        beg = frameBeginArr[i]
        if abs(a) <= 10:
            frameEndArr.append(beg+5)
        if abs(a) <= 20 and abs(a) > 10:
            frameEndArr.append(beg+10)
        if abs(a) <= 60 and abs(a) > 20:
            frameEndArr.append(beg+15)
        if abs(a) > 60:
            frameEndArr.append(beg+20)
        i+=1
    move.append(angleArr)
    move.append(axisArr)
    move.append(frameBeginArr)
    move.append(boneArr)
    move.append(frameEndArr)
    n1 = Obj(move, 0)
    print(n1.data)
    return n1
    
def random_move():
    move = [] 
    x = random.randint(1,2)
    boneArr = []
    angleArr = []
    frameBeginArr = []
    frameEndArr = []
    axisArr = []
    if x == 2:
        boneArr = [2,1,2,1,2]
    else:
        boneArr = [1,2,1,2,1]
    angleArr = [random.randint(-120,120), random.randint(-120,120), random.randint(-120,120)]
    angleArr.append(0-angleArr[1])
    angleArr.append(0-angleArr[0]-angleArr[2])
    x1 = random.randint(0,2)*5
    x2 = random.randint(1,3)*5
    x3 = random.randint(3,4)*5
    x4 = random.randint(4,5)*5
    frameBeginArr = [0, x1, x2, x3, x4]
    frameEndArr = [x1+random.randint(1,2)*5, x3+random.randint(1,2)*5, x2+random.randint(1,3)*5, x3+random.randint(3,4)*5, x4+random.randint(3,4)*5]
    axisArr=[50,50,50,50,50]
    """
    frameBeginArr = [0, random.randint(1,3)*5]
    print(frameBeginArr)
    frameBeginArr.append(frameBeginArr[0]+random.randint(1,3)*5)
    if SEGMENTS - 3 > 0:
        for i in range(SEGMENTS - 3):
            if i <= len(frameBeginArr):
                frameBeginArr.append(frameBeginArr[i]+random.randint(1,3)*5)
    
    for i in range(SEGMENTS):
        frameEndArr.append(frameBeginArr[i]+random.randint(1,3)*5)
        axisArr=[50,50,50,50,50]
    """
    move.append(angleArr)
    move.append(axisArr)
    move.append(frameBeginArr)
    move.append(boneArr)
    move.append(frameEndArr)
    n1 = Obj(move, 0)
    print(n1.data)
    return n1
    
def ocena_danych_1(model, obj, proc):
    data_tensor = torch.tensor(obj.data, dtype=torch.short).flatten()
    with torch.no_grad():
        output = model(data_tensor)
        #print(output)
    return output.item() > proc

def create_objects():
    for i in range(nrFile):
      #objects = []
      namefile = path_to_test + 'mech_animation'+str(i)+'.json'
      objects.append(create_obj(namefile))

def prepare_data(objects):
    data = []
    labels = []
    for obj in objects:
        flat_data = torch.tensor([item for sublist in obj.data for item in sublist], dtype=torch.short)
        data.append(flat_data)
        label = 1 if obj.quality >= threshold else 0
        labels.append(label)
    return torch.stack(data), torch.tensor(labels, dtype=torch.float32)

def ocena_danych(model, obj):
    data_tensor = torch.tensor(obj.data, dtype=torch.short).flatten()
    with torch.no_grad():
        output = model(data_tensor)
        #print(output)
    return output.item()

def choose_best():
    model = QualityClassifier()
    weights_only=True
    model.load_state_dict(torch.load(path_to_model,map_location=torch.device(my_device), weights_only=True))
    model.eval()
    create_objects()
    nmb = 0
    blady = 0
    results = []
    goodnum = 0
    for obj in objects:
        wynik = ocena_danych(model, obj)
        results.append(wynik)
        if (obj.quality > threshold):
            goodnum+=1
        print(str(nmb) + ' - ' + str(wynik > threshold)+ ' - ' + str(obj.quality > threshold))
        print(str(nmb) + ' : ' + str(wynik)+ ' : ' + str(obj.quality))
        if (wynik < threshold and obj.quality > threshold) or (wynik > threshold and obj.quality < threshold):
            blady+=1
        nmb+=1
    
    print('Błędy: '+str(blady)+'/'+str(nrFile))
    print('Błędy w procentach: '+str(100 * blady / nrFile)+'%')
    print('Liczba dobrych danych: '+str(goodnum))
    m = max(results)
    bestID = results.index(m)
    best = Obj(objects[bestID].data, objects[bestID].quality)
    print('Indeks: '+str(bestID))
    print(best.data)
    print(best.quality)
    
    resultsjson = {'angles': best.data[0],
                'axises' : best.data[1],
                'beginframe' : best.data[2],
                'bones' : best.data[3],
                'endframe' : best.data[4],
                'quality': best.quality}
    with io.open('results.json', 'w', encoding='utf8') as outfile:
                    str_ = json.dumps(resultsjson)
                    outfile.write(to_unicode(str_))

def train_data():
    create_objects()
    ArrAnaliz = []
    data, labels = prepare_data(objects)
    dataset = TensorDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    model = QualityClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 0
    val_loss = 1
    while val_loss/len(val_loader) > 0.05:
        if epoch > 8000:
            break
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            #optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), labels).item()
        epoch+=1
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        ArrAnaliz.append(val_loss/len(val_loader))
    model.eval()
    print(ArrAnaliz)
    torch.save(model.state_dict(), name_new_model+'.pth')
  
def generate():
    model = QualityClassifier()
    weights_only=True
    model.load_state_dict(torch.load(path_to_model,map_location=torch.device(my_device), weights_only=True))
    model.eval()
    model_watchman = QualityClassifier()
    model_watchman.load_state_dict(torch.load(dirname + '\\models\\'+'watchman.pth',map_location=torch.device("cpu"), weights_only=True))
    model_watchman.eval()
    phase = 1.451
    ind = 0
    objects1 = []
    while ind < 300:
        phase = (1.45 + ind/1000) * math.pi
        ampA = 60
        ampB = 15
        for j in range(5):
            ampA = ampA + 3 * j
            ampB = ampB + 3 * j
            objects1.append(create_move(phase, ampA, ampB))           
        ind+=4
    num = 0
    wynik = True
    for obj in objects1:
        print(obj.data)
        wynik = ocena_danych_1(model, obj, 0.54) and ocena_danych_1(model_watchman, obj, 0.4)
        #wynik = True
        if wynik == True and num !=0:
            rdata = {'angles': obj.data[0],
            'axises' : obj.data[1],
            'beginframe' : obj.data[2],
            'bones' : obj.data[3],
            'endframe' : obj.data[4],
            'quality': obj.quality}
            with io.open('results.json', 'w', encoding='utf8') as outfile:
                str_ = json.dumps(rdata)
                outfile.write(to_unicode(str_))
            
            print(rdata)
            break
        num = num + 1
    if wynik == False:
        for i in range(50000):
            objects1.append(random_move())
        for obj in objects1:
            print(obj.data)
            wynik = ocena_danych_1(model, obj, 0.54) and ocena_danych_1(model_watchman, obj, 0.4)
            
            if wynik == True and num !=0:
                rdata = {'angles': obj.data[0],
                'axises' : obj.data[1],
                'beginframe' : obj.data[2],
                'bones' : obj.data[3],
                'endframe' : obj.data[4],
                'quality': obj.quality}
                with io.open('results.json', 'w', encoding='utf8') as outfile:
                    str_ = json.dumps(rdata)
                    outfile.write(to_unicode(str_))
                
                print(rdata)
                break
            num = num + 1
if conf['SETTING']['choosing'] == 'True':
    choose_best()
if conf['SETTING']['training'] == 'True':
    train_data()
if conf['SETTING']['generating'] == 'True':
    generate()

pass





