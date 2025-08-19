import math
import bpy
import random
import time
from bpy.app.handlers import persistent
import json
import configparser
import statistics
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

#################################################################
#conf_aut = configparser.ConfigParser()
#conf_aut.readfp('C:\\Users\\rafal\\Desktop\\v2\\testing\\conf_aut.ini')
this_path = 'C:\\Users\\rafal\\Desktop\\v3\\testing\\'
conf_aut = open(this_path + 'set.txt','r+')
nr = conf_aut.readline()
conf_aut.close()
json_file = this_path +'\\test\\'+ 'mech_animation'+nr+'.json'
with open(json_file) as json_data:
    dataj = json.load(json_data)
    
x = len(dataj['angles'])
print(x)
boneArr = dataj['bones']
print(boneArr)
angleArr = dataj['angles']
print(angleArr)
frameBeginArr = dataj['beginframe']
print(frameBeginArr)
frameEndArr = dataj['endframe']
print(frameEndArr)
axisArr = dataj['axises']
qua = dataj['quality']


#shift = int(frameEndArr[len(frameEndArr)-1]/2)    
#add_rotation(pbone, axis, angle, frame_begin, frame_end)

def moveleg(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, ftime = 0, shift = 0):
    ob = bpy.data.objects['Armature']
    bpy.ops.object.mode_set(mode='POSE')
    namebone = 'leg' + str(boneArr[x])+nameString
    pbone = ob.pose.bones[namebone]
    pbone.rotation_mode = 'XYZ'
    # select axis in ['X','Y','Z']  <--bone local
    axis = axisArr[x]
    bpy.ops.object.mode_set(mode='OBJECT')
        
    frame_begin = arr1[x] + ftime + shift
    frame_end = arr2[x] + ftime + shift
    bpy.context.scene.frame_end = frame_end
        
    angle = 0
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
    bpy.ops.object.mode_set(mode='OBJECT')
    # Set the keyframe with that location, and which frame.
    pbone.keyframe_insert(data_path="rotation_euler", frame = frame_begin)
        
    angle = angleArr[x]
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
    pbone.keyframe_insert(data_path="rotation_euler", frame = frame_end)
    
def leg(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, ftime, shift):
    for k in range(50):
        for i in range(x):
            moveleg(i, boneArr, angleArr, arr1, arr2, axisArr, nameString, ftime, shift)
        ftime += arr2[len(arr2)-1]

def moving(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, shift):
    leg(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, 0, shift)

loc = []
prev_X = []
prev_Y = []
prev_Z = []
@persistent
def run_after_frame_change(dummy):
    #print("frame_changed")
    if bpy.context.scene.frame_current > frame_end:
        bpy.app.handlers.frame_change_post.remove(run_after_frame_change)
        bpy.ops.screen.animation_cancel()
    if bpy.context.scene.frame_current % 50 == 0 and bpy.context.scene.frame_current >= 150:
        robot = bpy.data.objects['Cylinder.005']
        location = robot.matrix_world.translation
        print(location)
        x,y,z = location
        if abs(x) < 10 and abs(y) < 2 and abs(z) < 5 and z > 0:
            L = 5*(0-x)-abs(3*y*y*y)-abs(3*z*z*z)
            if len(loc) >= 1:
                X = x + abs(prev_X[len(prev_X)-1])
                Y = y - abs(prev_Y[len(prev_Y)-1])
                Z = z - abs(prev_Z[len(prev_Z)-1])
                L = 8*(0-X)-abs(Y*Y)-abs(Z*Z)
                loc.append(L)
                print("1: "+str(L))
            else:
                print("2: "+str(L))
                loc.append(L)
        else:
            L = -4.1
            print("B: "+str(L))
            loc.append(L)
        prev_X.append(x)
        prev_Y.append(y) 
        prev_Z.append(z)
        
    if bpy.context.scene.frame_current > 450:
        print(statistics.median(loc))
        dataj['quality'] = statistics.median(loc)
        nrFile = int(nr)+1
        
        with io.open(this_path+'\\test\\'+'mech_animation'+nr+'.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(dataj,indent = 4, sort_keys = True,separators = (',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))
        """nrFile = int(conf_aut['SETTING']['nr_file'])+1
        #conf_aut.close()
        my_setting['SETTING'] = {'nr_file': nrFile}
        with open('conf_aut.ini', 'w') as conf_aut:
            my_setting.write(conf_aut)
        """
        fp = open(this_path + 'set.txt','w')
        fp.write(str(nrFile))
        fp.close()
        bpy.ops.wm.quit_blender()
    
moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'FrontRight', 0)
moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'BackLeft', 0)
s = int(frameEndArr[len(frameEndArr)-1]/2)
moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'BackRight', s)
moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'FrontLeft', s)

#-------------------------------------play animation----------------------------------------------------
n = 0
frame_start, frame_end = 0, frameEndArr[len(frameEndArr)-1]*25
bpy.ops.screen.animation_play()

bpy.app.handlers.frame_change_post.append(run_after_frame_change)   

