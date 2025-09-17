import json
import time
import math
import random
import io
from sys import platform
import os
import bpy
import configparser

from pathlib import Path
import subprocess

from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

dirname = os.path.dirname(__file__)
class MY_PG_SceneProperties(PropertyGroup):


    my_int: IntProperty(
        name = "Number of moves:",
        description="It is dependent of model",
        default = 5,
        min = 2,
        max = 100
        )


    my_name: StringProperty(
        name="Model name (Train):",
        description="Saves the trained model with the given name",
        default="newModel",
        maxlen=1024,
        )

    my_path: StringProperty(
        name = "Model path (Choose & Generate):",
        description="Choose a model",
        default= dirname,
        maxlen=1024,
        subtype= 'FILE_PATH'
        )
        
    my_op: EnumProperty(
        name="Options:",
        description="Choose one option",
        items=[ ('choose', "Choose", ""),
                ('train', "Train", ""),
                ('generate', "Generate", ""),
              ]
        )
    my_device: EnumProperty(
        name="Device:",
        description="Choose one option",
        items=[ ('cpu', "CPU", ""),
                ('gpu', "CUDA / GPU", ""),
              ]
        )

def moveleg(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, ftime = 0, shift = 0):
    ob = bpy.data.objects['Armature']
    bpy.ops.object.mode_set(mode='POSE')
    namebone = 'leg' + str(boneArr[x])+nameString
    pbone = ob.pose.bones[namebone]
    pbone.rotation_mode = 'XYZ'
    axis = axisArr[x]
    bpy.ops.object.mode_set(mode='OBJECT')
    frame_begin = arr1[x] + ftime + shift
    frame_end = arr2[x] + ftime + shift
    bpy.context.scene.frame_end = frame_end
    angle = 0
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
    bpy.ops.object.mode_set(mode='OBJECT')
    pbone.keyframe_insert(data_path="rotation_euler", frame = frame_begin)        
    angle = angleArr[x]
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
    pbone.keyframe_insert(data_path="rotation_euler", frame = frame_end)
    
def leg(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, ftime, shift):
    for k in range(25):
        for i in range(x):
            moveleg(i, boneArr, angleArr, arr1, arr2, axisArr, nameString, ftime, shift)
        ftime += arr2[len(arr2)-1]

def moving(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, shift):
    leg(x, boneArr, angleArr, arr1, arr2, axisArr, nameString, 0, shift)

class WM_OT_HelloWorld(Operator):
    bl_label = "Run machine learning"
    bl_idname = "wm.hello_world"     
    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool
        print('DIR '+ dirname)
        set_choose = False
        set_train = False
        set_generate = False
        set_moves  = 5
        set_device = 'cpu'
        set_name = 'newModel'
        set_path = dirname+'\models\model.pth'.format(os.getlogin())
        print('Set - '+set_path)
        if mytool.my_op == 'choose':
            set_choose = True
            set_train = False
            set_generate = False
        if mytool.my_op == 'train':
            set_choose = False
            set_train = True
            set_generate = False
        if mytool.my_op == 'generate':
            set_choose = False
            set_train = False
            set_generate = True
        if mytool.my_device == 'cpu':
            set_device = 'cpu'
        if mytool.my_device == 'gpu':
            set_device = 'cuda'
        set_moves  = mytool.my_int
        set_name = mytool.my_name
        set_path = mytool.my_path
        ppath1 = os.path.normpath(os.path.join(dirname, set_path))
        ppath = ppath1[2:]
        print("W - "+dirname+ppath)
        my_setting = configparser.ConfigParser()
        my_setting['SETTING'] = {'choosing': set_choose,
        ';choosing': set_choose,
        'training' : set_train,
        ';training' : set_train,
        'generating' : set_generate,
        ';generating' : set_generate,
        'segments' : set_moves,
        ';segments' : set_moves,
        'device': set_device,
        ';device': set_device,
        'name_model' : set_name,
        ';name_model' : set_name,
        'path_model' : ppath}
        
        conf_path = dirname+'\conf.ini'.format(os.getlogin())
        print(conf_path)
        with open(conf_path, 'w', encoding='utf-8') as configfile:
            my_setting.write(configfile)
        print(os.getlogin())
        if platform == 'linux' or platform == 'linux2':
            os.system('/home/'+str(os.getlogin())+'/.config/blender/4.30/scripts/startup/glue.sh')
        elif platform == 'darwin':
            print('/Users/'+str(os.getlogin())+'/Library/Application Support/Blender/4.30/scripts/startup/glue.zsh')
        elif platform == 'win32' or platform == 'win64':
            print("C:\\Users\\"+str(os.getlogin())+r"\\AppData\\Roaming\\Blender Foundation\\Blender\\4.3\\scripts\\startup\\glue.bat")
            version_tuple = bpy.app.version
            version_str = f"{version_tuple[0]}.{version_tuple[1]}"
            script_path = Path.home() / "AppData" / "Roaming" / "Blender Foundation" / "Blender" / version_str / "scripts" / "startup" / "glue.bat"
            subprocess.run([str(script_path)], shell=True)
            
        if set_choose == True or set_generate == True:     
            for i in range(90000): 
                with open('results.json') as results_json:
                    results = json.load(results_json) 
                if results['angles'] == [0]:
                    time.sleep(0.01)
                if len(results['angles']) > 1:
                    x = len(results['angles'])
                    print(x)
                    boneArr = results['bones']
                    print(boneArr)
                    angleArr = results['angles']
                    print(angleArr)
                    frameBeginArr = results['beginframe']
                    print(frameBeginArr)
                    frameEndArr = results['endframe']
                    print(frameEndArr)
                    axises = results['axises']
                    axisArr = []
                    for j in range(len(axises)):
                        if axises[j] == 50:
                            axisArr.append('X')
                        if axisArr[j] == 10:
                            axisArr.append('Z')
                        if axisArr[j] == -30:
                            axisArr.append('Y')
                    moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'FrontRight', 0)
                    moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'BackLeft', 0)
                    s = int(frameEndArr[len(frameEndArr)-1]/2)
                    moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'BackRight', s)
                    moving(x, boneArr, angleArr, frameBeginArr, frameEndArr, axisArr, 'FrontLeft', s)
                
                    clearjson = {'bones': [0],
                    'angles' : [0],
                    'beginframe' : [0],
                    'endframe' : [0],
                    'axises' : [0],
                    'quality': [0]}
                    with io.open('results.json', 'w', encoding='utf8') as outfile:
                        str_ = json.dumps(clearjson)
                        outfile.write(to_unicode(str_))
                    break
        return {'FINISHED'}

class OBJECT_PT_CustomPanel(Panel):
    bl_label = "Create animation with ML"
    bl_idname = "OBJECT_PT_custom_panel"
    bl_space_type = "VIEW_3D"   
    bl_region_type = "UI"
    bl_category = "ML"
    bl_context = "objectmode"   

    @classmethod
    def poll(self,context):
        return context.object is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        scene = context.scene
        mytool = scene.my_tool

        layout.prop(mytool, "my_op", expand=True)
        layout.prop(mytool, "my_device", expand=True)
        layout.prop(mytool, "my_int")
        layout.prop(mytool, "my_name")
        layout.prop(mytool, "my_path")
        
        layout.separator(factor=1.5)
        layout.operator("wm.hello_world")
        layout.separator()

classes = (
    WM_OT_HelloWorld,
    MY_PG_SceneProperties,
    OBJECT_PT_CustomPanel
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.my_tool = PointerProperty(type=MY_PG_SceneProperties)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.my_tool


if __name__ == "__main__":

    register()
