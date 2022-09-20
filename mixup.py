import cv2
import os
import random
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom

# 处理文件
def mkdir(path):

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
         # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

# 原始图片文件夹路径
#img_path = r"D:\Catalogue\master\generate_data\mitios\VOCdevkit2007\VOC2007\JPEGImages"  
img_path = r"/root/yanjun/faster-rcnn/VOCdevkit/VOC2007/JPEGImages"  
# 原始图片对应的标注文件xml文件夹的路径
xml_path = r"/root/yanjun/faster-rcnn/VOCdevkit/VOC2007/Annotations"  

# mixup的图片文件夹路径
save_path = r"/root/yanjun/faster-rcnn/VOCdevkit/VOC2007/JPEGImages" 
mkdir(save_path)
# mixup的图片对应的标注文件xml的文件夹路径      
save_xml = r"/root/yanjun/faster-rcnn/VOCdevkit/VOC2007/Annotations"        
mkdir(save_xml)

img_names = os.listdir(img_path)
img_num = len(img_names)
#print('img_num:', img_num) 

for imgname in img_names:
    # 第一张图片
    #imgpath = img_path + imgname # 得到图片文件夹中每个文件的路径
    imgpath = os.path.join(img_path, imgname)
    if not imgpath.endswith('jpg'):
        continue
    img = cv2.imread(imgpath) #加载图片
    img_h, img_w = img.shape[0], img.shape[1] # 得到当前图片的尺寸
    #print(img_h,img_w)

    # 第二张图片
    i = random.randint(0, img_num - 1) # 返回任意一个图像的序号
    #print('i:', i)
    #add_path = img_path + img_names[i] # 得到另一个要添加的图片的路径
    add_path = os.path.join(img_path, img_names[i])
    addimg = cv2.imread(add_path) # 读取要添加的图片
    add_h, add_w = addimg.shape[0], addimg.shape[1] # 得到它的尺度
    
    
    # 如果尺度不相等的话进行resize
    if add_h != img_h or add_w != img_w: 
        print('resize!')
        addimg = cv2.resize(addimg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    scale_h, scale_w = img_h / add_h, img_w / add_w # 得到尺寸的比例
    
    lam = np.random.beta(1.5, 1.5) # 返回一个beta的值，大概在0.5左右
    #print(lam)
    mixed_img = lam * img + (1 - lam) * addimg # 对两张图片进行融合
    save_img = os.path.join(save_path, imgname[:-4] + '_mixup_.jpg') # 将原始图片.jpg换成字符串里的内容
    cv2.imwrite(save_img, mixed_img) #  写文件
    #print(save_img) 

    #print(imgname, img_names[i]) # 输出两个图片的名称
    
    if imgname != img_names[i]: # 如果这两张图片不是同一张
        # 得到这两张图片对应的xml路径
        # xmlfile1 = xml_path + imgname[:-4] + '.xml'
        # xmlfile2 = xml_path + img_names[i][:-4] + '.xml'
        xmlfile1 = os.path.join(xml_path, imgname[:-4] + '.xml')
        xmlfile2 = os.path.join(xml_path, img_names[i][:-4] + '.xml')
        #print(xmlfile1,xmlfile2)
        # 读取两张图片对应的xml
        tree1 = ET.parse(xmlfile1)
        tree2 = ET.parse(xmlfile2)
        # 建树之类的常规操作
        doc = xml.dom.minidom.Document()
        root = doc.createElement("annotation")
        doc.appendChild(root)

        for folds in tree1.findall("folder"):
            folder = doc.createElement("folder")
            folder.appendChild(doc.createTextNode(str(folds.text)))
            root.appendChild(folder)
        for filenames in tree1.findall("filename"):
            filename = doc.createElement("filename")
            filename.appendChild(doc.createTextNode(str(filenames.text)))
            root.appendChild(filename)
        for paths in tree1.findall("path"):
            path = doc.createElement("path")
            path.appendChild(doc.createTextNode(str(paths.text)))
            root.appendChild(path)
        for sources in tree1.findall("source"):
            source = doc.createElement("source")
            database = doc.createElement("database")
            database.appendChild(doc.createTextNode(str("Unknow")))
            source.appendChild(database)
            root.appendChild(source)
        for sizes in tree1.findall("size"):
            size = doc.createElement("size")
            width = doc.createElement("width")
            height = doc.createElement("height")
            depth = doc.createElement("depth")
            width.appendChild(doc.createTextNode(str(img_w)))
            height.appendChild(doc.createTextNode(str(img_h)))
            depth.appendChild(doc.createTextNode(str(3)))
            size.appendChild(width)
            size.appendChild(height)
            size.appendChild(depth)
            root.appendChild(size)
        # 这里应该是个名称？
        nodeframe = doc.createElement("frame")
        nodeframe.appendChild(doc.createTextNode(imgname[:-4] + '_mixup_'))

        objects = []
        # 读取目标，将读取到的数据保存到一个字典当中，并用一个数组存储读取到的坐标值
        # 通过循环的方式读取到所有的object
        for obj in tree1.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = obj.find("truncated").text
            obj_struct["difficult"] = obj.find("difficult").text
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                  int(bbox.find("ymin").text),
                                  int(bbox.find("xmax").text),
                                  int(bbox.find("ymax").text)]
            objects.append(obj_struct)
        # 读取要添加的图片当中的目标，将他们都添加到同一个字典当中
        for obj in tree2.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = obj.find("truncated").text
            obj_struct["difficult"] = obj.find("difficult").text          # 有的版本的labelImg改参数为小写difficult
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [int(int(bbox.find("xmin").text) * scale_w),
                                  int(int(bbox.find("ymin").text) * scale_h),
                                  int(int(bbox.find("xmax").text) * scale_w),
                                  int(int(bbox.find("ymax").text) * scale_h)]
            objects.append(obj_struct)
            
        # 将所有的目标框保存到新的文件当中
        for obj in objects:
            nodeobject = doc.createElement("object")
            nodename = doc.createElement("name")
            nodepose = doc.createElement("pose")
            nodetruncated = doc.createElement("truncated")
            nodedifficult = doc.createElement("difficult")
            nodebndbox = doc.createElement("bndbox")
            nodexmin = doc.createElement("xmin")
            nodeymin = doc.createElement("ymin")
            nodexmax = doc.createElement("xmax")
            nodeymax = doc.createElement("ymax")
            nodename.appendChild(doc.createTextNode(obj["name"]))
            nodepose.appendChild(doc.createTextNode(obj["pose"]))
            nodepose.appendChild(doc.createTextNode(obj["truncated"]))
            nodedifficult.appendChild(doc.createTextNode(obj["difficult"]))
            nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
            nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
            nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
            nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

            nodebndbox.appendChild(nodexmin)
            nodebndbox.appendChild(nodeymin)
            nodebndbox.appendChild(nodexmax)
            nodebndbox.appendChild(nodeymax)

            nodeobject.appendChild(nodename)
            nodeobject.appendChild(nodepose)
            nodeobject.appendChild(nodetruncated)
            nodeobject.appendChild(nodedifficult)
            nodeobject.appendChild(nodebndbox)

            root.appendChild(nodeobject)

        #fp = open(save_xml + imgname[:-4] + "_mixup_.xml", "w")
        fp = open(os.path.join(save_xml, imgname[:-4] + "_mixup_.xml"), "w")
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()
    # 如果这两张图片是同一张，那么就只更改文件名就好，其他不用变
    else:
        xmlfile1 = os.path.join(xml_path, imgname[:-4] + '.xml')
        #print(xmlfile1)
        tree1 = ET.parse(xmlfile1)

        doc = xml.dom.minidom.Document()
        root = doc.createElement("annotation")


        doc.appendChild(root)

        for folds in tree1.findall("folder"):
            folder=doc.createElement("folder")
            folder.appendChild(doc.createTextNode(str(folds.text)))
            root.appendChild(folder)
        for filenames in tree1.findall("filename"):
            filename=doc.createElement("filename")
            filename.appendChild(doc.createTextNode(str(filenames.text)))
            root.appendChild(filename)
        for paths in tree1.findall("path"):
            path = doc.createElement("path")
            path.appendChild(doc.createTextNode(str(paths.text)))
            root.appendChild(path)
        for sources in tree1.findall("source"):
            source = doc.createElement("source")
            database = doc.createElement("database")
            database.appendChild(doc.createTextNode(str("Unknow")))
            source.appendChild(database)
            root.appendChild(source)
        for sizes in tree1.findall("size"):
            size = doc.createElement("size")
            width = doc.createElement("width")
            height = doc.createElement("height")
            depth = doc.createElement("depth")
            width.appendChild(doc.createTextNode(str(img_w)))
            height.appendChild(doc.createTextNode(str(img_h)))
            depth.appendChild(doc.createTextNode(str(3)))
            size.appendChild(width)
            size.appendChild(height)
            size.appendChild(depth)
            root.appendChild(size)


        nodeframe = doc.createElement("frame")
        nodeframe.appendChild(doc.createTextNode(imgname[:-4] + '_mixup_'))
        objects = []

        for obj in tree1.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = obj.find("truncated").text
            obj_struct["difficult"] = obj.find("difficult").text
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                  int(bbox.find("ymin").text),
                                  int(bbox.find("xmax").text),
                                  int(bbox.find("ymax").text)]
            objects.append(obj_struct)

        for obj in objects:
            nodeobject = doc.createElement("object")
            nodename = doc.createElement("name")
            nodepose = doc.createElement("pose")
            nodetruncated = doc.createElement("truncated")
            nodedifficult = doc.createElement("difficult")
            nodebndbox = doc.createElement("bndbox")
            nodexmin = doc.createElement("xmin")
            nodeymin = doc.createElement("ymin")
            nodexmax = doc.createElement("xmax")
            nodeymax = doc.createElement("ymax")
            nodename.appendChild(doc.createTextNode(obj["name"]))
            nodepose.appendChild(doc.createTextNode(obj["pose"]))
            nodetruncated.appendChild(doc.createTextNode(obj["truncated"]))
            nodedifficult.appendChild(doc.createTextNode(obj["difficult"]))
            nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
            nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
            nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
            nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

            nodebndbox.appendChild(nodexmin)
            nodebndbox.appendChild(nodeymin)
            nodebndbox.appendChild(nodexmax)
            nodebndbox.appendChild(nodeymax)

            nodeobject.appendChild(nodename)
            nodeobject.appendChild(nodepose)
            nodeobject.appendChild(nodetruncated)
            nodeobject.appendChild(nodedifficult)
            nodeobject.appendChild(nodebndbox)

            root.appendChild(nodeobject)

        #fp = open(save_xml + imgname[:-4] + "_mixup_.xml", "w")
        fp = open(os.path.join(save_xml, imgname[:-4] + "_mixup_.xml"), "w")
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()